// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The prover on the device: buffers, launches by ABI name, the pipeline,
//! and the per-kernel check.

use std::{collections::HashMap, ffi::c_void, sync::Arc};

use anyhow::{anyhow, Context as _, Result};
use cuda_core::simt::{
    launch_kernel_on_stream, CudaContext, CudaFunction, CudaModule, CudaStream, DeviceBuffer,
    DeviceCopy,
};
use risc0_groth16_core::{
    coeff::GroupedCoeff,
    ec::{Affine, Jacobian},
    field::{Field, Fp, Fr},
    fp2::Fp2,
    prover::{assemble, horner, CoefficientGroups, Msms, Proof, ProveError, Tables},
    zkey::Zkey,
};
use risc0_groth16_oxide::{
    abi,
    pipeline::{reduce_all_windows, sort_all_windows, WINDOW_BITS},
    schedule::{self, Buf, Step, Twiddles},
};

use crate::module::ModuleSource;

/// A `#[repr(C)]` record of `risc0-groth16-core` as plain device data.
/// (`DeviceCopy` is cuda-core's trait and the records are core's types, so
/// neither crate may implement it for the other; this transparent newtype
/// is ours.) The bytes are the record's bytes — the ABI's whole point.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Dev<T>(pub T);

// SAFETY: `Dev<T>` is `repr(transparent)` over a `Copy` type; copying its
// bytes to and from the device is what the ABI defines.
unsafe impl<T: Copy> DeviceCopy for Dev<T> {}

/// A device buffer of records.
pub type Buffer<T> = DeviceBuffer<Dev<T>>;

fn as_dev<T>(s: &[T]) -> &[Dev<T>] {
    // SAFETY: `Dev<T>` is `repr(transparent)` over `T`.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<Dev<T>>(), s.len()) }
}

/// Per-phase wall-clock on stderr when `RISC0_GROTH16_TIMING` is set — what
/// a GPU window should produce besides a total: where the seconds go.
struct Phase {
    enabled: bool,
    start: std::time::Instant,
    last: std::time::Instant,
}

impl Phase {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            enabled: std::env::var_os("RISC0_GROTH16_TIMING").is_some(),
            start: now,
            last: now,
        }
    }

    fn mark(&mut self, what: &str) {
        if self.enabled {
            let now = std::time::Instant::now();
            eprintln!(
                "[groth16-cuda] {what:<28} {:>8.3} s  (t = {:.3} s; device free {})",
                (now - self.last).as_secs_f64(),
                (now - self.start).as_secs_f64(),
                device_free()
            );
            self.last = now;
        }
    }
}

/// The device's free memory as "x.xx GiB", for the phase marks (the arm's
/// footprint beside the other tenant's), or "?" when it cannot be read.
fn device_free() -> String {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: a plain driver query on the thread's current context; the
    // out-pointers are valid for the call.
    let rc = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) };
    if rc == 0 {
        format!(
            "{:.2} GiB of {:.2}",
            free as f64 / 2f64.powi(30),
            total as f64 / 2f64.powi(30)
        )
    } else {
        "?".into()
    }
}

/// One kernel parameter, as the ABI passes it.
#[derive(Clone, Copy)]
enum Arg {
    /// A device pointer.
    Ptr(u64),
    /// A `u32` by value.
    U32(u32),
    /// A field element by value (32 bytes).
    Fr(Fr),
}

fn ptr<T>(b: &Buffer<T>) -> u64 {
    b.cu_deviceptr()
}

/// The ABI's `(ptr, len)` pair for a slice parameter.
fn slice<T>(b: &Buffer<T>) -> [Arg; 2] {
    [Arg::Ptr(ptr(b)), Arg::U32(b.len() as u32)]
}

/// The twiddle tables and coset shift powers on the device.
struct TransformBuffers {
    forward: Buffer<Fr>,
    inverse: Buffer<Fr>,
    shift: Buffer<Fr>,
}

/// What the coset transform needs besides its two buffers.
struct CosetTables<'a> {
    n_inv: Fr,
    lg_n: u32,
    tb: &'a TransformBuffers,
}

/// One GPU, one stream, the loaded device module, every kernel resolved.
pub struct CudaProver {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    #[allow(dead_code)] // keeps the module alive for as long as its functions
    module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
    source: String,
}

impl CudaProver {
    /// Initialise the driver, take device 0, load the module, resolve every
    /// kernel of [`abi::KERNELS`] (a module missing one fails here).
    pub fn new(source: ModuleSource) -> Result<Self> {
        // SAFETY: cuInit(0) is the documented first call; idempotent.
        unsafe { cuda_core::init(0) }.map_err(|e| anyhow!("cuInit: {e:?}"))?;
        let ctx = CudaContext::new(0).map_err(|e| anyhow!("CUDA device 0: {e:?}"))?;
        let stream = ctx.default_stream();
        let module = source.load(&ctx)?;
        let mut functions = HashMap::with_capacity(abi::KERNELS.len());
        for &name in abi::KERNELS {
            let f = module
                .load_function(name)
                .map_err(|e| anyhow!("kernel `{name}` is not in the device module: {e:?}"))?;
            functions.insert(name, f);
        }
        Ok(Self {
            ctx,
            stream,
            module,
            functions,
            source: source.describe(),
        })
    }

    /// The device's name, for reports.
    pub fn device_name(&self) -> String {
        self.ctx
            .device_name()
            .unwrap_or_else(|e| format!("<unknown: {e:?}>"))
    }

    /// Where the module came from, for reports.
    pub fn module_source(&self) -> &str {
        &self.source
    }

    fn upload<T: Copy>(&self, data: &[T]) -> Result<Buffer<T>> {
        DeviceBuffer::from_host(&self.stream, as_dev(data))
            .map_err(|e| anyhow!("uploading {} records: {e:?}", data.len()))
    }

    fn alloc<T: Copy>(&self, n: usize) -> Result<Buffer<T>> {
        DeviceBuffer::zeroed(&self.stream, n).map_err(|e| anyhow!("allocating {n} records: {e:?}"))
    }

    fn read<T: Copy>(&self, b: &Buffer<T>) -> Result<Vec<T>> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("stream synchronize: {e:?}"))?;
        let v = b
            .to_host_vec(&self.stream)
            .map_err(|e| anyhow!("reading {} records: {e:?}", b.len()))?;
        Ok(v.into_iter().map(|d| d.0).collect())
    }

    /// Launch `kernel` for `n` outputs into `out`, with the ABI's parameter
    /// list: `out, n, inputs…`. Every parameter is staged in its own 32-byte,
    /// 8-aligned slot and passed by pointer to `cuLaunchKernel`, which reads
    /// each parameter's declared size from the start of its slot (a `u32`
    /// occupies the low four bytes on this little-endian host).
    fn run(&self, kernel: &str, out: u64, n: usize, inputs: &[Arg]) -> Result<()> {
        let n32 = u32::try_from(n).context("launch larger than u32::MAX outputs")?;
        let grid = abi::blocks(n32);
        if grid == 0 {
            return Ok(());
        }
        let f = self
            .functions
            .get(kernel)
            .ok_or_else(|| anyhow!("kernel `{kernel}` is not in the ABI"))?;
        let mut slots: Vec<[u64; 4]> = Vec::with_capacity(inputs.len() + 2);
        for a in [Arg::Ptr(out), Arg::U32(n32)].iter().chain(inputs) {
            let mut s = [0u64; 4];
            match *a {
                Arg::Ptr(p) => s[0] = p,
                Arg::U32(x) => s[0] = u64::from(x),
                // SAFETY: `Fr` is `repr(C)` `[u64; 4]`; the slot is 32 bytes, 8-aligned.
                Arg::Fr(v) => unsafe { std::ptr::write(s.as_mut_ptr().cast::<Fr>(), v) },
            }
            slots.push(s);
        }
        let mut params: Vec<*mut c_void> = slots
            .iter_mut()
            .map(|s| s.as_mut_ptr().cast::<c_void>())
            .collect();
        // SAFETY: the function belongs to this context; `params` has one
        // pointer per declared parameter, each to storage that outlives the
        // call (the driver copies parameters at launch).
        unsafe {
            launch_kernel_on_stream(
                f,
                (grid, 1, 1),
                (abi::BLOCK, 1, 1),
                0,
                &self.stream,
                &mut params,
            )
        }
        .map_err(|e| anyhow!("launching `{kernel}` for {n} outputs: {e:?}"))
    }

    /// The coset transform, executing [`schedule::coset`] over `a` and `b`:
    /// the schedule names the buffer each launch reads and writes and the
    /// one holding the result.
    fn h_to_coset<'b>(
        &self,
        a: &'b Buffer<Fr>,
        b: &'b Buffer<Fr>,
        n: usize,
        coset: &CosetTables<'_>,
    ) -> Result<&'b Buffer<Fr>> {
        let (tb, n_inv, lg_n) = (coset.tb, coset.n_inv, coset.lg_n);
        let pick = |which: Buf| match which {
            Buf::A => a,
            Buf::B => b,
        };
        let sched = schedule::coset(n as u32);
        for step in &sched.steps {
            let (src, dst) = (pick(step.src()), pick(step.dst()));
            match *step {
                Step::BitReverse { .. } => self.run(
                    abi::BIT_REVERSE,
                    ptr(dst),
                    n,
                    &[&slice(src)[..], &[Arg::U32(lg_n)]].concat(),
                )?,
                Step::NttStage {
                    len,
                    stride,
                    twiddles,
                    ..
                } => {
                    let tw = match twiddles {
                        Twiddles::Inverse => &tb.inverse,
                        Twiddles::Forward => &tb.forward,
                    };
                    self.run(
                        abi::NTT_STAGE,
                        ptr(dst),
                        n,
                        &[
                            &slice(src)[..],
                            &[Arg::U32(len)],
                            &slice(tw),
                            &[Arg::U32(stride)],
                        ]
                        .concat(),
                    )?
                }
                Step::Scale { .. } => self.run(
                    abi::POINTWISE_SCALE,
                    ptr(dst),
                    n,
                    &[&slice(src)[..], &slice(&tb.shift), &[Arg::Fr(n_inv)]].concat(),
                )?,
            }
        }
        Ok(pick(sched.result))
    }

    /// One MSM over every window at once: one `digits_all` launch, the host
    /// sort (`pipeline::sort_all_windows`), one `bucket_sum` launch over
    /// `windows · buckets` outputs, then the reduction and Horner on the host
    /// — two launches and two reads per MSM instead of two per window.
    fn msm<F: Field + Copy>(
        &self,
        kernel: &str,
        points: &Buffer<Affine<F>>,
        scalars: &[Fr],
    ) -> Result<Jacobian<F>> {
        assert_eq!(points.len(), scalars.len(), "MSM point and scalar counts");
        let n = scalars.len();
        let w = WINDOW_BITS;
        let buckets = (1usize << w) - 1;
        let windows = 256u32.div_ceil(w) as usize;
        let canonical: Vec<[u64; 4]> = scalars.iter().map(Fr::to_canonical).collect();
        let canonical_b = self.upload(&canonical)?;
        let digits_b: Buffer<u32> = self.alloc(windows * n)?;
        self.run(
            abi::DIGITS_ALL,
            ptr(&digits_b),
            windows * n,
            &[&slice(&canonical_b)[..], &[Arg::U32(w)]].concat(),
        )?;
        let digits = self.read(&digits_b)?;
        drop((digits_b, canonical_b));
        let mut phase = Phase::new();
        let (order, starts) = sort_all_windows(&digits, n, buckets);
        phase.mark("  msm: host sort");
        if order.is_empty() {
            return Ok(Jacobian::INFINITY);
        }
        let order_b = self.upload(&order)?;
        let starts_b = self.upload(&starts)?;
        let sums_b: Buffer<Jacobian<F>> = self.alloc(windows * buckets)?;
        self.run(
            kernel,
            ptr(&sums_b),
            windows * buckets,
            &[&slice(points)[..], &slice(&order_b), &slice(&starts_b)].concat(),
        )?;
        let sums = self.read(&sums_b)?;
        phase.mark("  msm: bucket sums + read");
        let result = horner(&reduce_all_windows(&sums, buckets), w);
        phase.mark("  msm: reduce + Horner");
        Ok(result)
    }

    /// [`Self::prepare`] for a zkey the caller owns: every point set and the
    /// coefficient groups are uploaded and DROPPED one after another, so the
    /// parsed and the uploaded copies never coexist in host memory (the
    /// production zkey parses to ≈ 4 GB; the backends own it, so they use
    /// this).
    pub fn prepare_owned(
        &self,
        mut zkey: Zkey,
        mut groups: CoefficientGroups,
    ) -> Result<ResidentZkey> {
        let n = zkey.domain_size;
        let t = Tables::new(n);
        let tb = TransformBuffers {
            forward: self.upload(&t.forward)?,
            inverse: self.upload(&t.inverse)?,
            shift: self.upload(&t.shift_powers)?,
        };
        let meta = strip(&zkey);
        let a_constraint = std::mem::take(&mut groups.a_constraint);
        let b_constraint = std::mem::take(&mut groups.b_constraint);
        let ca = self.upload(&std::mem::take(&mut groups.a))?;
        let sa = self.upload(&std::mem::take(&mut groups.a_starts))?;
        let cb = self.upload(&std::mem::take(&mut groups.b))?;
        let sb = self.upload(&std::mem::take(&mut groups.b_starts))?;
        drop(groups);
        let pa = self.upload(&std::mem::take(&mut zkey.a))?;
        let pb1 = self.upload(&std::mem::take(&mut zkey.b1))?;
        let pc = self.upload(&std::mem::take(&mut zkey.c))?;
        let ph = self.upload(&std::mem::take(&mut zkey.h))?;
        let pb2 = self.upload(&std::mem::take(&mut zkey.b2))?;
        drop(zkey);
        Ok(ResidentZkey {
            meta,
            a_constraint,
            b_constraint,
            ca,
            sa,
            cb,
            sb,
            n_inv: t.n_inv,
            lg_n: t.lg_n,
            tb,
            pa,
            pb1,
            pc,
            ph,
            pb2,
        })
    }

    /// Produce a proof: the arm's implementation of the boundary.
    pub fn prove(&self, zkey: &Zkey, witness: &[Fr], r: &Fr, s: &Fr) -> Result<Proof> {
        let groups = CoefficientGroups::from_zkey(zkey);
        self.prove_grouped(zkey, &groups, witness, r, s)
    }

    /// [`Self::prove`] with the coefficient groups already built: upload the
    /// zkey for this one proof and drop it after.
    pub fn prove_grouped(
        &self,
        zkey: &Zkey,
        groups: &CoefficientGroups,
        witness: &[Fr],
        r: &Fr,
        s: &Fr,
    ) -> Result<Proof> {
        let resident = self.prepare(zkey, groups)?;
        self.prove_resident(&resident, witness, r, s)
    }

    /// Upload everything of a zkey that every proof reads — the five point
    /// sets, the grouped coefficients and their group starts, the NTT tables
    /// — once, and keep it on the device (DEF-G16-014). The canonical path
    /// re-maps and re-uploads the 3.45 GiB zkey on every call; a resident
    /// zkey costs one upload per process (≈ 4.6 GB on the production circuit,
    /// [`ResidentZkey::device_bytes`]) and per-proof traffic drops to the
    /// witness in and the proof out.
    pub fn prepare(&self, zkey: &Zkey, groups: &CoefficientGroups) -> Result<ResidentZkey> {
        let n = zkey.domain_size;
        let t = Tables::new(n);
        let tb = TransformBuffers {
            forward: self.upload(&t.forward)?,
            inverse: self.upload(&t.inverse)?,
            shift: self.upload(&t.shift_powers)?,
        };
        Ok(ResidentZkey {
            meta: strip(zkey),
            a_constraint: groups.a_constraint.clone(),
            b_constraint: groups.b_constraint.clone(),
            ca: self.upload(&groups.a)?,
            sa: self.upload(&groups.a_starts)?,
            cb: self.upload(&groups.b)?,
            sb: self.upload(&groups.b_starts)?,
            n_inv: t.n_inv,
            lg_n: t.lg_n,
            tb,
            pa: self.upload(&zkey.a)?,
            pb1: self.upload(&zkey.b1)?,
            pc: self.upload(&zkey.c)?,
            ph: self.upload(&zkey.h)?,
            pb2: self.upload(&zkey.b2)?,
        })
    }

    /// Produce a proof against a resident zkey: the witness goes up, the
    /// proof comes back; nothing of the zkey moves.
    pub fn prove_resident(
        &self,
        z: &ResidentZkey,
        witness: &[Fr],
        r: &Fr,
        s: &Fr,
    ) -> Result<Proof> {
        let zkey = &z.meta;
        check_witness(zkey, witness)?;
        let n = zkey.domain_size;
        let mut phase = Phase::new();
        let witness_b = self.upload(witness)?;
        phase.mark("witness upload");
        let a_poly = self.scatter_phase(&z.ca, &z.sa, &z.a_constraint, &witness_b, n)?;
        let b_poly = self.scatter_phase(&z.cb, &z.sb, &z.b_constraint, &witness_b, n)?;
        phase.mark("scatter A, B (+ placement)");
        let coset = CosetTables {
            n_inv: z.n_inv,
            lg_n: z.lg_n,
            tb: &z.tb,
        };
        let quotient = self.transform_phase(&a_poly, &b_poly, n, &coset, &mut phase)?;
        let private = &witness[zkey.num_public + 1..];
        let h = self.msm(abi::BUCKET_SUM_G1, &z.ph, &quotient)?;
        phase.mark("MSM h (G1)");
        let a = self.msm(abi::BUCKET_SUM_G1, &z.pa, witness)?;
        phase.mark("MSM a (G1)");
        let b1 = self.msm(abi::BUCKET_SUM_G1, &z.pb1, witness)?;
        phase.mark("MSM b1 (G1)");
        let b2 = self.msm::<Fp2>(abi::BUCKET_SUM_G2, &z.pb2, witness)?;
        phase.mark("MSM b2 (G2)");
        let c = self.msm(abi::BUCKET_SUM_G1, &z.pc, private)?;
        phase.mark("MSM c (G1)");
        let msms = Msms { h, a, b1, b2, c };
        let proof = assemble(zkey, &msms, r, s);
        phase.mark("assembly");
        Ok(proof)
    }

    /// Produce a proof with NOTHING resident: every phase uploads what it
    /// needs and frees it before the next — the smallest device footprint
    /// (≈ 2.6 GB on the production circuit against ≈ 8 GB resident), for a
    /// device shared with another tenant; the price is re-uploading the zkey
    /// per proof (≈ 5 GB, a fraction of a second over PCIe). The backend
    /// selects it with `RISC0_GROTH16_RESIDENT=0` (W-04's mitigation).
    pub fn prove_streaming(
        &self,
        zkey: &Zkey,
        groups: &CoefficientGroups,
        witness: &[Fr],
        r: &Fr,
        s: &Fr,
    ) -> Result<Proof> {
        check_witness(zkey, witness)?;
        let n = zkey.domain_size;
        let t = Tables::new(n);
        let mut phase = Phase::new();
        let witness_b = self.upload(witness)?;
        phase.mark("witness upload");
        let a_poly = {
            let (cb, sb) = (self.upload(&groups.a)?, self.upload(&groups.a_starts)?);
            self.scatter_phase(&cb, &sb, &groups.a_constraint, &witness_b, n)?
        };
        let b_poly = {
            let (cb, sb) = (self.upload(&groups.b)?, self.upload(&groups.b_starts)?);
            self.scatter_phase(&cb, &sb, &groups.b_constraint, &witness_b, n)?
        };
        phase.mark("scatter A, B (+ placement, streamed)");
        let quotient = {
            let tb = TransformBuffers {
                forward: self.upload(&t.forward)?,
                inverse: self.upload(&t.inverse)?,
                shift: self.upload(&t.shift_powers)?,
            };
            let coset = CosetTables {
                n_inv: t.n_inv,
                lg_n: t.lg_n,
                tb: &tb,
            };
            self.transform_phase(&a_poly, &b_poly, n, &coset, &mut phase)?
        };
        drop((a_poly, b_poly, t));
        let private = &witness[zkey.num_public + 1..];
        let msm_g1 = |points: &[Affine<Fp>], scalars: &[Fr]| -> Result<Jacobian<Fp>> {
            let pb = self.upload(points)?;
            self.msm(abi::BUCKET_SUM_G1, &pb, scalars)
        };
        let h = msm_g1(&zkey.h, &quotient)?;
        phase.mark("MSM h (G1, streamed)");
        let a = msm_g1(&zkey.a, witness)?;
        phase.mark("MSM a (G1, streamed)");
        let b1 = msm_g1(&zkey.b1, witness)?;
        phase.mark("MSM b1 (G1, streamed)");
        let b2 = {
            let pb2 = self.upload(&zkey.b2)?;
            self.msm::<Fp2>(abi::BUCKET_SUM_G2, &pb2, witness)?
        };
        phase.mark("MSM b2 (G2, streamed)");
        let c = msm_g1(&zkey.c, private)?;
        phase.mark("MSM c (G1, streamed)");
        let msms = Msms { h, a, b1, b2, c };
        let proof = assemble(zkey, &msms, r, s);
        phase.mark("assembly");
        Ok(proof)
    }

    /// Scatter one matrix: group sums on the device, placed at their
    /// constraint indices on the host.
    fn scatter_phase(
        &self,
        cb: &Buffer<GroupedCoeff>,
        sb: &Buffer<u32>,
        cons: &[u32],
        witness_b: &Buffer<Fr>,
        n: usize,
    ) -> Result<Vec<Fr>> {
        let out: Buffer<Fr> = self.alloc(cons.len())?;
        self.run(
            abi::SCATTER_GROUP,
            ptr(&out),
            cons.len(),
            &[&slice(cb)[..], &slice(sb), &slice(witness_b)].concat(),
        )?;
        let sums = self.read(&out)?;
        let mut poly = vec![Fr::ZERO; n];
        for (c, v) in cons.iter().zip(sums) {
            poly[*c as usize] = v;
        }
        Ok(poly)
    }

    /// From the A and B polynomials to the quotient's coset evaluations:
    /// C = A∘B, three coset transforms, the pointwise quotient. Four
    /// polynomial buffers at most: each transform's free ping-pong buffer is
    /// the next one's scratch, and the last free one holds the quotient —
    /// 1 GB of polynomials on the production circuit instead of 1.8 GB.
    fn transform_phase(
        &self,
        a_poly: &[Fr],
        b_poly: &[Fr],
        n: usize,
        coset: &CosetTables<'_>,
        phase: &mut Phase,
    ) -> Result<Vec<Fr>> {
        let (a1, b1) = (self.upload(a_poly)?, self.upload(b_poly)?);
        let c1 = self.alloc::<Fr>(n)?;
        self.run(
            abi::POINTWISE_MUL,
            ptr(&c1),
            n,
            &[&slice(&a1)[..], &slice(&b1)].concat(),
        )?;
        phase.mark("polynomial uploads, C = A∘B");
        let scratch = self.alloc::<Fr>(n)?;
        // each transform returns (its result, the buffer it left free)
        let (ac, free) = self.coset_pair(a1, scratch, n, coset)?;
        let (bc, free) = self.coset_pair(b1, free, n, coset)?;
        let (cc, q) = self.coset_pair(c1, free, n, coset)?;
        self.run(
            abi::POINTWISE_MUL_SUB,
            ptr(&q),
            n,
            &[&slice(&ac)[..], &slice(&bc), &slice(&cc)].concat(),
        )?;
        let quotient = self.read(&q)?;
        phase.mark("3 coset transforms, quotient");
        Ok(quotient)
    }

    /// The coset transform of `a` with `b` as scratch: the buffer holding the
    /// result and the one left free, by the schedule's word.
    fn coset_pair(
        &self,
        a: Buffer<Fr>,
        b: Buffer<Fr>,
        n: usize,
        coset: &CosetTables<'_>,
    ) -> Result<(Buffer<Fr>, Buffer<Fr>)> {
        let result_is_a = {
            let r = self.h_to_coset(&a, &b, n, coset)?;
            std::ptr::eq(r, &a)
        };
        Ok(if result_is_a { (a, b) } else { (b, a) })
    }
}

/// The witness has the circuit's length and the constant 1 at index 0.
fn check_witness(zkey: &Zkey, witness: &[Fr]) -> Result<()> {
    if witness.len() != zkey.num_vars {
        return Err(ProveError::WitnessLength {
            expected: zkey.num_vars,
            found: witness.len(),
        }
        .into());
    }
    if witness[0] != Fr::ONE {
        return Err(ProveError::WitnessConstant.into());
    }
    Ok(())
}

/// A zkey uploaded once and kept on the device across proofs: the five point
/// sets, the grouped coefficients with their group starts, the NTT tables.
/// The host keeps only what assembly and the scatter placement need.
pub struct ResidentZkey {
    /// Sizes and the verifying key; every point vector empty.
    meta: Zkey,
    a_constraint: Vec<u32>,
    b_constraint: Vec<u32>,
    ca: Buffer<GroupedCoeff>,
    sa: Buffer<u32>,
    cb: Buffer<GroupedCoeff>,
    sb: Buffer<u32>,
    n_inv: Fr,
    lg_n: u32,
    tb: TransformBuffers,
    pa: Buffer<Affine<Fp>>,
    pb1: Buffer<Affine<Fp>>,
    pc: Buffer<Affine<Fp>>,
    ph: Buffer<Affine<Fp>>,
    pb2: Buffer<Affine<Fp2>>,
}

impl ResidentZkey {
    /// Bytes held on the device.
    pub fn device_bytes(&self) -> usize {
        self.ca.num_bytes()
            + self.sa.num_bytes()
            + self.cb.num_bytes()
            + self.sb.num_bytes()
            + self.tb.forward.num_bytes()
            + self.tb.inverse.num_bytes()
            + self.tb.shift.num_bytes()
            + self.pa.num_bytes()
            + self.pb1.num_bytes()
            + self.pc.num_bytes()
            + self.ph.num_bytes()
            + self.pb2.num_bytes()
    }

    /// The circuit's variable count.
    pub fn num_vars(&self) -> usize {
        self.meta.num_vars
    }

    /// The circuit's public input count.
    pub fn num_public(&self) -> usize {
        self.meta.num_public
    }
}

/// The zkey without its point and coefficient vectors: what `assemble` and
/// the size checks read.
fn strip(zkey: &Zkey) -> Zkey {
    Zkey {
        num_vars: zkey.num_vars,
        num_public: zkey.num_public,
        domain_size: zkey.domain_size,
        vk: zkey.vk.clone(),
        ic: zkey.ic.clone(),
        coefficients: Vec::new(),
        a: Vec::new(),
        b1: Vec::new(),
        b2: Vec::new(),
        c: Vec::new(),
        h: Vec::new(),
    }
}

impl std::fmt::Debug for CudaProver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaProver({}, {})", self.device_name(), self.source)
    }
}

pub use risc0_groth16_oxide::check::KernelCheck;

impl CudaProver {
    /// Run every kernel of the ABI, the MSM and a fixture proof against the
    /// shared cases (`risc0_groth16_oxide::check`) — the first thing to run
    /// on a CUDA host, before any proof: it localises a codegen, layout, ABI
    /// or orchestration mismatch to one kernel or one composite step.
    pub fn kernel_check(&self) -> Result<Vec<KernelCheck>> {
        use risc0_groth16_oxide::check::{
            first_diff, first_point_diff, proof_diff, Cases, Fixture, WINDOWS,
        };
        let c = Cases::new();
        let n = c.n;
        let mut out = Vec::with_capacity(abi::KERNELS.len() + 3);
        let mut record = |kernel: &'static str, r: Result<String>| {
            out.push(match r {
                Ok(detail) => KernelCheck::from_detail(kernel, detail),
                Err(e) => KernelCheck::from_detail(kernel, format!("launch failed: {e:#}")),
            })
        };
        let fr_out = |kernel: &str, want: &[Fr], inputs: Vec<Arg>| -> Result<String> {
            let o: Buffer<Fr> = self.alloc(want.len())?;
            self.run(kernel, ptr(&o), want.len(), &inputs)?;
            Ok(first_diff(&self.read(&o)?, want))
        };
        let (fb, fb2, fb3) = (
            self.upload(&c.fr)?,
            self.upload(&c.fr2)?,
            self.upload(&c.fr3)?,
        );
        record(
            abi::SCATTER_GROUP,
            (|| {
                let (cb, sb) = (self.upload(&c.coeffs)?, self.upload(&c.starts)?);
                fr_out(
                    abi::SCATTER_GROUP,
                    &c.scatter(),
                    [&slice(&cb)[..], &slice(&sb), &slice(&fb)].concat(),
                )
            })(),
        );
        record(
            abi::POINTWISE_MUL,
            fr_out(
                abi::POINTWISE_MUL,
                &c.pointwise_mul(),
                [&slice(&fb)[..], &slice(&fb2)].concat(),
            ),
        );
        record(
            abi::POINTWISE_MUL_SUB,
            fr_out(
                abi::POINTWISE_MUL_SUB,
                &c.pointwise_mul_sub(),
                [&slice(&fb)[..], &slice(&fb2), &slice(&fb3)].concat(),
            ),
        );
        record(
            abi::POINTWISE_SCALE,
            fr_out(
                abi::POINTWISE_SCALE,
                &c.pointwise_scale(),
                [&slice(&fb)[..], &slice(&fb2), &[Arg::Fr(c.n_inv)]].concat(),
            ),
        );
        record(
            abi::BIT_REVERSE,
            fr_out(
                abi::BIT_REVERSE,
                &c.bit_reverse(),
                [&slice(&fb)[..], &[Arg::U32(c.lg_n)]].concat(),
            ),
        );
        record(
            abi::NTT_STAGE,
            fr_out(
                abi::NTT_STAGE,
                &c.ntt_stage(),
                [
                    &slice(&fb)[..],
                    &[Arg::U32(c.stage_len as u32)],
                    &slice(&fb2),
                    &[Arg::U32(c.stage_stride as u32)],
                ]
                .concat(),
            ),
        );
        // digits: full-width scalars, every window
        record(
            abi::DIGITS,
            (|| {
                let sb = self.upload(&c.wide_canonical())?;
                let o: Buffer<u32> = self.alloc(n)?;
                let mut bad = Vec::new();
                for window in 0..WINDOWS {
                    self.run(
                        abi::DIGITS,
                        ptr(&o),
                        n,
                        &[&slice(&sb)[..], &[Arg::U32(window), Arg::U32(WINDOW_BITS)]].concat(),
                    )?;
                    if self.read(&o)? != c.digits(window) {
                        bad.push(window);
                    }
                }
                Ok(if bad.is_empty() {
                    String::new()
                } else {
                    format!("windows {bad:?} differ")
                })
            })(),
        );
        record(
            abi::DIGITS_ALL,
            (|| {
                let sb = self.upload(&c.wide_canonical())?;
                let o: Buffer<u32> = self.alloc(WINDOWS as usize * n)?;
                self.run(
                    abi::DIGITS_ALL,
                    ptr(&o),
                    WINDOWS as usize * n,
                    &[&slice(&sb)[..], &[Arg::U32(WINDOW_BITS)]].concat(),
                )?;
                Ok(first_diff(&self.read(&o)?, &c.digits_all()))
            })(),
        );
        // bucket sums on G1 and G2: infinity input, empty bucket, P + (−P), P + P
        fn bucket_check<F: Field + Copy + std::fmt::Debug>(
            p: &CudaProver,
            kernel: &str,
            points: &[Affine<F>],
            order: &[u32],
            starts: &[u32],
            want: &[Jacobian<F>],
        ) -> Result<String> {
            let (pb, ob, sb) = (p.upload(points)?, p.upload(order)?, p.upload(starts)?);
            let o: Buffer<Jacobian<F>> = p.alloc(want.len())?;
            p.run(
                kernel,
                ptr(&o),
                want.len(),
                &[&slice(&pb)[..], &slice(&ob), &slice(&sb)].concat(),
            )?;
            Ok(first_point_diff(&p.read(&o)?, want))
        }
        record(
            abi::BUCKET_SUM_G1,
            bucket_check(
                self,
                abi::BUCKET_SUM_G1,
                &c.g1,
                &c.order,
                &c.bstarts,
                &c.bucket_sums_g1(),
            ),
        );
        record(
            abi::BUCKET_SUM_G2,
            bucket_check(
                self,
                abi::BUCKET_SUM_G2,
                &c.g2,
                &c.order,
                &c.bstarts,
                &c.bucket_sums_g2(),
            ),
        );
        // end-to-end MSM (digits + host sort + bucket sums + reduction + Horner)
        record(
            "msm (g1)",
            (|| {
                let pb = self.upload(&c.msm_g1)?;
                let got = self.msm(abi::BUCKET_SUM_G1, &pb, &c.msm_scalars)?;
                Ok(first_point_diff(&[got], &[c.msm_g1()]))
            })(),
        );
        record(
            "msm (g2)",
            (|| {
                let pb = self.upload(&c.msm_g2)?;
                let got = self.msm::<Fp2>(abi::BUCKET_SUM_G2, &pb, &c.msm_scalars)?;
                Ok(first_point_diff(&[got], &[c.msm_g2()]))
            })(),
        );
        // a whole proof on the in-tree fixture, fixed blinding, byte-identical to the core prover
        {
            let f = Fixture::multiplier2();
            record(
                "proof (fixture)",
                self.prove(&f.zkey, &f.witness, &f.r, &f.s)
                    .map(|p| proof_diff(&p, &f))
                    .map_err(|e| anyhow!("prove failed: {e:#}")),
            );
        }
        Ok(out)
    }
}
