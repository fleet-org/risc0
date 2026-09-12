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
    field::{Field, Fr},
    fp2::Fp2,
    prover::{
        assemble, counting_sort_by_digit, horner, reduce_buckets, CoefficientGroups, Msms, Proof,
        ProveError, Tables,
    },
    zkey::Zkey,
};
use risc0_groth16_oxide::{
    abi,
    pipeline::WINDOW_BITS,
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

/// The twiddle tables and coset shift powers, uploaded once per proof.
struct TransformBuffers {
    forward: Buffer<Fr>,
    inverse: Buffer<Fr>,
    shift: Buffer<Fr>,
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
        t: &Tables,
        tb: &TransformBuffers,
    ) -> Result<&'b Buffer<Fr>> {
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
                    &[&slice(src)[..], &[Arg::U32(t.lg_n)]].concat(),
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
                    &[&slice(src)[..], &slice(&tb.shift), &[Arg::Fr(t.n_inv)]].concat(),
                )?,
            }
        }
        Ok(pick(sched.result))
    }

    /// One MSM: digits on the device, counting sort on the host, bucket sums
    /// on the device, reduction and Horner on the host — the Metal arm's split.
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
        let windows = 256u32.div_ceil(w);
        let canonical: Vec<[u64; 4]> = scalars.iter().map(Fr::to_canonical).collect();
        let canonical_b = self.upload(&canonical)?;
        let digits_b: Buffer<u32> = self.alloc(n)?;
        let sums_b: Buffer<Jacobian<F>> = self.alloc(buckets)?;
        let mut window_sums = Vec::with_capacity(windows as usize);
        for window in 0..windows {
            self.run(
                abi::DIGITS,
                ptr(&digits_b),
                n,
                &[&slice(&canonical_b)[..], &[Arg::U32(window), Arg::U32(w)]].concat(),
            )?;
            let digits = self.read(&digits_b)?;
            let (order, starts) = counting_sort_by_digit(&digits, buckets);
            if order.is_empty() {
                window_sums.push(Jacobian::INFINITY);
                continue;
            }
            let order_b = self.upload(&order)?;
            let starts_b = self.upload(&starts)?;
            self.run(
                kernel,
                ptr(&sums_b),
                buckets,
                &[&slice(points)[..], &slice(&order_b), &slice(&starts_b)].concat(),
            )?;
            let sums = self.read(&sums_b)?;
            window_sums.push(reduce_buckets(&sums));
        }
        Ok(horner(&window_sums, w))
    }

    /// Produce a proof: the arm's implementation of the boundary.
    pub fn prove(&self, zkey: &Zkey, witness: &[Fr], r: &Fr, s: &Fr) -> Result<Proof> {
        let groups = CoefficientGroups::from_zkey(zkey);
        self.prove_grouped(zkey, &groups, witness, r, s)
    }

    /// [`Self::prove`] with the coefficient groups already built.
    pub fn prove_grouped(
        &self,
        zkey: &Zkey,
        groups: &CoefficientGroups,
        witness: &[Fr],
        r: &Fr,
        s: &Fr,
    ) -> Result<Proof> {
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
        let n = zkey.domain_size;
        let t = Tables::new(n);
        let witness_b = self.upload(witness)?;
        let tb = TransformBuffers {
            forward: self.upload(&t.forward)?,
            inverse: self.upload(&t.inverse)?,
            shift: self.upload(&t.shift_powers)?,
        };
        // scatter A and B (group sums on the device), placed at their constraint
        // indices on the host
        let scatter = |coeffs: &[GroupedCoeff], starts: &[u32], cons: &[u32]| -> Result<Vec<Fr>> {
            let cb = self.upload(coeffs)?;
            let sb = self.upload(starts)?;
            let out: Buffer<Fr> = self.alloc(cons.len())?;
            self.run(
                abi::SCATTER_GROUP,
                ptr(&out),
                cons.len(),
                &[&slice(&cb)[..], &slice(&sb), &slice(&witness_b)].concat(),
            )?;
            let sums = self.read(&out)?;
            let mut poly = vec![Fr::ZERO; n];
            for (c, v) in cons.iter().zip(sums) {
                poly[*c as usize] = v;
            }
            Ok(poly)
        };
        let a_poly = scatter(&groups.a, &groups.a_starts, &groups.a_constraint)?;
        let b_poly = scatter(&groups.b, &groups.b_starts, &groups.b_constraint)?;
        let (a1, a2) = (self.upload(&a_poly)?, self.alloc::<Fr>(n)?);
        let (b1, b2) = (self.upload(&b_poly)?, self.alloc::<Fr>(n)?);
        let (c1, c2) = (self.alloc::<Fr>(n)?, self.alloc::<Fr>(n)?);
        self.run(
            abi::POINTWISE_MUL,
            ptr(&c1),
            n,
            &[&slice(&a1)[..], &slice(&b1)].concat(),
        )?;
        let ac = self.h_to_coset(&a1, &a2, n, &t, &tb)?;
        let bc = self.h_to_coset(&b1, &b2, n, &t, &tb)?;
        let cc = self.h_to_coset(&c1, &c2, n, &t, &tb)?;
        let q: Buffer<Fr> = self.alloc(n)?;
        self.run(
            abi::POINTWISE_MUL_SUB,
            ptr(&q),
            n,
            &[&slice(ac)[..], &slice(bc), &slice(cc)].concat(),
        )?;
        let quotient = self.read(&q)?;
        // free the polynomial buffers (7 × n × 32 B) before the points go up
        drop((a1, a2, b1, b2, c1, c2, q, tb));
        let (pa, pb1, pc, ph) = (
            self.upload(&zkey.a)?,
            self.upload(&zkey.b1)?,
            self.upload(&zkey.c)?,
            self.upload(&zkey.h)?,
        );
        let pb2 = self.upload(&zkey.b2)?;
        let private = &witness[zkey.num_public + 1..];
        let msms = Msms {
            h: self.msm(abi::BUCKET_SUM_G1, &ph, &quotient)?,
            a: self.msm(abi::BUCKET_SUM_G1, &pa, witness)?,
            b1: self.msm(abi::BUCKET_SUM_G1, &pb1, witness)?,
            b2: self.msm::<Fp2>(abi::BUCKET_SUM_G2, &pb2, witness)?,
            c: self.msm(abi::BUCKET_SUM_G1, &pc, private)?,
        };
        Ok(assemble(zkey, &msms, r, s))
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
