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

//! The host pipeline on a Metal device (macOS 13+, Apple Silicon): the
//! canonical kernel sequence dispatched over the kernels of `kernels.metal`,
//! with the sort, reduction and assembly on the host from
//! `risc0-groth16-core`. Written against `metal` 0.29, the crate the
//! in-tree `risc0-zkp` Metal HAL uses.

use std::collections::HashMap;

use anyhow::{anyhow, Context as _, Result};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLLanguageVersion, MTLResourceOptions, MTLSize,
};
use risc0_groth16_core::{
    ec::Jacobian,
    field::{Field, Fr},
    fp2::Fp2,
    prover::{assemble, horner, CoefficientGroups, Msms, Proof, ProveError, Tables},
    zkey::Zkey,
};

use risc0_groth16_oxide::pipeline::{reduce_all_windows, sort_all_windows};

use crate::{pack, MSL_SOURCE, WINDOW_BITS};

const KERNELS: &[&str] = &[
    "scatter_group",
    "pointwise_mul",
    "pointwise_mul_sub",
    "pointwise_scale",
    "bit_reverse",
    "ntt_stage",
    "digits",
    "digits_all",
    "bucket_sum_g1",
    "bucket_sum_g2",
];

/// A device, its command queue, and the compiled kernels.
pub struct MetalProver {
    device: Device,
    queue: CommandQueue,
    _library: Library,
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

/// The transform tables uploaded once per proof.
/// What the coset transform needs besides its two buffers.
struct CosetTables<'a> {
    n_inv: &'a [u8; 32],
    lg_n: u32,
    tb: &'a TransformBuffers,
}

struct TransformBuffers {
    forward: Buffer,
    inverse: Buffer,
    shift: Buffer,
}

/// One kernel argument: a buffer or a small constant passed by value.
#[derive(Clone, Copy)]
enum Arg<'a> {
    Buf(&'a Buffer),
    Bytes(&'a [u8]),
}

impl MetalProver {
    /// Open the system default device and compile the shader source.
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow!("no Metal device"))?;
        let options = CompileOptions::new();
        options.set_language_version(MTLLanguageVersion::V3_0);
        let library = device
            .new_library_with_source(MSL_SOURCE, &options)
            .map_err(|e| anyhow!("compiling the Groth16 kernels: {e}"))?;
        let mut pipelines = HashMap::new();
        for name in KERNELS {
            let f = library
                .get_function(name, None)
                .map_err(|e| anyhow!("kernel {name}: {e}"))?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| anyhow!("pipeline {name}: {e}"))?;
            pipelines.insert(*name, p);
        }
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            _library: library,
            pipelines,
        })
    }

    fn upload(&self, bytes: &[u8]) -> Buffer {
        let len = bytes.len().max(16) as u64;
        let buf = self
            .device
            .new_buffer(len, MTLResourceOptions::StorageModeShared);
        if !bytes.is_empty() {
            // SAFETY: the buffer has at least `bytes.len()` bytes and is CPU-visible (shared storage).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buf.contents() as *mut u8,
                    bytes.len(),
                )
            };
        }
        buf
    }

    fn alloc(&self, bytes: usize) -> Buffer {
        self.device
            .new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
    }

    fn read(&self, buf: &Buffer, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        // SAFETY: shared-storage buffer of at least `bytes` bytes, written by completed command buffers.
        unsafe {
            std::ptr::copy_nonoverlapping(buf.contents() as *const u8, out.as_mut_ptr(), bytes)
        };
        out
    }

    /// Dispatch `kernel` over `n` threads with the given arguments, in order, and wait.
    fn run(&self, kernel: &str, n: usize, args: &[Arg<'_>]) {
        let pipeline = &self.pipelines[kernel];
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (i, a) in args.iter().enumerate() {
            match a {
                Arg::Buf(b) => enc.set_buffer(i as u64, Some(b), 0),
                Arg::Bytes(b) => enc.set_bytes(i as u64, b.len() as u64, b.as_ptr() as *const _),
            }
        }
        let width = pipeline.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(width, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// The coset transform, executing [`risc0_groth16_oxide::schedule::coset`]
    /// over the two buffers: the schedule names the buffer each launch reads
    /// and writes, and the one holding the result.
    fn h_to_coset<'b>(
        &self,
        a: &'b Buffer,
        b: &'b Buffer,
        n: usize,
        coset: &CosetTables<'_>,
    ) -> &'b Buffer {
        let tb = coset.tb;
        use risc0_groth16_oxide::schedule::{Buf, Step, Twiddles};
        let pick = |which: Buf| match which {
            Buf::A => a,
            Buf::B => b,
        };
        let lg = coset.lg_n.to_le_bytes();
        let n_inv = coset.n_inv;
        let sched = risc0_groth16_oxide::schedule::coset(n as u32);
        for step in &sched.steps {
            let (src, dst) = (pick(step.src()), pick(step.dst()));
            match *step {
                Step::BitReverse { .. } => self.run(
                    "bit_reverse",
                    n,
                    &[Arg::Buf(src), Arg::Bytes(&lg), Arg::Buf(dst)],
                ),
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
                        "ntt_stage",
                        n,
                        &[
                            Arg::Buf(src),
                            Arg::Bytes(&len.to_le_bytes()),
                            Arg::Buf(tw),
                            Arg::Bytes(&stride.to_le_bytes()),
                            Arg::Buf(dst),
                        ],
                    )
                }
                Step::Scale { .. } => self.run(
                    "pointwise_scale",
                    n,
                    &[
                        Arg::Buf(src),
                        Arg::Buf(&tb.shift),
                        Arg::Bytes(n_inv),
                        Arg::Buf(dst),
                    ],
                ),
            }
        }
        pick(sched.result)
    }

    /// One MSM over every window at once: one `digits_all` launch, the host
    /// sort (`pipeline::sort_all_windows`), one `bucket_sum` launch over
    /// `windows · buckets` outputs, then the reduction and Horner on the host
    /// — two launches and two reads per MSM instead of two per window.
    fn msm<F: Field>(
        &self,
        kernel: &str,
        points: &Buffer,
        n_points: usize,
        point_bytes: usize,
        scalars: &[Fr],
        unpack: impl Fn(&[u8]) -> Vec<Jacobian<F>>,
    ) -> Jacobian<F> {
        assert_eq!(n_points, scalars.len(), "MSM point and scalar counts");
        let n = scalars.len();
        let w = WINDOW_BITS;
        let buckets = (1usize << w) - 1;
        let windows = 256u32.div_ceil(w) as usize;
        let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let canonical = self.upload(&pack::pack_canonical(scalars));
        let digits_b = self.alloc(windows * n * 4);
        self.run(
            "digits_all",
            windows * n,
            &[
                Arg::Buf(&canonical),
                Arg::Bytes(&(n as u32).to_le_bytes()),
                Arg::Bytes(&w.to_le_bytes()),
                Arg::Buf(&digits_b),
            ],
        );
        let digits: Vec<u32> = self
            .read(&digits_b, windows * n * 4)
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        drop((digits_b, canonical));
        let (order, starts) = sort_all_windows(&digits, n, buckets);
        if order.is_empty() {
            return Jacobian::INFINITY;
        }
        let order_b = self.upload(&u32s(&order));
        let starts_b = self.upload(&u32s(&starts));
        let sum_bytes = point_bytes * 3 / 2;
        let sums = self.alloc(windows * buckets * sum_bytes);
        self.run(
            kernel,
            windows * buckets,
            &[
                Arg::Buf(points),
                Arg::Buf(&order_b),
                Arg::Buf(&starts_b),
                Arg::Buf(&sums),
            ],
        );
        let jac = unpack(&self.read(&sums, windows * buckets * sum_bytes));
        horner(&reduce_all_windows(&jac, buckets), w)
    }

    /// Produce a proof: the arm's implementation of the boundary — upload the
    /// zkey for this one proof and drop it after.
    pub fn prove(&self, zkey: &Zkey, witness: &[Fr], r: &Fr, s: &Fr) -> Result<Proof> {
        let groups = CoefficientGroups::from_zkey(zkey);
        let resident = self.prepare(zkey, &groups);
        self.prove_resident(&resident, witness, r, s)
    }

    /// Upload everything of a zkey that every proof reads — the five point
    /// sets, the grouped coefficients and their group starts, the NTT tables
    /// — once, and keep it on the device (DEF-G16-014). With unified memory
    /// the buffers are the zkey's only copy the GPU needs; per-proof traffic
    /// drops to the witness in and the proof out.
    pub fn prepare(&self, zkey: &Zkey, groups: &CoefficientGroups) -> ResidentZkey {
        let n = zkey.domain_size;
        let t = Tables::new(n);
        let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        ResidentZkey {
            meta: strip(zkey),
            a_constraint: groups.a_constraint.clone(),
            b_constraint: groups.b_constraint.clone(),
            ca: self.upload(&pack::pack_coeffs(&groups.a)),
            sa: self.upload(&u32s(&groups.a_starts)),
            cb: self.upload(&pack::pack_coeffs(&groups.b)),
            sb: self.upload(&u32s(&groups.b_starts)),
            n_inv: pack::fr_bytes(&t.n_inv),
            lg_n: t.lg_n,
            tb: TransformBuffers {
                forward: self.upload(&pack::pack_fr(&t.forward)),
                inverse: self.upload(&pack::pack_fr(&t.inverse)),
                shift: self.upload(&pack::pack_fr(&t.shift_powers)),
            },
            pa: self.upload(&pack::pack_g1(&zkey.a)),
            pb1: self.upload(&pack::pack_g1(&zkey.b1)),
            pc: self.upload(&pack::pack_g1(&zkey.c)),
            ph: self.upload(&pack::pack_g1(&zkey.h)),
            pb2: self.upload(&pack::pack_g2(&zkey.b2)),
            counts: [
                zkey.a.len(),
                zkey.b1.len(),
                zkey.c.len(),
                zkey.h.len(),
                zkey.b2.len(),
            ],
        }
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
        let witness_b = self.upload(&pack::pack_fr(witness));
        // scatter A and B (group sums), placed at their constraint indices on the host
        let scatter = |cb: &Buffer, sb: &Buffer, cons: &[u32]| -> Vec<u8> {
            let out = self.alloc(cons.len() * 32);
            self.run(
                "scatter_group",
                cons.len(),
                &[
                    Arg::Buf(cb),
                    Arg::Buf(sb),
                    Arg::Buf(&witness_b),
                    Arg::Buf(&out),
                ],
            );
            let sums = pack::unpack_fr(&self.read(&out, cons.len() * 32));
            let mut poly = vec![Fr::ZERO; n];
            for (c, v) in cons.iter().zip(sums) {
                poly[*c as usize] = v;
            }
            pack::pack_fr(&poly)
        };
        let a_bytes = scatter(&z.ca, &z.sa, &z.a_constraint);
        let b_bytes = scatter(&z.cb, &z.sb, &z.b_constraint);
        let (a1, a2) = (self.upload(&a_bytes), self.alloc(n * 32));
        let (b1, b2) = (self.upload(&b_bytes), self.alloc(n * 32));
        let (c1, c2) = (self.alloc(n * 32), self.alloc(n * 32));
        self.run(
            "pointwise_mul",
            n,
            &[Arg::Buf(&a1), Arg::Buf(&b1), Arg::Buf(&c1)],
        );
        let coset = CosetTables {
            n_inv: &z.n_inv,
            lg_n: z.lg_n,
            tb: &z.tb,
        };
        let ac = self.h_to_coset(&a1, &a2, n, &coset);
        let bc = self.h_to_coset(&b1, &b2, n, &coset);
        let cc = self.h_to_coset(&c1, &c2, n, &coset);
        let q = self.alloc(n * 32);
        self.run(
            "pointwise_mul_sub",
            n,
            &[Arg::Buf(ac), Arg::Buf(bc), Arg::Buf(cc), Arg::Buf(&q)],
        );
        let quotient = pack::unpack_fr(&self.read(&q, n * 32));
        let [na, nb1, nc, nh, nb2] = z.counts;
        let private = &witness[zkey.num_public + 1..];
        let msms = Msms {
            h: self.msm(
                "bucket_sum_g1",
                &z.ph,
                nh,
                64,
                &quotient,
                pack::unpack_jac_g1,
            ),
            a: self.msm("bucket_sum_g1", &z.pa, na, 64, witness, pack::unpack_jac_g1),
            b1: self.msm(
                "bucket_sum_g1",
                &z.pb1,
                nb1,
                64,
                witness,
                pack::unpack_jac_g1,
            ),
            b2: self.msm::<Fp2>(
                "bucket_sum_g2",
                &z.pb2,
                nb2,
                128,
                witness,
                pack::unpack_jac_g2,
            ),
            c: self.msm("bucket_sum_g1", &z.pc, nc, 64, private, pack::unpack_jac_g1),
        };
        Ok(assemble(zkey, &msms, r, s))
    }
}

/// A zkey uploaded once and kept on the device across proofs: the five point
/// sets (packed), the grouped coefficients with their group starts, the NTT
/// tables. The host keeps only what assembly and the scatter placement need.
pub struct ResidentZkey {
    /// Sizes and the verifying key; every point vector empty.
    meta: Zkey,
    a_constraint: Vec<u32>,
    b_constraint: Vec<u32>,
    ca: Buffer,
    sa: Buffer,
    cb: Buffer,
    sb: Buffer,
    n_inv: [u8; 32],
    lg_n: u32,
    tb: TransformBuffers,
    pa: Buffer,
    pb1: Buffer,
    pc: Buffer,
    ph: Buffer,
    pb2: Buffer,
    /// Point counts of `a, b1, c, h, b2` (the MSM asserts them against the scalars).
    counts: [usize; 5],
}

impl ResidentZkey {
    /// Bytes held on the device.
    pub fn device_bytes(&self) -> usize {
        [
            &self.ca,
            &self.sa,
            &self.cb,
            &self.sb,
            &self.tb.forward,
            &self.tb.inverse,
            &self.tb.shift,
            &self.pa,
            &self.pb1,
            &self.pc,
            &self.ph,
            &self.pb2,
        ]
        .iter()
        .map(|b| b.length() as usize)
        .sum()
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

impl std::fmt::Debug for MetalProver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalProver({})", self.device.name())
    }
}

pub use risc0_groth16_oxide::check::KernelCheck;

impl MetalProver {
    /// Run every kernel, the MSM and a fixture proof on the device against
    /// the shared cases (`risc0_groth16_oxide::check`) — the first thing to
    /// run on a Mac, before any proof: it localises an MSL arithmetic, layout
    /// or orchestration bug to one kernel or one composite step.
    pub fn kernel_check(&self) -> Result<Vec<KernelCheck>> {
        use risc0_groth16_oxide::check::{
            first_diff, first_point_diff, proof_diff, Cases, Fixture, WINDOWS,
        };
        let c = Cases::new();
        let n = c.n;
        let mut out = Vec::new();
        let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let fr_out = |kernel: &'static str, len: usize, args: &[Arg<'_>], want: &[Fr]| {
            let o = self.alloc(len * 32);
            let mut a: Vec<Arg<'_>> = args.to_vec();
            a.push(Arg::Buf(&o));
            self.run(kernel, len, &a);
            KernelCheck::from_detail(
                kernel,
                first_diff(&pack::unpack_fr(&self.read(&o, len * 32)), want),
            )
        };
        let (fb, fb2, fb3) = (
            self.upload(&pack::pack_fr(&c.fr)),
            self.upload(&pack::pack_fr(&c.fr2)),
            self.upload(&pack::pack_fr(&c.fr3)),
        );
        {
            let cb = self.upload(&pack::pack_coeffs(&c.coeffs));
            let sb = self.upload(&u32s(&c.starts));
            out.push(fr_out(
                "scatter_group",
                8,
                &[Arg::Buf(&cb), Arg::Buf(&sb), Arg::Buf(&fb)],
                &c.scatter(),
            ));
        }
        out.push(fr_out(
            "pointwise_mul",
            n,
            &[Arg::Buf(&fb), Arg::Buf(&fb2)],
            &c.pointwise_mul(),
        ));
        out.push(fr_out(
            "pointwise_mul_sub",
            n,
            &[Arg::Buf(&fb), Arg::Buf(&fb2), Arg::Buf(&fb3)],
            &c.pointwise_mul_sub(),
        ));
        let n_inv = pack::fr_bytes(&c.n_inv);
        out.push(fr_out(
            "pointwise_scale",
            n,
            &[Arg::Buf(&fb), Arg::Buf(&fb2), Arg::Bytes(&n_inv)],
            &c.pointwise_scale(),
        ));
        let lg = c.lg_n.to_le_bytes();
        out.push(fr_out(
            "bit_reverse",
            n,
            &[Arg::Buf(&fb), Arg::Bytes(&lg)],
            &c.bit_reverse(),
        ));
        let (len_b, stride_b) = (
            (c.stage_len as u32).to_le_bytes(),
            (c.stage_stride as u32).to_le_bytes(),
        );
        out.push(fr_out(
            "ntt_stage",
            n,
            &[
                Arg::Buf(&fb),
                Arg::Bytes(&len_b),
                Arg::Buf(&fb2),
                Arg::Bytes(&stride_b),
            ],
            &c.ntt_stage(),
        ));
        // digits: full-width scalars, every window
        {
            let canonical = self.upload(&pack::pack_canonical(&c.wide));
            let db = self.alloc(n * 4);
            let w = WINDOW_BITS.to_le_bytes();
            let mut bad = Vec::new();
            for window in 0..WINDOWS {
                self.run(
                    "digits",
                    n,
                    &[
                        Arg::Buf(&canonical),
                        Arg::Bytes(&window.to_le_bytes()),
                        Arg::Bytes(&w),
                        Arg::Buf(&db),
                    ],
                );
                let got: Vec<u32> = self
                    .read(&db, n * 4)
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                if got != c.digits(window) {
                    bad.push(window);
                }
            }
            out.push(KernelCheck::from_detail(
                "digits",
                if bad.is_empty() {
                    String::new()
                } else {
                    format!("windows {bad:?} differ")
                },
            ));
        }
        {
            let canonical = self.upload(&pack::pack_canonical(&c.wide));
            let total = WINDOWS as usize * n;
            let db = self.alloc(total * 4);
            self.run(
                "digits_all",
                total,
                &[
                    Arg::Buf(&canonical),
                    Arg::Bytes(&(n as u32).to_le_bytes()),
                    Arg::Bytes(&WINDOW_BITS.to_le_bytes()),
                    Arg::Buf(&db),
                ],
            );
            let got: Vec<u32> = self
                .read(&db, total * 4)
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            out.push(KernelCheck::from_detail(
                "digits_all",
                first_diff(&got, &c.digits_all()),
            ));
        }
        // bucket sums on G1 and G2: infinity input, empty bucket, P + (−P), P + P
        let (ob, sb) = (self.upload(&u32s(&c.order)), self.upload(&u32s(&c.bstarts)));
        let buckets = c.buckets();
        {
            let pb = self.upload(&pack::pack_g1(&c.g1));
            let sums = self.alloc(buckets * 96);
            self.run(
                "bucket_sum_g1",
                buckets,
                &[Arg::Buf(&pb), Arg::Buf(&ob), Arg::Buf(&sb), Arg::Buf(&sums)],
            );
            let got = pack::unpack_jac_g1(&self.read(&sums, buckets * 96));
            out.push(KernelCheck::from_detail(
                "bucket_sum_g1",
                first_point_diff(&got, &c.bucket_sums_g1()),
            ));
        }
        {
            let pb = self.upload(&pack::pack_g2(&c.g2));
            let sums = self.alloc(buckets * 192);
            self.run(
                "bucket_sum_g2",
                buckets,
                &[Arg::Buf(&pb), Arg::Buf(&ob), Arg::Buf(&sb), Arg::Buf(&sums)],
            );
            let got = pack::unpack_jac_g2(&self.read(&sums, buckets * 192));
            out.push(KernelCheck::from_detail(
                "bucket_sum_g2",
                first_point_diff(&got, &c.bucket_sums_g2()),
            ));
        }
        // end-to-end MSM (digits + host sort + bucket sums + reduction + Horner)
        {
            let pb = self.upload(&pack::pack_g1(&c.msm_g1));
            let got = self.msm(
                "bucket_sum_g1",
                &pb,
                c.msm_g1.len(),
                64,
                &c.msm_scalars,
                pack::unpack_jac_g1,
            );
            out.push(KernelCheck::from_detail(
                "msm (g1)",
                first_point_diff(&[got], &[c.msm_g1()]),
            ));
            let pb2 = self.upload(&pack::pack_g2(&c.msm_g2));
            let got = self.msm::<Fp2>(
                "bucket_sum_g2",
                &pb2,
                c.msm_g2.len(),
                128,
                &c.msm_scalars,
                pack::unpack_jac_g2,
            );
            out.push(KernelCheck::from_detail(
                "msm (g2)",
                first_point_diff(&[got], &[c.msm_g2()]),
            ));
        }
        // a whole proof on the in-tree fixture, fixed blinding, byte-identical to the core prover
        {
            let f = Fixture::multiplier2();
            out.push(match self.prove(&f.zkey, &f.witness, &f.r, &f.s) {
                Ok(p) => KernelCheck::from_detail("proof (fixture)", proof_diff(&p, &f)),
                Err(e) => {
                    KernelCheck::from_detail("proof (fixture)", format!("prove failed: {e:#}"))
                }
            });
        }
        Ok(out)
    }
}

/// A convenience: prove once on the default device.
pub fn prove(zkey: &Zkey, witness: &[Fr], r: &Fr, s: &Fr) -> Result<Proof> {
    MetalProver::new()
        .context("Metal device")?
        .prove(zkey, witness, r, s)
}
