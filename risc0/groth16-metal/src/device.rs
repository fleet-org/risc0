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

//! The host pipeline over the kernels of `kernels.metal`: the canonical
//! kernel sequence dispatched by name, with the sort, reduction and assembly
//! on the host from `risc0-groth16-core`. The pipeline is generic over a
//! [`Backend`] — where the shaders run:
//!
//! - [`MetalBackend`] (macOS 13+, Apple Silicon): the shaders compiled by the
//!   Metal compiler at start-up, dispatched on the system default device
//!   through `metal` 0.29, the crate the in-tree `risc0-zkp` Metal HAL uses.
//! - [`HostBackend`] (every other target): the SAME shader source compiled as
//!   C++ by the build script (`msl-host/`) and run one thread index at a
//!   time on the CPU — the Metal arm's `oxide-cpu`. It proves the shaders'
//!   arithmetic, layouts and kernels, and the whole pipeline down to a
//!   fixture proof, where there is no Metal device; what it cannot prove is
//!   the Metal compiler and the device themselves.
//!
//! One orchestration, two executors: a divergence between the two is a
//! difference in the backend, never in the pipeline.

use anyhow::Result;
use risc0_groth16_core::{
    ec::Jacobian,
    field::{Field, Fr},
    fp2::Fp2,
    prover::{assemble, horner, CoefficientGroups, Msms, Proof, ProveError, Tables},
    zkey::Zkey,
};
pub use risc0_groth16_oxide::check::KernelCheck;
use risc0_groth16_oxide::pipeline::{plan_ranges, reduce_all_windows, sort_all_windows, CHUNK};

use crate::{pack, WINDOW_BITS};

/// Every kernel of `kernels.metal`, by name.
pub const KERNELS: &[&str] = &[
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
    "jacobian_sum_g1",
    "jacobian_sum_g2",
];

/// One kernel argument, in `[[buffer(i)]]` order: a buffer, or a small
/// constant passed by value.
pub enum Arg<'a, T> {
    /// A device buffer.
    Buf(&'a T),
    /// Bytes passed by value (a `constant T&` parameter).
    Bytes(&'a [u8]),
}

// By hand: a derive would demand `T: Clone`, and the buffers are not clonable — the argument is
// two references either way.
impl<T> Clone for Arg<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Arg<'_, T> {}

/// Where the shaders run.
pub trait Backend {
    /// A buffer the shaders read and write.
    type Buf;
    /// A buffer holding `bytes`.
    fn upload(&self, bytes: &[u8]) -> Self::Buf;
    /// A zeroed buffer of `bytes` bytes.
    fn alloc(&self, bytes: usize) -> Self::Buf;
    /// The first `bytes` bytes of `buf`, after every previous `run` completed.
    fn read(&self, buf: &Self::Buf, bytes: usize) -> Vec<u8>;
    /// Bytes held by `buf`.
    fn len(&self, buf: &Self::Buf) -> usize;
    /// Run `kernel` over `n` thread indices with `args` in `[[buffer(i)]]`
    /// order, and wait.
    fn run(&self, kernel: &str, n: usize, args: &[Arg<'_, Self::Buf>]);
    /// The device, for reports.
    fn describe(&self) -> String;
}

/// The prover pipeline over a backend.
pub struct Prover<B: Backend> {
    backend: B,
}

/// The transform tables on the device.
struct TransformBuffers<T> {
    forward: T,
    inverse: T,
    shift: T,
}

/// What the coset transform needs besides its two buffers.
struct CosetTables<'a, T> {
    n_inv: &'a [u8; 32],
    lg_n: u32,
    tb: &'a TransformBuffers<T>,
}

impl<B: Backend> Prover<B> {
    /// A prover over `backend`.
    pub fn with_backend(backend: B) -> Self {
        Self { backend }
    }

    /// The backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    fn upload(&self, bytes: &[u8]) -> B::Buf {
        self.backend.upload(bytes)
    }

    fn alloc(&self, bytes: usize) -> B::Buf {
        self.backend.alloc(bytes)
    }

    fn read(&self, buf: &B::Buf, bytes: usize) -> Vec<u8> {
        self.backend.read(buf, bytes)
    }

    fn run(&self, kernel: &str, n: usize, args: &[Arg<'_, B::Buf>]) {
        self.backend.run(kernel, n, args)
    }

    /// The coset transform, executing [`risc0_groth16_oxide::schedule::coset`]
    /// over the two buffers: the schedule names the buffer each launch reads
    /// and writes, and the one holding the result.
    fn h_to_coset<'b>(
        &self,
        a: &'b B::Buf,
        b: &'b B::Buf,
        n: usize,
        coset: &CosetTables<'_, B::Buf>,
    ) -> &'b B::Buf {
        use risc0_groth16_oxide::schedule::{Buf, Step, Twiddles};
        let tb = coset.tb;
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
    /// pieces of at most [`CHUNK`] points, then the levels of `jacobian_sum`
    /// the plan (`pipeline::plan_ranges`) needs until there is one sum per
    /// `(window, bucket)`, then the reduction and Horner on the host — no
    /// thread adds a long chain (C21). `kernels` names the bucket-sum and the
    /// Jacobian-sum kernel of the group.
    fn msm<F: Field>(
        &self,
        kernels: (&str, &str),
        points: &B::Buf,
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
        // the bucket sums in pieces of at most CHUNK points, then the levels of
        // Jacobian sums the plan needs — no thread adds a long chain (C21)
        let plan = plan_ranges(&starts, CHUNK);
        let sum_bytes = point_bytes * 3 / 2;
        let order_b = self.upload(&u32s(&order));
        let mut len = plan[0].len() - 1;
        let mut sums = {
            let starts_b = self.upload(&u32s(&plan[0]));
            let o = self.alloc(len * sum_bytes);
            self.run(
                kernels.0,
                len,
                &[
                    Arg::Buf(points),
                    Arg::Buf(&order_b),
                    Arg::Buf(&starts_b),
                    Arg::Buf(&o),
                ],
            );
            o
        };
        for level in &plan[1..] {
            let starts_b = self.upload(&u32s(level));
            let next_len = level.len() - 1;
            let next = self.alloc(next_len * sum_bytes);
            self.run(
                kernels.1,
                next_len,
                &[Arg::Buf(&sums), Arg::Buf(&starts_b), Arg::Buf(&next)],
            );
            sums = next;
            len = next_len;
        }
        let jac = unpack(&self.read(&sums, len * sum_bytes));
        debug_assert_eq!(jac.len(), windows * buckets);
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
    pub fn prepare(&self, zkey: &Zkey, groups: &CoefficientGroups) -> ResidentZkey<B> {
        let n = zkey.domain_size;
        let t = Tables::new(n);
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

    /// [`Self::prepare`] for a zkey the caller owns: every point set and the
    /// coefficient groups are packed, uploaded and DROPPED one after another,
    /// so the parsed and the packed copies never coexist in host memory.
    pub fn prepare_owned(&self, mut zkey: Zkey, mut groups: CoefficientGroups) -> ResidentZkey<B> {
        let n = zkey.domain_size;
        let t = Tables::new(n);
        let meta = strip(&zkey);
        let counts = [
            zkey.a.len(),
            zkey.b1.len(),
            zkey.c.len(),
            zkey.h.len(),
            zkey.b2.len(),
        ];
        let a_constraint = std::mem::take(&mut groups.a_constraint);
        let b_constraint = std::mem::take(&mut groups.b_constraint);
        let ca = self.upload(&pack::pack_coeffs(&std::mem::take(&mut groups.a)));
        let sa = self.upload(&u32s(&std::mem::take(&mut groups.a_starts)));
        let cb = self.upload(&pack::pack_coeffs(&std::mem::take(&mut groups.b)));
        let sb = self.upload(&u32s(&std::mem::take(&mut groups.b_starts)));
        drop(groups);
        let pa = self.upload(&pack::pack_g1(&std::mem::take(&mut zkey.a)));
        let pb1 = self.upload(&pack::pack_g1(&std::mem::take(&mut zkey.b1)));
        let pc = self.upload(&pack::pack_g1(&std::mem::take(&mut zkey.c)));
        let ph = self.upload(&pack::pack_g1(&std::mem::take(&mut zkey.h)));
        let pb2 = self.upload(&pack::pack_g2(&std::mem::take(&mut zkey.b2)));
        drop(zkey);
        ResidentZkey {
            meta,
            a_constraint,
            b_constraint,
            ca,
            sa,
            cb,
            sb,
            n_inv: pack::fr_bytes(&t.n_inv),
            lg_n: t.lg_n,
            tb: TransformBuffers {
                forward: self.upload(&pack::pack_fr(&t.forward)),
                inverse: self.upload(&pack::pack_fr(&t.inverse)),
                shift: self.upload(&pack::pack_fr(&t.shift_powers)),
            },
            pa,
            pb1,
            pc,
            ph,
            pb2,
            counts,
        }
    }

    /// Bytes a resident zkey holds on the device.
    pub fn resident_bytes(&self, z: &ResidentZkey<B>) -> usize {
        [
            &z.ca,
            &z.sa,
            &z.cb,
            &z.sb,
            &z.tb.forward,
            &z.tb.inverse,
            &z.tb.shift,
            &z.pa,
            &z.pb1,
            &z.pc,
            &z.ph,
            &z.pb2,
        ]
        .iter()
        .map(|b| self.backend.len(b))
        .sum()
    }

    /// Produce a proof against a resident zkey: the witness goes up, the
    /// proof comes back; nothing of the zkey moves.
    pub fn prove_resident(
        &self,
        z: &ResidentZkey<B>,
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
        let scatter = |cb: &B::Buf, sb: &B::Buf, cons: &[u32]| -> Vec<u8> {
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
        let g1 = |points, count, scalars: &[Fr]| {
            self.msm(
                ("bucket_sum_g1", "jacobian_sum_g1"),
                points,
                count,
                64,
                scalars,
                pack::unpack_jac_g1,
            )
        };
        let msms = Msms {
            h: g1(&z.ph, nh, &quotient),
            a: g1(&z.pa, na, witness),
            b1: g1(&z.pb1, nb1, witness),
            b2: self.msm::<Fp2>(
                ("bucket_sum_g2", "jacobian_sum_g2"),
                &z.pb2,
                nb2,
                128,
                witness,
                pack::unpack_jac_g2,
            ),
            c: g1(&z.pc, nc, private),
        };
        Ok(assemble(zkey, &msms, r, s))
    }

    /// Run every kernel, the MSM and a fixture proof against the shared cases
    /// (`risc0_groth16_oxide::check`) — the first thing to run on a Mac, and
    /// what `cargo test` runs on the CPU everywhere else: it localises a
    /// shader arithmetic, layout or orchestration bug to one kernel or one
    /// composite step.
    pub fn kernel_check(&self) -> Result<Vec<KernelCheck>> {
        use risc0_groth16_oxide::check::{
            first_diff, first_point_diff, proof_diff, Cases, Fixture, WINDOWS,
        };
        let c = Cases::new();
        let n = c.n;
        let mut out = Vec::new();
        let fr_out = |kernel: &'static str, len: usize, args: &[Arg<'_, B::Buf>], want: &[Fr]| {
            let o = self.alloc(len * 32);
            let mut a: Vec<Arg<'_, B::Buf>> = args.to_vec();
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
                if u32s_from(&self.read(&db, n * 4)) != c.digits(window) {
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
            let total = WINDOWS as usize * n;
            let db = self.alloc(total * 4);
            self.run(
                "digits_all",
                total,
                &[
                    Arg::Buf(&canonical),
                    Arg::Bytes(&(n as u32).to_le_bytes()),
                    Arg::Bytes(&w),
                    Arg::Buf(&db),
                ],
            );
            out.push(KernelCheck::from_detail(
                "digits_all",
                first_diff(&u32s_from(&self.read(&db, total * 4)), &c.digits_all()),
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
        // one level of the reduction above the bucket sums: {B0,B1} {} {B2,B3,B4}
        let jsb = self.upload(&u32s(&c.jstarts));
        let jn = c.jstarts.len() - 1;
        {
            let jb = self.upload(&pack::pack_jac_g1(&c.bucket_sums_g1()));
            let sums = self.alloc(jn * 96);
            self.run(
                "jacobian_sum_g1",
                jn,
                &[Arg::Buf(&jb), Arg::Buf(&jsb), Arg::Buf(&sums)],
            );
            let got = pack::unpack_jac_g1(&self.read(&sums, jn * 96));
            out.push(KernelCheck::from_detail(
                "jacobian_sum_g1",
                first_point_diff(&got, &c.jacobian_sums_g1()),
            ));
        }
        {
            let jb = self.upload(&pack::pack_jac_g2(&c.bucket_sums_g2()));
            let sums = self.alloc(jn * 192);
            self.run(
                "jacobian_sum_g2",
                jn,
                &[Arg::Buf(&jb), Arg::Buf(&jsb), Arg::Buf(&sums)],
            );
            let got = pack::unpack_jac_g2(&self.read(&sums, jn * 192));
            out.push(KernelCheck::from_detail(
                "jacobian_sum_g2",
                first_point_diff(&got, &c.jacobian_sums_g2()),
            ));
        }
        // end-to-end MSM (digits + host sort + bucket sums + reduction + Horner)
        {
            let pb = self.upload(&pack::pack_g1(&c.msm_g1));
            let got = self.msm(
                ("bucket_sum_g1", "jacobian_sum_g1"),
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
                ("bucket_sum_g2", "jacobian_sum_g2"),
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

impl<B: Backend> std::fmt::Debug for Prover<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Prover({})", self.backend.describe())
    }
}

/// A zkey uploaded once and kept on the device across proofs: the five point
/// sets (packed), the grouped coefficients with their group starts, the NTT
/// tables. The host keeps only what assembly and the scatter placement need.
pub struct ResidentZkey<B: Backend> {
    /// Sizes and the verifying key; every point vector empty.
    meta: Zkey,
    a_constraint: Vec<u32>,
    b_constraint: Vec<u32>,
    ca: B::Buf,
    sa: B::Buf,
    cb: B::Buf,
    sb: B::Buf,
    n_inv: [u8; 32],
    lg_n: u32,
    tb: TransformBuffers<B::Buf>,
    pa: B::Buf,
    pb1: B::Buf,
    pc: B::Buf,
    ph: B::Buf,
    pb2: B::Buf,
    /// Point counts of `a, b1, c, h, b2` (the MSM asserts them against the scalars).
    counts: [usize; 5],
}

impl<B: Backend> ResidentZkey<B> {
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

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u32s_from(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ---- the Metal device ----------------------------------------------------------------------

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod metal_backend {
    use std::collections::HashMap;

    use anyhow::{anyhow, Result};
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
        MTLLanguageVersion, MTLResourceOptions, MTLSize,
    };

    use super::{Arg, Backend, Prover, KERNELS};
    use crate::MSL_SOURCE;

    /// A Metal device, its command queue, and the compiled kernels.
    pub struct MetalBackend {
        device: Device,
        queue: CommandQueue,
        _library: Library,
        pipelines: HashMap<&'static str, ComputePipelineState>,
    }

    impl MetalBackend {
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
    }

    impl Backend for MetalBackend {
        type Buf = Buffer;

        fn upload(&self, bytes: &[u8]) -> Buffer {
            let len = bytes.len().max(16) as u64;
            let buf = self
                .device
                .new_buffer(len, MTLResourceOptions::StorageModeShared);
            if !bytes.is_empty() {
                // SAFETY: the buffer has at least `bytes.len()` bytes and is CPU-visible.
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
            // SAFETY: shared-storage buffer of at least `bytes` bytes, written by completed
            // command buffers.
            unsafe {
                std::ptr::copy_nonoverlapping(buf.contents() as *const u8, out.as_mut_ptr(), bytes)
            };
            out
        }

        fn len(&self, buf: &Buffer) -> usize {
            buf.length() as usize
        }

        /// Dispatch `kernel` over `n` threads with the arguments, in order, and wait.
        fn run(&self, kernel: &str, n: usize, args: &[Arg<'_, Buffer>]) {
            let pipeline = &self.pipelines[kernel];
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            for (i, a) in args.iter().enumerate() {
                match a {
                    Arg::Buf(b) => enc.set_buffer(i as u64, Some(b), 0),
                    Arg::Bytes(b) => {
                        enc.set_bytes(i as u64, b.len() as u64, b.as_ptr() as *const _)
                    }
                }
            }
            let width = pipeline.max_total_threads_per_threadgroup().min(256);
            enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(width, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        fn describe(&self) -> String {
            format!("Metal: {}", self.device.name())
        }
    }

    /// The prover on the system default Metal device.
    pub type MetalProver = Prover<MetalBackend>;

    impl MetalProver {
        /// Open the system default device and compile the shader source.
        pub fn new() -> Result<Self> {
            Ok(Prover::with_backend(MetalBackend::new()?))
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use metal_backend::{MetalBackend, MetalProver};

// ---- the shaders on the CPU ----------------------------------------------------------------

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod host_backend {
    use std::{
        cell::UnsafeCell,
        ffi::{c_char, c_void, CString},
    };

    use anyhow::Result;

    use super::{Arg, Backend, Prover};

    extern "C" {
        /// `msl-host/shim.cpp`: the kernel by name, its arguments in
        /// `[[buffer(i)]]` order as raw pointers, the thread count.
        fn msl_run(kernel: *const c_char, args: *const *const c_void, nargs: u32, n: u32) -> i32;
    }

    /// A buffer the shaders read and write on the CPU.
    pub struct HostBuf(UnsafeCell<Vec<u8>>);

    impl HostBuf {
        fn ptr(&self) -> *mut u8 {
            // SAFETY: runs are sequential and single-threaded; a kernel writes only its
            // output buffer, which no argument aliases.
            let v: &mut Vec<u8> = unsafe { &mut *self.0.get() };
            v.as_mut_ptr()
        }

        fn len(&self) -> usize {
            // SAFETY: as above; the length never changes after construction.
            let v: &Vec<u8> = unsafe { &*self.0.get() };
            v.len()
        }
    }

    /// The shaders compiled as C++ by the build script, run one thread index at a time.
    #[derive(Default)]
    pub struct HostBackend;

    impl Backend for HostBackend {
        type Buf = HostBuf;

        fn upload(&self, bytes: &[u8]) -> HostBuf {
            let mut v = bytes.to_vec();
            v.resize(bytes.len().max(16), 0);
            HostBuf(UnsafeCell::new(v))
        }

        fn alloc(&self, bytes: usize) -> HostBuf {
            HostBuf(UnsafeCell::new(vec![0u8; bytes.max(16)]))
        }

        fn read(&self, buf: &HostBuf, bytes: usize) -> Vec<u8> {
            // SAFETY: no run is in progress; the buffer holds at least `bytes` bytes.
            let v: &Vec<u8> = unsafe { &*buf.0.get() };
            v[..bytes].to_vec()
        }

        fn len(&self, buf: &HostBuf) -> usize {
            buf.len()
        }

        fn run(&self, kernel: &str, n: usize, args: &[Arg<'_, HostBuf>]) {
            let name = CString::new(kernel).expect("kernel name");
            let ptrs: Vec<*const c_void> = args
                .iter()
                .map(|a| match a {
                    Arg::Buf(b) => b.ptr() as *const c_void,
                    Arg::Bytes(b) => b.as_ptr() as *const c_void,
                })
                .collect();
            // SAFETY: the shim reads each argument as the kernel's `[[buffer(i)]]` type; the
            // buffers were sized by the same pipeline that sizes them for Metal.
            let rc = unsafe { msl_run(name.as_ptr(), ptrs.as_ptr(), ptrs.len() as u32, n as u32) };
            assert_eq!(rc, 0, "kernel `{kernel}` is not in the shim");
        }

        fn describe(&self) -> String {
            "the Metal shaders compiled as C++, on the CPU".into()
        }
    }

    /// The prover over the shaders on the CPU — the Metal arm's `oxide-cpu`.
    pub type HostMslProver = Prover<HostBackend>;

    impl HostMslProver {
        /// The shaders as built by the build script.
        pub fn new() -> Result<Self> {
            Ok(Prover::with_backend(HostBackend))
        }
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub use host_backend::{HostBackend, HostBuf, HostMslProver};

/// Produce a proof on the system default Metal device (macOS), or on the CPU
/// through the shaders compiled as C++ (elsewhere).
pub fn prove(zkey: &Zkey, witness: &[Fr], r: &Fr, s: &Fr) -> Result<Proof> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let prover = MetalProver::new()?;
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let prover = HostMslProver::new()?;
    prover.prove(zkey, witness, r, s)
}

#[cfg(all(test, not(all(target_os = "macos", target_arch = "aarch64"))))]
mod host_tests {
    use super::*;

    /// The Metal shaders, run on the CPU: every kernel, the MSM on G1 and G2,
    /// and a whole fixture proof agree with the Rust bodies and the core
    /// prover — the same check the Mac runs on the device.
    #[test]
    fn shaders_on_the_cpu_agree_with_the_rust_bodies_down_to_a_fixture_proof() {
        let prover = HostMslProver::new().unwrap();
        let checks = prover.kernel_check().unwrap();
        assert_eq!(checks.len(), 15, "12 kernels, 2 MSMs, 1 proof");
        for c in &checks {
            assert!(c.ok, "{}: {}", c.kernel, c.detail);
        }
    }
}
