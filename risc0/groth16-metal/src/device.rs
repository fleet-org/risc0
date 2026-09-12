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
    ec::{Affine, Jacobian},
    field::{Field, Fp, Fr},
    fp2::Fp2,
    prover::{
        assemble, counting_sort_by_digit, horner, reduce_buckets, CoefficientGroups, Msms, Proof,
        ProveError, Tables,
    },
    zkey::Zkey,
};

use crate::{pack, MSL_SOURCE, WINDOW_BITS};

const KERNELS: &[&str] = &[
    "scatter_group",
    "pointwise_mul",
    "pointwise_mul_sub",
    "pointwise_scale",
    "bit_reverse",
    "ntt_stage",
    "digits",
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
struct TransformBuffers {
    forward: Buffer,
    inverse: Buffer,
    shift: Buffer,
}

/// One kernel argument: a buffer or a small constant passed by value.
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
        t: &Tables,
        tb: &TransformBuffers,
    ) -> &'b Buffer {
        use risc0_groth16_oxide::schedule::{Buf, Step, Twiddles};
        let pick = |which: Buf| match which {
            Buf::A => a,
            Buf::B => b,
        };
        let lg = t.lg_n.to_le_bytes();
        let n_inv = pack::fr_bytes(&t.n_inv);
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
                        Arg::Bytes(&n_inv),
                        Arg::Buf(dst),
                    ],
                ),
            }
        }
        pick(sched.result)
    }

    /// One MSM: digits on the device, counting sort on the host, bucket sums on
    /// the device, reduction and Horner on the host.
    fn msm<F: Field>(
        &self,
        kernel: &str,
        points: &Buffer,
        point_bytes: usize,
        scalars: &[Fr],
        unpack: impl Fn(&[u8]) -> Vec<Jacobian<F>>,
    ) -> Jacobian<F> {
        let n = scalars.len();
        let w = WINDOW_BITS;
        let buckets = (1usize << w) - 1;
        let windows = 256u32.div_ceil(w);
        let canonical = self.upload(&pack::pack_canonical(scalars));
        let digits = self.alloc(n * 4);
        let sums = self.alloc(buckets * point_bytes * 3 / 2);
        let mut window_sums = Vec::with_capacity(windows as usize);
        for window in 0..windows {
            self.run(
                "digits",
                n,
                &[
                    Arg::Buf(&canonical),
                    Arg::Bytes(&window.to_le_bytes()),
                    Arg::Bytes(&w.to_le_bytes()),
                    Arg::Buf(&digits),
                ],
            );
            let d: Vec<u32> = self
                .read(&digits, n * 4)
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let (order, starts) = counting_sort_by_digit(&d, buckets);
            let order_b = self.upload(
                &order
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let starts_b = self.upload(
                &starts
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            self.run(
                kernel,
                buckets,
                &[
                    Arg::Buf(points),
                    Arg::Buf(&order_b),
                    Arg::Buf(&starts_b),
                    Arg::Buf(&sums),
                ],
            );
            let jac = unpack(&self.read(&sums, buckets * point_bytes * 3 / 2));
            window_sums.push(reduce_buckets(&jac));
        }
        horner(&window_sums, w)
    }

    /// Produce a proof: the arm's implementation of the boundary.
    pub fn prove(&self, zkey: &Zkey, witness: &[Fr], r: &Fr, s: &Fr) -> Result<Proof> {
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
        let groups = CoefficientGroups::from_zkey(zkey);
        let t = Tables::new(n);
        let witness_b = self.upload(&pack::pack_fr(witness));
        let tb = TransformBuffers {
            forward: self.upload(&pack::pack_fr(&t.forward)),
            inverse: self.upload(&pack::pack_fr(&t.inverse)),
            shift: self.upload(&pack::pack_fr(&t.shift_powers)),
        };

        // scatter A and B (group sums), placed at their constraint indices on the host
        let scatter = |coeffs: &[risc0_groth16_core::coeff::GroupedCoeff],
                       starts: &[u32],
                       cons: &[u32]|
         -> Vec<u8> {
            let cb = self.upload(&pack::pack_coeffs(coeffs));
            let sb = self.upload(
                &starts
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let out = self.alloc(cons.len() * 32);
            self.run(
                "scatter_group",
                cons.len(),
                &[
                    Arg::Buf(&cb),
                    Arg::Buf(&sb),
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
        let a_bytes = scatter(&groups.a, &groups.a_starts, &groups.a_constraint);
        let b_bytes = scatter(&groups.b, &groups.b_starts, &groups.b_constraint);
        let (a1, a2) = (self.upload(&a_bytes), self.alloc(n * 32));
        let (b1, b2) = (self.upload(&b_bytes), self.alloc(n * 32));
        let (c1, c2) = (self.alloc(n * 32), self.alloc(n * 32));
        self.run(
            "pointwise_mul",
            n,
            &[Arg::Buf(&a1), Arg::Buf(&b1), Arg::Buf(&c1)],
        );

        let ac = self.h_to_coset(&a1, &a2, n, &t, &tb);
        let bc = self.h_to_coset(&b1, &b2, n, &t, &tb);
        let cc = self.h_to_coset(&c1, &c2, n, &t, &tb);
        let q = self.alloc(n * 32);
        self.run(
            "pointwise_mul_sub",
            n,
            &[Arg::Buf(ac), Arg::Buf(bc), Arg::Buf(cc), Arg::Buf(&q)],
        );
        let quotient = pack::unpack_fr(&self.read(&q, n * 32));

        let g1 = |ps: &[Affine<Fp>]| self.upload(&pack::pack_g1(ps));
        let (pa, pb1, pc, ph) = (g1(&zkey.a), g1(&zkey.b1), g1(&zkey.c), g1(&zkey.h));
        let pb2 = self.upload(&pack::pack_g2(&zkey.b2));
        let private = &witness[zkey.num_public + 1..];
        let msms = Msms {
            h: self.msm("bucket_sum_g1", &ph, 64, &quotient, pack::unpack_jac_g1),
            a: self.msm("bucket_sum_g1", &pa, 64, witness, pack::unpack_jac_g1),
            b1: self.msm("bucket_sum_g1", &pb1, 64, witness, pack::unpack_jac_g1),
            b2: self.msm::<Fp2>("bucket_sum_g2", &pb2, 128, witness, pack::unpack_jac_g2),
            c: self.msm("bucket_sum_g1", &pc, 64, private, pack::unpack_jac_g1),
        };
        Ok(assemble(zkey, &msms, r, s))
    }
}

impl std::fmt::Debug for MetalProver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalProver({})", self.device.name())
    }
}

/// One kernel-check result: the kernel name and whether the device agreed
/// with the Rust bodies on a small input.
#[derive(Clone, Debug)]
pub struct KernelCheck {
    /// Kernel name.
    pub kernel: &'static str,
    /// Agreement with `risc0-groth16-core` / the oxide bodies.
    pub ok: bool,
    /// What differed, when it did.
    pub detail: String,
}

impl MetalProver {
    /// Run every kernel on a small input and compare with the Rust bodies —
    /// the first thing to run on a Mac, before any proof: it localises an MSL
    /// arithmetic or layout bug to one kernel.
    pub fn kernel_check(&self) -> Result<Vec<KernelCheck>> {
        use risc0_groth16_core::{coeff::GroupedCoeff, ntt};
        let mut out = Vec::new();
        let n = 64usize;
        // deterministic inputs
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let fr: Vec<Fr> = (0..n).map(|_| Fr::from_u64(next() >> 2)).collect();
        let fr2: Vec<Fr> = (0..n).map(|_| Fr::from_u64(next() >> 2)).collect();
        let fr3: Vec<Fr> = (0..n).map(|_| Fr::from_u64(next() >> 2)).collect();
        let (a, b, c) = (
            self.upload(&pack::pack_fr(&fr)),
            self.upload(&pack::pack_fr(&fr2)),
            self.upload(&pack::pack_fr(&fr3)),
        );
        let outb = self.alloc(n * 32);

        // pointwise_mul
        self.run(
            "pointwise_mul",
            n,
            &[Arg::Buf(&a), Arg::Buf(&b), Arg::Buf(&outb)],
        );
        let got = pack::unpack_fr(&self.read(&outb, n * 32));
        let want: Vec<Fr> = fr.iter().zip(&fr2).map(|(x, y)| x.mul(y)).collect();
        out.push(KernelCheck {
            kernel: "pointwise_mul",
            ok: got == want,
            detail: first_diff(&got, &want),
        });

        // pointwise_mul_sub
        self.run(
            "pointwise_mul_sub",
            n,
            &[Arg::Buf(&a), Arg::Buf(&b), Arg::Buf(&c), Arg::Buf(&outb)],
        );
        let got = pack::unpack_fr(&self.read(&outb, n * 32));
        let want: Vec<Fr> = fr
            .iter()
            .zip(&fr2)
            .zip(&fr3)
            .map(|((x, y), z)| x.mul(y).sub(z))
            .collect();
        out.push(KernelCheck {
            kernel: "pointwise_mul_sub",
            ok: got == want,
            detail: first_diff(&got, &want),
        });

        // pointwise_scale (table = fr2, k = fr3[0])
        let k = pack::fr_bytes(&fr3[0]);
        self.run(
            "pointwise_scale",
            n,
            &[Arg::Buf(&a), Arg::Buf(&b), Arg::Bytes(&k), Arg::Buf(&outb)],
        );
        let got = pack::unpack_fr(&self.read(&outb, n * 32));
        let want: Vec<Fr> = fr
            .iter()
            .zip(&fr2)
            .map(|(x, y)| x.mul(y).mul(&fr3[0]))
            .collect();
        out.push(KernelCheck {
            kernel: "pointwise_scale",
            ok: got == want,
            detail: first_diff(&got, &want),
        });

        // bit_reverse
        let lg = (n.trailing_zeros()).to_le_bytes();
        self.run(
            "bit_reverse",
            n,
            &[Arg::Buf(&a), Arg::Bytes(&lg), Arg::Buf(&outb)],
        );
        let got = pack::unpack_fr(&self.read(&outb, n * 32));
        let mut want = fr.clone();
        ntt::bit_reverse_permute(&mut want);
        out.push(KernelCheck {
            kernel: "bit_reverse",
            ok: got == want,
            detail: first_diff(&got, &want),
        });

        // full coset transform through h_to_coset vs the core transform
        let t = Tables::new(n);
        let tb = TransformBuffers {
            forward: self.upload(&pack::pack_fr(&t.forward)),
            inverse: self.upload(&pack::pack_fr(&t.inverse)),
            shift: self.upload(&pack::pack_fr(&t.shift_powers)),
        };
        let (x, y) = (self.upload(&pack::pack_fr(&fr)), self.alloc(n * 32));
        let res = self.h_to_coset(&x, &y, n, &t, &tb);
        let got = pack::unpack_fr(&self.read(res, n * 32));
        let mut want = fr.clone();
        ntt::h_to_coset(&mut want, &Fr::two_adic_root(t.lg_n + 1));
        out.push(KernelCheck {
            kernel: "ntt_stage (h_to_coset)",
            ok: got == want,
            detail: first_diff(&got, &want),
        });

        // scatter_group: 4 groups of 3 coefficients
        let coeffs: Vec<GroupedCoeff> = (0..12)
            .map(|i| GroupedCoeff {
                signal: (i * 5 % n) as u32,
                value: fr2[i],
            })
            .collect();
        let starts: Vec<u32> = vec![0, 3, 6, 9, 12];
        let cb = self.upload(&pack::pack_coeffs(&coeffs));
        let sb = self.upload(
            &starts
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let wb = self.upload(&pack::pack_fr(&fr));
        let ob = self.alloc(4 * 32);
        self.run(
            "scatter_group",
            4,
            &[Arg::Buf(&cb), Arg::Buf(&sb), Arg::Buf(&wb), Arg::Buf(&ob)],
        );
        let got = pack::unpack_fr(&self.read(&ob, 4 * 32));
        let want: Vec<Fr> = (0..4)
            .map(|g| {
                let mut s = Fr::ZERO;
                for co in &coeffs[starts[g] as usize..starts[g + 1] as usize] {
                    s = s.add(&fr[co.signal as usize].mul(&co.value));
                }
                s
            })
            .collect();
        out.push(KernelCheck {
            kernel: "scatter_group",
            ok: got == want,
            detail: first_diff(&got, &want),
        });

        // digits
        let canonical = self.upload(&pack::pack_canonical(&fr));
        let db = self.alloc(n * 4);
        let (window, w) = (3u32, WINDOW_BITS);
        self.run(
            "digits",
            n,
            &[
                Arg::Buf(&canonical),
                Arg::Bytes(&window.to_le_bytes()),
                Arg::Bytes(&w.to_le_bytes()),
                Arg::Buf(&db),
            ],
        );
        let got: Vec<u32> = self
            .read(&db, n * 4)
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let want: Vec<u32> = fr
            .iter()
            .map(|x| risc0_groth16_core::scalar::digit(&x.to_canonical(), window, w) as u32)
            .collect();
        out.push(KernelCheck {
            kernel: "digits",
            ok: got == want,
            detail: if got == want {
                String::new()
            } else {
                format!("{got:?} vs {want:?}")
            },
        });

        // bucket sums on G1 and G2 with a few multiples of the generator-derived points from the fixture is not
        // available here; use points derived from the zkey-free path: scalar multiples of a fixed valid point.
        let g1: Vec<Affine<Fp>> = {
            let base = risc0_groth16_core::ec::Jacobian {
                x: Fp::from_u64(1),
                y: Fp::from_u64(2),
                z: Fp::ONE,
            };
            (1..=8u64)
                .map(|k| base.mul(&Fr::from_u64(k)).to_affine())
                .collect()
        };
        let order: Vec<u32> = (0..8).collect();
        let bstarts: Vec<u32> = vec![0, 3, 5, 8];
        let pb = self.upload(&pack::pack_g1(&g1));
        let ordb = self.upload(
            &order
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let stb = self.upload(
            &bstarts
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let sums = self.alloc(3 * 96);
        self.run(
            "bucket_sum_g1",
            3,
            &[
                Arg::Buf(&pb),
                Arg::Buf(&ordb),
                Arg::Buf(&stb),
                Arg::Buf(&sums),
            ],
        );
        let got = pack::unpack_jac_g1(&self.read(&sums, 3 * 96));
        let want: Vec<Jacobian<Fp>> = (0..3)
            .map(|b| {
                let mut acc = Jacobian::INFINITY;
                for &i in &order[bstarts[b] as usize..bstarts[b + 1] as usize] {
                    acc = acc.add_affine(&g1[i as usize]);
                }
                acc
            })
            .collect();
        let ok = got.iter().zip(&want).all(|(g, w)| g == w);
        out.push(KernelCheck {
            kernel: "bucket_sum_g1",
            ok,
            detail: if ok {
                String::new()
            } else {
                "bucket sums differ (projective equality)".into()
            },
        });
        Ok(out)
    }
}

fn first_diff(got: &[Fr], want: &[Fr]) -> String {
    match got.iter().zip(want).position(|(g, w)| g != w) {
        None if got.len() == want.len() => String::new(),
        None => format!("length {} vs {}", got.len(), want.len()),
        Some(i) => format!(
            "first difference at index {i}: got {:?}, want {:?}",
            got[i], want[i]
        ),
    }
}

/// A convenience: prove once on the default device.
pub fn prove(zkey: &Zkey, witness: &[Fr], r: &Fr, s: &Fr) -> Result<Proof> {
    MetalProver::new()
        .context("Metal device")?
        .prove(zkey, witness, r, s)
}
