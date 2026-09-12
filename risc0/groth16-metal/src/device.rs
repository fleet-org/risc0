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

    /// Evaluations on H (in `a`) → evaluations on the coset, through the kernels;
    /// `a`, `b` are ping-pong buffers of `n` elements. Returns the buffer holding the result.
    fn h_to_coset<'b>(
        &self,
        a: &'b Buffer,
        b: &'b Buffer,
        n: usize,
        t: &Tables,
        tb: &TransformBuffers,
    ) -> &'b Buffer {
        let (fwd, inv, shift) = (&tb.forward, &tb.inverse, &tb.shift);
        let lg = t.lg_n.to_le_bytes();
        // inverse NTT
        self.run(
            "bit_reverse",
            n,
            &[Arg::Buf(a), Arg::Bytes(&lg), Arg::Buf(b)],
        );
        let mut in_b = true;
        let mut len = 2usize;
        while len <= n {
            let (src, dst) = if in_b { (b, a) } else { (a, b) };
            self.run(
                "ntt_stage",
                n,
                &[
                    Arg::Buf(src),
                    Arg::Bytes(&(len as u32).to_le_bytes()),
                    Arg::Buf(inv),
                    Arg::Bytes(&((n / len) as u32).to_le_bytes()),
                    Arg::Buf(dst),
                ],
            );
            in_b = !in_b;
            len <<= 1;
        }
        let (src, dst) = if in_b { (b, a) } else { (a, b) };
        // coefficients · n⁻¹ · shift^i
        let n_inv = pack::fr_bytes(&t.n_inv);
        self.run(
            "pointwise_scale",
            n,
            &[
                Arg::Buf(src),
                Arg::Buf(shift),
                Arg::Bytes(&n_inv),
                Arg::Buf(dst),
            ],
        );
        let (a2, b2) = if in_b { (a, b) } else { (b, a) };
        // forward NTT from dst
        self.run(
            "bit_reverse",
            n,
            &[Arg::Buf(b2), Arg::Bytes(&lg), Arg::Buf(a2)],
        );
        let mut in_a = true;
        let mut len = 2usize;
        while len <= n {
            let (src, dst) = if in_a { (a2, b2) } else { (b2, a2) };
            self.run(
                "ntt_stage",
                n,
                &[
                    Arg::Buf(src),
                    Arg::Bytes(&(len as u32).to_le_bytes()),
                    Arg::Buf(fwd),
                    Arg::Bytes(&((n / len) as u32).to_le_bytes()),
                    Arg::Buf(dst),
                ],
            );
            in_a = !in_a;
            len <<= 1;
        }
        if in_a {
            a2
        } else {
            b2
        }
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

/// A convenience: prove once on the default device.
pub fn prove(zkey: &Zkey, witness: &[Fr], r: &Fr, s: &Fr) -> Result<Proof> {
    MetalProver::new()
        .context("Metal device")?
        .prove(zkey, witness, r, s)
}
