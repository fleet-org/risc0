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

//! The CUDA arm's device module: one `#[kernel]` per entry of
//! `risc0_groth16_oxide::abi::KERNELS`, each a wrapper that computes its
//! output index from the thread index and calls the corresponding body in
//! `risc0_groth16_oxide::kernels` — the bodies already proven byte-identical
//! to the reference prover on the production circuit through the CPU
//! launcher (`oxide-cpu`). Nothing here does arithmetic; a wrapper that did
//! would be a second implementation.
//!
//! The parameter list of every kernel is the ABI's, in the ABI's order
//! (output first, then `n`, then the body's inputs with each slice as a
//! `(ptr, len)` pair). The host (`risc0-groth16-cuda`) passes exactly that.
//!
//! TWO POINTS ARE UNVERIFIED and are the first things to settle on a CUDA host
//! (both are one-line fixes if wrong; the ABI does not move):
//! 1. the intrinsic that yields the global thread index ([`gid`]);
//! 2. whether `#[kernel]` accepts raw-pointer parameters as written; if it
//!    wants cuda-oxide's own slice types, wrap the pointer/len pairs there and
//!    keep the ABI order.

#![no_std]

use cuda_macros::{cuda_module, kernel};

/// Global thread index along x. UNVERIFIED: the exact intrinsic path in the
/// pinned cuda-oxide tree; see `crates/cuda-device` and the examples there.
#[inline(always)]
fn gid() -> u32 {
    cuda_device::intrinsics::block_idx_x() * cuda_device::intrinsics::block_dim_x()
        + cuda_device::intrinsics::thread_idx_x()
}

/// A device slice from an ABI `(ptr, len)` pair.
///
/// # Safety
/// `ptr` must point at `len` initialised `T`s that outlive the launch.
#[inline(always)]
unsafe fn sl<'a, T>(ptr: *const T, len: u32) -> &'a [T] {
    core::slice::from_raw_parts(ptr, len as usize)
}

#[cuda_module]
pub mod groth16 {
    use risc0_groth16_core::{
        ec::{Affine, Jacobian},
        field::{Fp, Fr},
        fp2::Fp2,
    };
    use risc0_groth16_oxide::{abi::GroupedCoeff, kernels};

    use super::{gid, sl};

    /// `abi::SCATTER_GROUP`
    #[kernel]
    pub unsafe fn scatter_group(
        out: *mut Fr,
        n: u32,
        coeffs: *const GroupedCoeff,
        coeffs_len: u32,
        starts: *const u32,
        starts_len: u32,
        witness: *const Fr,
        witness_len: u32,
    ) {
        let i = gid();
        if i < n {
            *out.add(i as usize) = kernels::scatter_group(
                i as usize,
                sl(coeffs, coeffs_len),
                sl(starts, starts_len),
                sl(witness, witness_len),
            );
        }
    }

    /// `abi::POINTWISE_MUL`
    #[kernel]
    pub unsafe fn pointwise_mul(
        out: *mut Fr,
        n: u32,
        a: *const Fr,
        a_len: u32,
        b: *const Fr,
        b_len: u32,
    ) {
        let i = gid();
        if i < n {
            *out.add(i as usize) = kernels::pointwise_mul(i as usize, sl(a, a_len), sl(b, b_len));
        }
    }

    /// `abi::POINTWISE_MUL_SUB`
    #[kernel]
    pub unsafe fn pointwise_mul_sub(
        out: *mut Fr,
        n: u32,
        a: *const Fr,
        a_len: u32,
        b: *const Fr,
        b_len: u32,
        c: *const Fr,
        c_len: u32,
    ) {
        let i = gid();
        if i < n {
            *out.add(i as usize) =
                kernels::pointwise_mul_sub(i as usize, sl(a, a_len), sl(b, b_len), sl(c, c_len));
        }
    }

    /// `abi::POINTWISE_SCALE`: `a[i]·k[i]·n_inv`.
    #[kernel]
    pub unsafe fn pointwise_scale(
        out: *mut Fr,
        n: u32,
        a: *const Fr,
        a_len: u32,
        k: *const Fr,
        k_len: u32,
        n_inv: Fr,
    ) {
        let i = gid();
        if i < n {
            *out.add(i as usize) =
                kernels::scale(0, &[kernels::pointwise_scale(i as usize, sl(a, a_len), sl(k, k_len))], &n_inv);
        }
    }

    /// `abi::BIT_REVERSE`
    #[kernel]
    pub unsafe fn bit_reverse(out: *mut Fr, n: u32, a: *const Fr, a_len: u32, lg_n: u32) {
        let i = gid();
        if i < n {
            *out.add(i as usize) = kernels::bit_reverse(i as usize, sl(a, a_len), lg_n);
        }
    }

    /// `abi::NTT_STAGE`
    #[kernel]
    pub unsafe fn ntt_stage(
        out: *mut Fr,
        n: u32,
        a: *const Fr,
        a_len: u32,
        len: u32,
        twiddles: *const Fr,
        twiddles_len: u32,
        stride: u32,
    ) {
        let i = gid();
        if i < n {
            *out.add(i as usize) = kernels::ntt_stage(
                i as usize,
                sl(a, a_len),
                len as usize,
                sl(twiddles, twiddles_len),
                stride as usize,
            );
        }
    }

    /// `abi::DIGITS`
    #[kernel]
    pub unsafe fn digits(
        out: *mut u32,
        n: u32,
        scalars: *const [u64; 4],
        scalars_len: u32,
        window: u32,
        w: u32,
    ) {
        let i = gid();
        if i < n {
            *out.add(i as usize) = kernels::digit(i as usize, sl(scalars, scalars_len), window, w);
        }
    }

    /// `abi::BUCKET_SUM_G1`
    #[kernel]
    pub unsafe fn bucket_sum_g1(
        out: *mut Jacobian<Fp>,
        n: u32,
        points: *const Affine<Fp>,
        points_len: u32,
        order: *const u32,
        order_len: u32,
        starts: *const u32,
        starts_len: u32,
    ) {
        let b = gid();
        if b < n {
            *out.add(b as usize) = kernels::bucket_sum(
                b as usize,
                sl(points, points_len),
                sl(order, order_len),
                sl(starts, starts_len),
            );
        }
    }

    /// `abi::BUCKET_SUM_G2`
    #[kernel]
    pub unsafe fn bucket_sum_g2(
        out: *mut Jacobian<Fp2>,
        n: u32,
        points: *const Affine<Fp2>,
        points_len: u32,
        order: *const u32,
        order_len: u32,
        starts: *const u32,
        starts_len: u32,
    ) {
        let b = gid();
        if b < n {
            *out.add(b as usize) = kernels::bucket_sum(
                b as usize,
                sl(points, points_len),
                sl(order, order_len),
                sl(starts, starts_len),
            );
        }
    }
}
