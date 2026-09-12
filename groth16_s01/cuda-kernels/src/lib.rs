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
//! `(ptr, len: u32)` pair). cuda-oxide passes a raw pointer as one `.u64`
//! param, a `u32` as one `.u32` param and a `#[repr(C)]` struct by value as
//! one byval param (cuda-oxide `crates/mir-lower/src/convert/types/func_abi.rs`),
//! which is what the host (`risc0-groth16-cuda`) stages per parameter. The
//! shape is the pinned examples' (`examples/interop_cubin_identity`,
//! `examples/cutile_inter_kernel/simt`): raw-pointer kernels in a
//! `#[cuda_module]`, `thread::index_1d().get()` for the global index of a
//! 1-D launch, and the `n` bound check.

#![no_std]

use cuda_device::{cuda_module, kernel, thread};

/// A device slice from an ABI `(ptr, len)` pair — a plain helper (the collector
/// compiles every function a kernel reaches; only thread-index readers need
/// `#[device]`).
///
/// # Safety
/// `ptr` must point at `len` initialised `T`s that outlive the launch.
#[inline(always)]
unsafe fn sl<'a, T>(ptr: *const T, len: u32) -> &'a [T] {
    // SAFETY: the caller's contract, restated from the ABI.
    unsafe { core::slice::from_raw_parts(ptr, len as usize) }
}

#[cuda_module]
pub mod groth16 {
    use cuda_device::{kernel, thread};
    use risc0_groth16_core::{
        ec::{Affine, Jacobian},
        field::{Fp, Fr},
        fp2::Fp2,
    };
    use risc0_groth16_oxide::{abi::GroupedCoeff, kernels};

    use super::sl;

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
        let i = thread::index_1d().get();
        if i < n as usize {
            // SAFETY: the host sized `out` for `n` outputs and the inputs per the ABI.
            unsafe { *out.add(i) = kernels::scatter_group(i, sl(coeffs, coeffs_len), sl(starts, starts_len), sl(witness, witness_len)) };
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
        let i = thread::index_1d().get();
        if i < n as usize {
            // SAFETY: as above.
            unsafe { *out.add(i) = kernels::pointwise_mul(i, sl(a, a_len), sl(b, b_len)) };
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
        let i = thread::index_1d().get();
        if i < n as usize {
            // SAFETY: as above.
            unsafe { *out.add(i) = kernels::pointwise_mul_sub(i, sl(a, a_len), sl(b, b_len), sl(c, c_len)) };
        }
    }

    /// `abi::POINTWISE_SCALE`: `a[i]·k[i]·n_inv`. `n_inv` crosses as one
    /// 32-byte byval param (`Fr` is `#[repr(C)]`, not `repr(transparent)`).
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
        let i = thread::index_1d().get();
        if i < n as usize {
            // SAFETY: as above.
            unsafe { *out.add(i) = kernels::scale(0, &[kernels::pointwise_scale(i, sl(a, a_len), sl(k, k_len))], &n_inv) };
        }
    }

    /// `abi::BIT_REVERSE`
    #[kernel]
    pub unsafe fn bit_reverse(out: *mut Fr, n: u32, a: *const Fr, a_len: u32, lg_n: u32) {
        let i = thread::index_1d().get();
        if i < n as usize {
            // SAFETY: as above.
            unsafe { *out.add(i) = kernels::bit_reverse(i, sl(a, a_len), lg_n) };
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
        let i = thread::index_1d().get();
        if i < n as usize {
            // SAFETY: as above.
            unsafe { *out.add(i) = kernels::ntt_stage(i, sl(a, a_len), len as usize, sl(twiddles, twiddles_len), stride as usize) };
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
        let i = thread::index_1d().get();
        if i < n as usize {
            // SAFETY: as above.
            unsafe { *out.add(i) = kernels::digit(i, sl(scalars, scalars_len), window, w) };
        }
    }

    /// `abi::DIGITS_ALL`: every window in one launch (`n` here is the total
    /// `windows · scalars` output count; the body derives the window).
    #[kernel]
    pub unsafe fn digits_all(
        out: *mut u32,
        n: u32,
        scalars: *const [u64; 4],
        scalars_len: u32,
        w: u32,
    ) {
        let i = thread::index_1d().get();
        if i < n as usize {
            // SAFETY: as above.
            unsafe { *out.add(i) = kernels::digit_all(i, sl(scalars, scalars_len), w) };
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
        let b = thread::index_1d().get();
        if b < n as usize {
            // SAFETY: as above.
            unsafe { *out.add(b) = kernels::bucket_sum(b, sl(points, points_len), sl(order, order_len), sl(starts, starts_len)) };
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
        let b = thread::index_1d().get();
        if b < n as usize {
            // SAFETY: as above.
            unsafe { *out.add(b) = kernels::bucket_sum(b, sl(points, points_len), sl(order, order_len), sl(starts, starts_len)) };
        }
    }
}
