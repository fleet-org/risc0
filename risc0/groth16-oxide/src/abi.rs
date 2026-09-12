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

//! The contract between the CUDA arm's two halves: the device module (the
//! `#[kernel]` wrappers around [`crate::kernels`], compiled by `cargo oxide`
//! to PTX in `groth16_s01/cuda-kernels`) and the host prover
//! (`risc0-groth16-cuda`, which loads that module and launches by name).
//!
//! Both halves compile this file, so a change here is a change on both sides
//! or a type error. What it fixes:
//!
//! - **Records.** Every buffer element is a `#[repr(C)]` type of
//!   `risc0-groth16-core`: `Fr` (32 bytes, four little-endian u64 limbs,
//!   Montgomery form), `Affine<Fp>` (72: x, y, then the `infinity` flag and
//!   7 bytes of padding), `Affine<Fp2>` (136), `Jacobian<Fp>` (96),
//!   `Jacobian<Fp2>` (192), [`GroupedCoeff`] (40: `signal: u32` at 0,
//!   `value: Fr` at 8), canonical scalars `[u64; 4]`, and `u32`. No packing
//!   step exists on this arm: the bytes the host uploads are the bytes the
//!   kernel body reads, because both are the same Rust type — and the test
//!   below pins the sizes the driver will see.
//! - **Parameters.** Every kernel takes its output first (`out`, `n`), then
//!   its inputs in the order of the body's arguments, where a slice is a
//!   `(ptr, len: u32)` pair and a scalar is passed by value. Sizes that the
//!   bodies take as `usize` cross the boundary as `u32`.
//! - **Geometry.** One thread per output index, [`BLOCK`] threads per block,
//!   `ceil(n / BLOCK)` blocks; a thread with index `>= n` returns.
//! - **Names.** [`KERNELS`] is the complete list; the host loads every one at
//!   start-up so a module missing a kernel fails before any proof.

pub use risc0_groth16_core::coeff::GroupedCoeff;

/// Threads per block for every kernel.
pub const BLOCK: u32 = 256;

/// `out[g] = Σ witness[c.signal]·c.value` over group `g` of `coeffs`
/// (`starts[g]..starts[g+1]`). Params: `out, n, coeffs, coeffs_len, starts,
/// starts_len, witness, witness_len`.
pub const SCATTER_GROUP: &str = "scatter_group";
/// `out[i] = a[i]·b[i]`. Params: `out, n, a, a_len, b, b_len`.
pub const POINTWISE_MUL: &str = "pointwise_mul";
/// `out[i] = a[i]·b[i] − c[i]`. Params: `out, n, a, a_len, b, b_len, c, c_len`.
pub const POINTWISE_MUL_SUB: &str = "pointwise_mul_sub";
/// `out[i] = a[i]·k[i]·n_inv` (the coset shift and the inverse NTT's `1/n` in
/// one pass). Params: `out, n, a, a_len, k, k_len, n_inv: Fr`.
pub const POINTWISE_SCALE: &str = "pointwise_scale";
/// `out[i] = a[rev_lg(i)]`. Params: `out, n, a, a_len, lg_n: u32`.
pub const BIT_REVERSE: &str = "bit_reverse";
/// One radix-2 stage of length `len` with twiddle stride `stride`.
/// Params: `out, n, a, a_len, len: u32, twiddles, twiddles_len, stride: u32`.
pub const NTT_STAGE: &str = "ntt_stage";
/// `out[i] = digit(scalars[i], window, w)`. Params: `out, n, scalars,
/// scalars_len, window: u32, w: u32`; `scalars` are canonical `[u64; 4]`.
pub const DIGITS: &str = "digits";
/// `out[i] = digit(scalars[i % n], i / n, w)` for `i < windows · n` — every
/// window in one launch. Params: `out, n_total, scalars, scalars_len, w: u32`.
pub const DIGITS_ALL: &str = "digits_all";
/// `out[b] = Σ points[order[j]]` for `j` in `starts[b]..starts[b+1]`, G1.
/// Params: `out, n, points, points_len, order, order_len, starts, starts_len`.
/// With `order` the concatenation of every window's sorted indices and
/// `starts` flat over `(window, bucket)` (see `pipeline::sort_all_windows`),
/// one launch of `windows · buckets` outputs sums every window.
pub const BUCKET_SUM_G1: &str = "bucket_sum_g1";
/// The same over G2 (`Affine<Fp2>` in, `Jacobian<Fp2>` out).
pub const BUCKET_SUM_G2: &str = "bucket_sum_g2";

/// Every kernel the device module must export, by name.
pub const KERNELS: &[&str] = &[
    SCATTER_GROUP,
    POINTWISE_MUL,
    POINTWISE_MUL_SUB,
    POINTWISE_SCALE,
    BIT_REVERSE,
    NTT_STAGE,
    DIGITS,
    DIGITS_ALL,
    BUCKET_SUM_G1,
    BUCKET_SUM_G2,
];

/// Blocks for `n` outputs.
#[inline]
pub const fn blocks(n: u32) -> u32 {
    n.div_ceil(BLOCK)
}

#[cfg(test)]
mod tests {
    use super::*;
    use risc0_groth16_core::{
        ec::{Affine, Jacobian},
        field::{Fp, Fr},
        fp2::Fp2,
    };

    #[test]
    fn record_sizes_are_the_documented_ones() {
        assert_eq!(core::mem::size_of::<Fr>(), 32);
        assert_eq!(core::mem::size_of::<Affine<Fp>>(), 72);
        assert_eq!(core::mem::offset_of!(Affine<Fp>, infinity), 64);
        assert_eq!(core::mem::size_of::<Affine<Fp2>>(), 136);
        assert_eq!(core::mem::size_of::<Jacobian<Fp>>(), 96);
        assert_eq!(core::mem::size_of::<Jacobian<Fp2>>(), 192);
        assert_eq!(core::mem::size_of::<GroupedCoeff>(), 40);
        assert_eq!(core::mem::offset_of!(GroupedCoeff, signal), 0);
        assert_eq!(core::mem::offset_of!(GroupedCoeff, value), 8);
    }

    #[test]
    fn geometry() {
        assert_eq!(blocks(0), 0);
        assert_eq!(blocks(1), 1);
        assert_eq!(blocks(BLOCK), 1);
        assert_eq!(blocks(BLOCK + 1), 2);
        assert_eq!(KERNELS.len(), 10);
    }
}
