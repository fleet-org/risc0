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

//! Kernel bodies: one thread's work each, written as a function of the
//! output index. A `#[kernel]` wrapper on the GPU computes `i` from the
//! thread index and stores the result in its `DisjointSlice` slot; the CPU
//! launcher does the same with a chunked map. No body allocates, branches on
//! data it does not own, or writes anywhere but its own output.

use risc0_groth16_core::{
    ec::{Affine, Jacobian},
    field::{Field, Fr},
};

pub use risc0_groth16_core::coeff::GroupedCoeff as Coeff;

/// Scatter: group `g` of the coefficient list — all coefficients of one
/// constraint — reduces to that constraint's polynomial value:
/// `Σ w[signal]·value`. Groups are the `witness_into_poly` kernel's unit of
/// work; `starts` has one more entry than there are groups.
#[inline]
pub fn scatter_group(g: usize, coeffs: &[Coeff], starts: &[u32], witness: &[Fr]) -> Fr {
    let (from, to) = (starts[g] as usize, starts[g + 1] as usize);
    let mut sum = Fr::ZERO;
    for c in &coeffs[from..to] {
        sum = sum.add(&witness[c.signal as usize].mul(&c.value));
    }
    sum
}

/// `a[i]·b[i]`.
#[inline]
pub fn pointwise_mul(i: usize, a: &[Fr], b: &[Fr]) -> Fr {
    a[i].mul(&b[i])
}

/// `a[i]·b[i] − c[i]` — the quotient's coset evaluations.
#[inline]
pub fn pointwise_mul_sub(i: usize, a: &[Fr], b: &[Fr], c: &[Fr]) -> Fr {
    a[i].mul(&b[i]).sub(&c[i])
}

/// `a[i]·k[i]` — apply a per-index factor (coset shift powers).
#[inline]
pub fn pointwise_scale(i: usize, a: &[Fr], k: &[Fr]) -> Fr {
    a[i].mul(&k[i])
}

/// `a[i]·k` — one factor for every index (the `1/n` of an inverse NTT).
#[inline]
pub fn scale(i: usize, a: &[Fr], k: &Fr) -> Fr {
    a[i].mul(k)
}

/// Bit-reversal permutation: output `i` takes input `rev(i)`.
#[inline]
pub fn bit_reverse(i: usize, a: &[Fr], lg_n: u32) -> Fr {
    a[i.reverse_bits() >> (usize::BITS - lg_n)]
}

/// One radix-2 Cooley–Tukey stage, out of place: for stage length `len`
/// (a power of two ≥ 2) with the primitive `len`-th root's powers in
/// `twiddles[k·stride]` (`stride = n / len`, table of `omega^j` for `j < n/2`),
/// output `i` is the upper or lower half of its butterfly.
#[inline]
pub fn ntt_stage(i: usize, a: &[Fr], len: usize, twiddles: &[Fr], stride: usize) -> Fr {
    let half = len / 2;
    let k = i % len;
    if k < half {
        // u + w·v
        let u = a[i];
        let v = a[i + half].mul(&twiddles[k * stride]);
        u.add(&v)
    } else {
        // u − w·v
        let u = a[i - half];
        let v = a[i].mul(&twiddles[(k - half) * stride]);
        u.sub(&v)
    }
}

/// The `w`-bit digit `window` of scalar `i` (canonical limbs).
#[inline]
pub fn digit(i: usize, scalars: &[[u64; 4]], window: u32, w: u32) -> u32 {
    risc0_groth16_core::scalar::digit(&scalars[i], window, w) as u32
}

/// Every window's digits in one launch: output `i` is scalar `i % n`'s
/// digit for window `i / n` (`n = scalars.len()`), so the host reads one
/// buffer of `windows · n` digits instead of one per window.
#[inline]
pub fn digit_all(i: usize, scalars: &[[u64; 4]], w: u32) -> u32 {
    let n = scalars.len();
    digit(i % n, scalars, (i / n) as u32, w)
}

/// Bucket sum for bucket `b` (digit `b + 1`): the points whose digit is
/// `b + 1` occupy `order[starts[b]..starts[b + 1]]` after the host's
/// counting sort; add them into one Jacobian accumulator.
#[inline]
pub fn bucket_sum<F: Field>(
    b: usize,
    points: &[Affine<F>],
    order: &[u32],
    starts: &[u32],
) -> Jacobian<F> {
    let (from, to) = (starts[b] as usize, starts[b + 1] as usize);
    let mut acc = Jacobian::INFINITY;
    for &p in &order[from..to] {
        acc = acc.add_affine(&points[p as usize]);
    }
    acc
}

/// Sum of the Jacobian points `sums[starts[b]..starts[b + 1]]` — one level
/// of the bounded-chain reduction above `bucket_sum`
/// (`pipeline::plan_ranges`): the pieces of a bucket, then the pieces of
/// those, until one thread's chain is short everywhere.
#[inline]
pub fn jacobian_sum<F: Field>(b: usize, sums: &[Jacobian<F>], starts: &[u32]) -> Jacobian<F> {
    let (from, to) = (starts[b] as usize, starts[b + 1] as usize);
    let mut acc = Jacobian::INFINITY;
    for s in &sums[from..to] {
        acc = acc.add(s);
    }
    acc
}
