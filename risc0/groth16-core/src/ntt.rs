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

//! Number-theoretic transforms over `Fr` on power-of-two domains, in place and
//! allocation-free: the radix-2 Cooley–Tukey NTT (natural order in and out),
//! its inverse, and the coset scaling that moves a polynomial from `H` to
//! `g·H`. This is the sequence the canonical kernels run per polynomial —
//! inverse NTT, multiply by the shift powers, forward NTT — expressed once.

#![allow(clippy::needless_range_loop)]

use crate::field::{Field, Fr};

/// `log2(n)` for a power of two `n`.
pub fn lg2(n: usize) -> u32 {
    assert!(n.is_power_of_two(), "domain size must be a power of two");
    n.trailing_zeros()
}

/// Permute `a` into bit-reversed index order (an involution).
pub fn bit_reverse_permute(a: &mut [Fr]) {
    let n = a.len();
    if n <= 2 {
        return;
    }
    let bits = lg2(n);
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            a.swap(i, j);
        }
    }
}

/// In-place radix-2 Cooley–Tukey transform with the primitive `n`-th root
/// `omega`: `a[k] = Σ_j a[j]·omega^(jk)`. Natural order in and out.
pub fn ntt_in_place(a: &mut [Fr], omega: &Fr) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    bit_reverse_permute(a);
    let mut len = 2;
    while len <= n {
        // w_len = omega^(n/len) is a primitive len-th root of unity.
        let w_len = omega.pow(&[(n / len) as u64, 0, 0, 0]);
        let mut start = 0;
        while start < n {
            let mut w = Fr::ONE;
            for k in 0..len / 2 {
                let u = a[start + k];
                let v = a[start + k + len / 2].mul(&w);
                a[start + k] = u.add(&v);
                a[start + k + len / 2] = u.sub(&v);
                w = w.mul(&w_len);
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Coefficients → evaluations on `H = <omega_n>`.
pub fn forward(a: &mut [Fr]) {
    let omega = Fr::two_adic_root(lg2(a.len()));
    ntt_in_place(a, &omega);
}

/// Evaluations on `H` → coefficients (inverse transform, scaled by `1/n`).
pub fn inverse(a: &mut [Fr]) {
    let n = a.len();
    let omega_inv = Fr::two_adic_root(lg2(n))
        .inverse()
        .expect("a root of unity is invertible");
    ntt_in_place(a, &omega_inv);
    let n_inv = Fr::from_u64(n as u64)
        .inverse()
        .expect("n is nonzero in Fr");
    for x in a.iter_mut() {
        *x = x.mul(&n_inv);
    }
}

/// Multiply coefficient `i` by `shift^i`, so that a forward transform of the
/// result evaluates the polynomial on the coset `shift·H`.
pub fn coset_scale(a: &mut [Fr], shift: &Fr) {
    let mut power = Fr::ONE;
    for x in a.iter_mut() {
        *x = x.mul(&power);
        power = power.mul(shift);
    }
}

/// The canonical kernels' per-polynomial sequence: evaluations on `H` in,
/// evaluations on the coset `shift·H` out (inverse NTT, coset scale, forward
/// NTT).
pub fn h_to_coset(a: &mut [Fr], shift: &Fr) {
    inverse(a);
    coset_scale(a, shift);
    forward(a);
}

#[cfg(test)]
mod tests {
    use ark_poly::{EvaluationDomain as _, Radix2EvaluationDomain};

    use super::*;
    use crate::ec::conv::{fr_from_ark, fr_to_ark, Scalars};

    fn random_polys(n: usize, seed: u64) -> (Vec<Fr>, Vec<ark_bn254::Fr>) {
        let mut s = Scalars(seed | 1);
        let ark: Vec<ark_bn254::Fr> = (0..n).map(|_| s.next_fr()).collect();
        (ark.iter().copied().map(fr_from_ark).collect(), ark)
    }

    #[test]
    fn forward_matches_ark_poly_fft() {
        for n in [1usize, 2, 4, 8, 64, 1024] {
            let (mut mine, ark) = random_polys(n, 7 + n as u64);
            forward(&mut mine);
            let domain = Radix2EvaluationDomain::<ark_bn254::Fr>::new(n).unwrap();
            let expected = domain.fft(&ark);
            let got: Vec<_> = mine.iter().map(fr_to_ark).collect();
            assert_eq!(got, expected, "n = {n}");
        }
    }

    #[test]
    fn inverse_undoes_forward_and_matches_ark_poly_ifft() {
        for n in [1usize, 2, 16, 256] {
            let (original, ark) = random_polys(n, 99 + n as u64);
            let mut mine = original.clone();
            forward(&mut mine);
            inverse(&mut mine);
            assert_eq!(mine, original, "n = {n}");
            let mut evals = original.clone();
            inverse(&mut evals);
            let domain = Radix2EvaluationDomain::<ark_bn254::Fr>::new(n).unwrap();
            let expected = domain.ifft(&ark);
            let got: Vec<_> = evals.iter().map(fr_to_ark).collect();
            assert_eq!(got, expected, "n = {n}");
        }
    }

    #[test]
    fn forward_is_the_naive_dft() {
        let n = 8;
        let (mine, _) = random_polys(n, 5);
        let omega = Fr::two_adic_root(3);
        let mut fast = mine.clone();
        forward(&mut fast);
        for k in 0..n {
            let mut acc = Fr::ZERO;
            for (j, coeff) in mine.iter().enumerate() {
                acc = acc.add(&coeff.mul(&omega.pow(&[(j * k) as u64, 0, 0, 0])));
            }
            assert_eq!(fast[k], acc, "k = {k}");
        }
    }

    #[test]
    fn coset_evaluation_matches_ark_poly_coset_fft() {
        let n = 32;
        let (mut mine, ark) = random_polys(n, 31);
        let shift = Fr::two_adic_root(lg2(n) + 1); // the canonical kernels' shift: a 2n-th root
        coset_scale(&mut mine, &shift);
        forward(&mut mine);
        let domain = Radix2EvaluationDomain::<ark_bn254::Fr>::new(n)
            .unwrap()
            .get_coset(fr_to_ark(&shift))
            .unwrap();
        let expected = domain.fft(&ark);
        let got: Vec<_> = mine.iter().map(fr_to_ark).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn h_to_coset_moves_evaluations_between_domains() {
        // Evaluate a known polynomial on H, move to the coset, compare with direct evaluation.
        let n = 16;
        let (coeffs, _) = random_polys(n, 77);
        let shift = Fr::two_adic_root(lg2(n) + 1);
        let mut evals_h = coeffs.clone();
        forward(&mut evals_h);
        h_to_coset(&mut evals_h, &shift);
        let omega = Fr::two_adic_root(lg2(n));
        for k in 0..n {
            let x = shift.mul(&omega.pow(&[k as u64, 0, 0, 0]));
            let mut acc = Fr::ZERO;
            let mut xp = Fr::ONE;
            for c in &coeffs {
                acc = acc.add(&c.mul(&xp));
                xp = xp.mul(&x);
            }
            assert_eq!(evals_h[k], acc, "k = {k}");
        }
    }

    #[test]
    fn bit_reversal_is_an_involution() {
        let (original, _) = random_polys(64, 3);
        let mut a = original.clone();
        bit_reverse_permute(&mut a);
        assert_ne!(a, original);
        bit_reverse_permute(&mut a);
        assert_eq!(a, original);
    }
}
