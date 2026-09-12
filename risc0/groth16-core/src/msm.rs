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

//! Multi-scalar multiplication `Σ kᵢ·Pᵢ`: a naive form that is obviously right
//! and the windowed bucket method (Pippenger) the GPU arms implement, so the
//! reference and the kernels share the algorithm and the tests share the
//! oracle.

#![allow(clippy::needless_range_loop)]

use crate::{
    ec::{Affine, Jacobian},
    field::{Field, Fr},
};

/// `Σ kᵢ·Pᵢ` by independent scalar multiplications; the oracle for
/// [`msm_pippenger`] and for kernels.
pub fn msm_naive<F: Field>(points: &[Affine<F>], scalars: &[Fr]) -> Jacobian<F> {
    assert_eq!(
        points.len(),
        scalars.len(),
        "points and scalars must pair up"
    );
    let mut acc = Jacobian::INFINITY;
    for (p, k) in points.iter().zip(scalars) {
        acc = acc.add(&p.to_jacobian().mul(k));
    }
    acc
}

pub use crate::scalar::digit;

/// `Σ kᵢ·Pᵢ` by the windowed bucket method with `w`-bit windows
/// (`1 <= w <= 20`): per window, add every point to the bucket of its digit,
/// reduce the buckets by a running sum, and combine windows by Horner's rule.
pub fn msm_pippenger<F: Field>(points: &[Affine<F>], scalars: &[Fr], w: u32) -> Jacobian<F> {
    assert_eq!(
        points.len(),
        scalars.len(),
        "points and scalars must pair up"
    );
    assert!((1..=20).contains(&w), "window width out of range");
    let canonical: Vec<[u64; 4]> = scalars.iter().map(Fr::to_canonical).collect();
    let windows = 256u32.div_ceil(w);
    let mut result = Jacobian::INFINITY;
    for window in (0..windows).rev() {
        for _ in 0..w {
            result = result.double();
        }
        let mut buckets = vec![Jacobian::<F>::INFINITY; (1usize << w) - 1];
        for (p, k) in points.iter().zip(&canonical) {
            let d = digit(k, window, w);
            if d != 0 {
                buckets[(d - 1) as usize] = buckets[(d - 1) as usize].add_affine(p);
            }
        }
        // Σ d·B_d as a running sum from the top bucket down.
        let mut running = Jacobian::INFINITY;
        let mut window_sum = Jacobian::INFINITY;
        for b in buckets.iter().rev() {
            running = running.add(b);
            window_sum = window_sum.add(&running);
        }
        result = result.add(&window_sum);
    }
    result
}

#[cfg(test)]
mod tests {
    use ark_ec::{CurveGroup as _, PrimeGroup as _, VariableBaseMSM as _};

    use super::*;
    use crate::ec::conv::*;

    fn g1_points(n: usize, seed: u64) -> (Vec<Affine<crate::field::Fp>>, Vec<ark_bn254::G1Affine>) {
        let mut s = Scalars(seed);
        let g = ark_bn254::G1Projective::generator();
        let ark: Vec<_> = (0..n).map(|_| (g * s.next_fr()).into_affine()).collect();
        (ark.iter().copied().map(g1_from_ark).collect(), ark)
    }

    fn scalars(n: usize, seed: u64) -> (Vec<Fr>, Vec<ark_bn254::Fr>) {
        let mut s = Scalars(seed);
        let ark: Vec<_> = (0..n).map(|_| s.next_fr()).collect();
        (ark.iter().copied().map(fr_from_ark).collect(), ark)
    }

    #[test]
    fn digits_reassemble_the_scalar() {
        let k = [
            0x0123_4567_89ab_cdefu64,
            0xfedc_ba98_7654_3210,
            0x0f0f_0f0f_0f0f_0f0f,
            0x1234_5678_9abc_def0,
        ];
        for w in [1u32, 3, 8, 13, 16, 20] {
            let windows = 256u32.div_ceil(w);
            let mut acc = [0u64; 4];
            for window in (0..windows).rev() {
                // acc = acc * 2^w + digit
                let mut carry = digit(&k, window, w);
                for limb in acc.iter_mut() {
                    let wide = ((*limb as u128) << w) | (carry as u128);
                    *limb = wide as u64;
                    carry = (wide >> 64) as u64;
                }
            }
            assert_eq!(acc, k, "w = {w}");
        }
    }

    #[test]
    fn pippenger_equals_naive_equals_arkworks_on_g1() {
        for (n, w) in [(1usize, 4u32), (7, 3), (64, 8), (200, 13)] {
            let (pts, ark_pts) = g1_points(n, 11 + n as u64);
            let (ks, ark_ks) = scalars(n, 23 + n as u64);
            let naive = msm_naive(&pts, &ks);
            let fast = msm_pippenger(&pts, &ks, w);
            let expected = ark_bn254::G1Projective::msm(&ark_pts, &ark_ks)
                .unwrap()
                .into_affine();
            assert_eq!(g1_proj_to_ark(&naive), expected, "naive n = {n}");
            assert_eq!(
                g1_proj_to_ark(&fast),
                expected,
                "pippenger n = {n}, w = {w}"
            );
            assert_eq!(fast, naive);
        }
    }

    #[test]
    fn pippenger_matches_arkworks_on_g2() {
        let n = 24;
        let mut s = Scalars(0x5151);
        let g = ark_bn254::G2Projective::generator();
        let ark_pts: Vec<_> = (0..n).map(|_| (g * s.next_fr()).into_affine()).collect();
        let pts: Vec<_> = ark_pts.iter().copied().map(g2_from_ark).collect();
        let (ks, ark_ks) = scalars(n, 0x7a7a);
        let expected = ark_bn254::G2Projective::msm(&ark_pts, &ark_ks)
            .unwrap()
            .into_affine();
        assert_eq!(g2_proj_to_ark(&msm_pippenger(&pts, &ks, 6)), expected);
        assert_eq!(g2_proj_to_ark(&msm_naive(&pts, &ks)), expected);
    }

    #[test]
    fn zero_scalars_and_infinity_points_contribute_nothing() {
        let (pts, _) = g1_points(5, 3);
        let zeros = vec![Fr::ZERO; 5];
        assert!(msm_pippenger(&pts, &zeros, 8).is_infinity());
        assert!(msm_naive(&pts, &zeros).is_infinity());
        let (ks, _) = scalars(5, 4);
        let infs = vec![Affine::<crate::field::Fp>::INFINITY; 5];
        assert!(msm_pippenger(&infs, &ks, 8).is_infinity());
        let mut mixed = pts.clone();
        mixed[2] = Affine::INFINITY;
        let mut ks2 = ks.clone();
        ks2[2] = Fr::ZERO;
        assert_eq!(msm_pippenger(&mixed, &ks, 8), msm_pippenger(&pts, &ks2, 8));
    }
}
