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

//! The per-kernel check, as data: one deterministic set of inputs and the
//! expected outputs computed with the Rust bodies, shared by every device
//! arm's `kernel_check`. An arm only uploads, launches, reads and compares —
//! so the cases (and their coverage) are the same on a Mac and on a CUDA
//! host, and a gap found on one is closed for both.
//!
//! Coverage, chosen from the Metal parity review: full-width scalars over
//! EVERY window of the MSM digit extraction (the cross-limb branch, the top
//! window); bucket sums with an affine point at infinity, an empty bucket,
//! `P + (−P)` and `P + P`, on G1 AND G2; an end-to-end MSM against the naive
//! sum; and a whole proof on the in-tree fixture against the core prover with
//! fixed blinding (byte-identical, as the CPU pipeline is tested).

use risc0_groth16_core::{
    coeff::GroupedCoeff,
    ec::{g1_generator, g2_generator, Affine, Jacobian},
    field::{Field, Fp, Fr},
    fp2::Fp2,
    msm::msm_naive,
    prover::{self, Proof},
    zkey::{parse_wtns, Zkey},
};

use crate::{kernels, pipeline::WINDOW_BITS};

/// Number of MSM windows for 256-bit scalars.
pub const WINDOWS: u32 = 256u32.div_ceil(WINDOW_BITS);

/// One kernel-check result.
#[derive(Clone, Debug)]
pub struct KernelCheck {
    /// Kernel (or composite step) name.
    pub kernel: &'static str,
    /// Agreement with the Rust bodies.
    pub ok: bool,
    /// What differed, when it did.
    pub detail: String,
}

impl KernelCheck {
    /// A result from a "what differed" string: empty means agreement.
    pub fn from_detail(kernel: &'static str, detail: String) -> Self {
        Self {
            kernel,
            ok: detail.is_empty(),
            detail,
        }
    }
}

/// The inputs.
pub struct Cases {
    /// Element count of the vector cases (a power of two).
    pub n: usize,
    /// `log2(n)`.
    pub lg_n: u32,
    /// Three vectors of small field elements (`< 2^62`).
    pub fr: Vec<Fr>,
    /// Second vector.
    pub fr2: Vec<Fr>,
    /// Third vector.
    pub fr3: Vec<Fr>,
    /// The by-value scalar of `pointwise_scale`.
    pub n_inv: Fr,
    /// `n` full-width scalars (`< 2^253`), for `digits` and the MSM.
    pub wide: Vec<Fr>,
    /// Scatter: `n` coefficients in 8 groups of `n / 8`.
    pub coeffs: Vec<GroupedCoeff>,
    /// Scatter group starts (9 entries).
    pub starts: Vec<u32>,
    /// NTT stage length under test.
    pub stage_len: usize,
    /// Its twiddle stride (`n / stage_len`).
    pub stage_stride: usize,
    /// G1 points: `1G..=8G`, the point at infinity, `−G`.
    pub g1: Vec<Affine<Fp>>,
    /// G2 points, the same shape.
    pub g2: Vec<Affine<Fp2>>,
    /// Bucket order over the 10 points.
    pub order: Vec<u32>,
    /// Bucket starts: `{G,2G,3G} {∞} {G,−G} {} {5G..8G}`.
    pub bstarts: Vec<u32>,
    /// Ranges over the bucket sums for `jacobian_sum`: `{B0,B1} {} {B2,B3,B4}`
    /// (a two-term sum, an empty range, a run through infinity).
    pub jstarts: Vec<u32>,
    /// 64 G1 points for the end-to-end MSM.
    pub msm_g1: Vec<Affine<Fp>>,
    /// 64 G2 points for the end-to-end MSM.
    pub msm_g2: Vec<Affine<Fp2>>,
    /// Their scalars (the first 64 of `wide`).
    pub msm_scalars: Vec<Fr>,
}

impl Default for Cases {
    fn default() -> Self {
        Self::new()
    }
}

impl Cases {
    /// The deterministic inputs (xorshift-seeded).
    pub fn new() -> Self {
        let n = 64usize;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let small = |next: &mut dyn FnMut() -> u64| -> Vec<Fr> {
            (0..n).map(|_| Fr::from_u64(next() >> 2)).collect()
        };
        let fr = small(&mut next);
        let fr2 = small(&mut next);
        let fr3 = small(&mut next);
        let wide: Vec<Fr> = (0..n)
            .map(|_| Fr::from_canonical([next(), next(), next(), next() >> 3]))
            .collect();
        let coeffs: Vec<GroupedCoeff> = (0..n)
            .map(|j| GroupedCoeff {
                signal: ((j * 7) % n) as u32,
                value: fr2[j],
            })
            .collect();
        let starts: Vec<u32> = (0..=8).map(|g| (g * n / 8) as u32).collect();
        let g1_base = g1_generator().to_jacobian();
        let g2_base = g2_generator().to_jacobian();
        let mut g1: Vec<Affine<Fp>> = (1..=8u64)
            .map(|k| g1_base.mul(&Fr::from_u64(k)).to_affine())
            .collect();
        g1.push(Affine::INFINITY);
        g1.push(g1[0].neg());
        let mut g2: Vec<Affine<Fp2>> = (1..=8u64)
            .map(|k| g2_base.mul(&Fr::from_u64(k)).to_affine())
            .collect();
        g2.push(Affine::INFINITY);
        g2.push(g2[0].neg());
        let msm_g1 = (1..=64u64)
            .map(|k| g1_base.mul(&Fr::from_u64(3 * k + 1)).to_affine())
            .collect();
        let msm_g2 = (1..=64u64)
            .map(|k| g2_base.mul(&Fr::from_u64(3 * k + 1)).to_affine())
            .collect();
        Self {
            n,
            lg_n: n.trailing_zeros(),
            n_inv: fr3[0],
            msm_scalars: wide[..64].to_vec(),
            fr,
            fr2,
            fr3,
            wide,
            coeffs,
            starts,
            stage_len: 8,
            stage_stride: n / 8,
            g1,
            g2,
            order: vec![0, 1, 2, 8, 0, 9, 4, 5, 6, 7],
            bstarts: vec![0, 3, 4, 6, 6, 10],
            jstarts: vec![0, 2, 2, 5],
            msm_g1,
            msm_g2,
        }
    }

    /// Number of buckets in the bucket-sum case.
    pub fn buckets(&self) -> usize {
        self.bstarts.len() - 1
    }

    /// Expected `scatter_group` output (8 groups).
    pub fn scatter(&self) -> Vec<Fr> {
        (0..8)
            .map(|g| kernels::scatter_group(g, &self.coeffs, &self.starts, &self.fr))
            .collect()
    }

    /// Expected `pointwise_mul(fr, fr2)`.
    pub fn pointwise_mul(&self) -> Vec<Fr> {
        (0..self.n)
            .map(|i| kernels::pointwise_mul(i, &self.fr, &self.fr2))
            .collect()
    }

    /// Expected `pointwise_mul_sub(fr, fr2, fr3)`.
    pub fn pointwise_mul_sub(&self) -> Vec<Fr> {
        (0..self.n)
            .map(|i| kernels::pointwise_mul_sub(i, &self.fr, &self.fr2, &self.fr3))
            .collect()
    }

    /// Expected `pointwise_scale(fr, fr2) · n_inv`.
    pub fn pointwise_scale(&self) -> Vec<Fr> {
        (0..self.n)
            .map(|i| kernels::pointwise_scale(i, &self.fr, &self.fr2).mul(&self.n_inv))
            .collect()
    }

    /// Expected `bit_reverse(fr, lg_n)`.
    pub fn bit_reverse(&self) -> Vec<Fr> {
        (0..self.n)
            .map(|i| kernels::bit_reverse(i, &self.fr, self.lg_n))
            .collect()
    }

    /// Expected `ntt_stage(fr, stage_len, fr2, stage_stride)`.
    pub fn ntt_stage(&self) -> Vec<Fr> {
        (0..self.n)
            .map(|i| kernels::ntt_stage(i, &self.fr, self.stage_len, &self.fr2, self.stage_stride))
            .collect()
    }

    /// Canonical limbs of `wide`, the `digits` input.
    pub fn wide_canonical(&self) -> Vec<[u64; 4]> {
        self.wide.iter().map(Fr::to_canonical).collect()
    }

    /// Expected `digits(wide, window)`.
    pub fn digits(&self, window: u32) -> Vec<u32> {
        let canonical = self.wide_canonical();
        (0..self.n)
            .map(|i| kernels::digit(i, &canonical, window, WINDOW_BITS))
            .collect()
    }

    /// Expected `digits_all(wide)`: every window's digits in one buffer.
    pub fn digits_all(&self) -> Vec<u32> {
        let canonical = self.wide_canonical();
        (0..WINDOWS as usize * self.n)
            .map(|i| kernels::digit_all(i, &canonical, WINDOW_BITS))
            .collect()
    }

    /// Expected G1 bucket sums.
    pub fn bucket_sums_g1(&self) -> Vec<Jacobian<Fp>> {
        (0..self.buckets())
            .map(|b| kernels::bucket_sum(b, &self.g1, &self.order, &self.bstarts))
            .collect()
    }

    /// Expected G2 bucket sums.
    pub fn bucket_sums_g2(&self) -> Vec<Jacobian<Fp2>> {
        (0..self.buckets())
            .map(|b| kernels::bucket_sum(b, &self.g2, &self.order, &self.bstarts))
            .collect()
    }

    /// Expected G1 `jacobian_sum` over the G1 bucket sums with `jstarts`.
    pub fn jacobian_sums_g1(&self) -> Vec<Jacobian<Fp>> {
        let sums = self.bucket_sums_g1();
        (0..self.jstarts.len() - 1)
            .map(|b| kernels::jacobian_sum(b, &sums, &self.jstarts))
            .collect()
    }

    /// Expected G2 `jacobian_sum` over the G2 bucket sums with `jstarts`.
    pub fn jacobian_sums_g2(&self) -> Vec<Jacobian<Fp2>> {
        let sums = self.bucket_sums_g2();
        (0..self.jstarts.len() - 1)
            .map(|b| kernels::jacobian_sum(b, &sums, &self.jstarts))
            .collect()
    }

    /// Expected G1 MSM (naive sum).
    pub fn msm_g1(&self) -> Jacobian<Fp> {
        msm_naive(&self.msm_g1, &self.msm_scalars)
    }

    /// Expected G2 MSM (naive sum).
    pub fn msm_g2(&self) -> Jacobian<Fp2> {
        msm_naive(&self.msm_g2, &self.msm_scalars)
    }
}

/// The in-tree circom fixture (`multiplier2`) with fixed blinding: a whole
/// proof through an arm must equal the core prover's, byte for byte.
pub struct Fixture {
    /// The zkey.
    pub zkey: Zkey,
    /// The witness.
    pub witness: Vec<Fr>,
    /// Blinding `r`.
    pub r: Fr,
    /// Blinding `s`.
    pub s: Fr,
}

impl Fixture {
    /// `multiplier2_final.zkey` + `multiplier2.wtns`, `r = 7`, `s = 11`.
    pub fn multiplier2() -> Self {
        const ZKEY: &[u8] =
            include_bytes!("../../../groth16_proof/circom-compat/test/data/multiplier2_final.zkey");
        const WTNS: &[u8] =
            include_bytes!("../../../groth16_proof/circom-compat/test/data/multiplier2.wtns");
        Self {
            zkey: Zkey::parse(ZKEY).expect("in-tree fixture zkey parses"),
            witness: parse_wtns(WTNS).expect("in-tree fixture witness parses"),
            r: Fr::from_u64(7),
            s: Fr::from_u64(11),
        }
    }

    /// The core prover's proof for these inputs.
    pub fn expected(&self) -> Proof {
        prover::prove(&self.zkey, &self.witness, &self.r, &self.s)
            .expect("core prover on the fixture")
    }
}

/// The first index where two vectors differ, as a message; empty when equal.
pub fn first_diff<T: PartialEq + core::fmt::Debug>(got: &[T], want: &[T]) -> String {
    if got.len() != want.len() {
        return format!("{} outputs, expected {}", got.len(), want.len());
    }
    match got.iter().zip(want).position(|(g, w)| g != w) {
        Some(i) => format!(
            "first difference at index {i}: got {:?}, want {:?}",
            got[i], want[i]
        ),
        None => String::new(),
    }
}

/// Projective comparison of point lists (affine forms compared); empty when
/// equal.
pub fn first_point_diff<F: Field + core::fmt::Debug>(
    got: &[Jacobian<F>],
    want: &[Jacobian<F>],
) -> String {
    let ga: Vec<Affine<F>> = got.iter().map(Jacobian::to_affine).collect();
    let wa: Vec<Affine<F>> = want.iter().map(Jacobian::to_affine).collect();
    match first_diff(&ga, &wa) {
        s if s.is_empty() => s,
        s => format!("{s} (affine forms compared)"),
    }
}

/// The fixture proof compared with the core prover's; empty when equal.
pub fn proof_diff(got: &Proof, fixture: &Fixture) -> String {
    let want = fixture.expected();
    if *got == want {
        String::new()
    } else {
        format!("proof differs from the core prover's on the fixture: got {got:?}, want {want:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_are_self_consistent() {
        let c = Cases::new();
        assert_eq!(c.g1.len(), 10);
        assert!(c.g1[8].infinity);
        assert_eq!(c.g1[9], c.g1[0].neg());
        assert_eq!(c.buckets(), 5);
        let sums = c.bucket_sums_g1();
        // {∞}, {G,−G}, {} are all infinity; {G,2G,3G} = 6G
        assert!(sums[1].is_infinity() && sums[2].is_infinity() && sums[3].is_infinity());
        assert_eq!(
            sums[0].to_affine(),
            g1_generator()
                .to_jacobian()
                .mul(&Fr::from_u64(6))
                .to_affine()
        );
        assert!(!c.bucket_sums_g2()[0].is_infinity());
        // every window has a non-zero digit somewhere on full-width scalars
        for w in 0..WINDOWS {
            assert!(c.digits(w).iter().any(|&d| d != 0), "window {w}");
        }
        // digits_all is the windows' digits concatenated
        let all = c.digits_all();
        for w in 0..WINDOWS as usize {
            assert_eq!(&all[w * c.n..(w + 1) * c.n], &c.digits(w as u32)[..]);
        }
    }

    #[test]
    fn fixture_proof_is_reproducible() {
        let f = Fixture::multiplier2();
        assert_eq!(f.expected(), f.expected());
        assert!(proof_diff(&f.expected(), &f).is_empty());
    }
}
