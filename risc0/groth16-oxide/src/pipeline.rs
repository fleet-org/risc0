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

//! The host pipeline: it owns buffers, the coefficient grouping, the
//! twiddle and shift tables, the per-window counting sort, and the launch
//! sequence. It launches only the bodies in [`crate::kernels`], in the order
//! the canonical kernels run them (`BOUNDARY.md` §2), and hands the five MSM
//! results to the shared assembly.

use risc0_groth16_core::{
    ec::{Affine, Jacobian},
    field::{Field, Fr},
    prover::{assemble, counting_sort_by_digit, horner, reduce_buckets, Msms, Proof, ProveError},
    zkey::Zkey,
};

use crate::{kernels, launch::Launcher};

/// Pippenger window width used by this arm.
pub const WINDOW_BITS: u32 = 12;

pub use risc0_groth16_core::prover::{CoefficientGroups, Tables};

/// Evaluate one matrix's polynomial on the domain from its groups.
fn scatter<L: Launcher>(
    l: &L,
    coeffs: &[kernels::Coeff],
    starts: &[u32],
    constraint: &[u32],
    witness: &[Fr],
    domain: usize,
) -> Vec<Fr> {
    let mut sums = vec![Fr::ZERO; constraint.len()];
    l.map(&mut sums, |g| {
        kernels::scatter_group(g, coeffs, starts, witness)
    });
    let mut poly = vec![Fr::ZERO; domain];
    for (c, v) in constraint.iter().zip(sums) {
        poly[*c as usize] = v;
    }
    poly
}

/// Evaluations on `H` → evaluations on the coset `shift·H` (inverse NTT,
/// scale by `1/n`, coset scale, forward NTT), through the kernels.
pub fn h_to_coset<L: Launcher>(l: &L, evals: Vec<Fr>, t: &Tables) -> Vec<Fr> {
    let mut scratch = vec![Fr::ZERO; evals.len()];
    h_to_coset_with(l, evals, &mut scratch, t)
}

/// [`h_to_coset`] with a caller-owned scratch buffer of the same length (so a
/// long run keeps one scratch allocation): executes [`schedule::coset`] over
/// the two buffers and returns the one the schedule names as the result,
/// leaving the other in `scratch`.
pub fn h_to_coset_with<L: Launcher>(
    l: &L,
    evals: Vec<Fr>,
    scratch: &mut Vec<Fr>,
    t: &Tables,
) -> Vec<Fr> {
    use crate::schedule::{Buf, Step, Twiddles};
    let n = evals.len();
    let mut a = evals;
    let mut b = std::mem::take(scratch);
    assert_eq!(b.len(), n, "scratch must match the domain");
    let sched = crate::schedule::coset(n as u32);
    for step in &sched.steps {
        let (src, dst) = match step.src() {
            Buf::A => (&a, &mut b),
            Buf::B => (&b, &mut a),
        };
        match *step {
            Step::BitReverse { .. } => l.map(dst, |i| kernels::bit_reverse(i, src, t.lg_n)),
            Step::NttStage {
                len,
                stride,
                twiddles,
                ..
            } => {
                let tw = match twiddles {
                    Twiddles::Inverse => &t.inverse,
                    Twiddles::Forward => &t.forward,
                };
                l.map(dst, |i| {
                    kernels::ntt_stage(i, src, len as usize, tw, stride as usize)
                })
            }
            Step::Scale { .. } => l.map(dst, |i| {
                kernels::pointwise_scale(i, src, &t.shift_powers).mul(&t.n_inv)
            }),
        }
    }
    let (result, leftover) = match sched.result {
        Buf::A => (a, b),
        Buf::B => (b, a),
    };
    *scratch = leftover;
    result
}

/// The host's half of an all-windows MSM: counting-sort each window's digits
/// (`digits[w·n..(w+1)·n]`) and lay the results out flat — `order` is every
/// window's sorted point indices concatenated, `starts` has one entry per
/// `(window, bucket)` plus a final end, in absolute `order` offsets — so one
/// `bucket_sum` launch over `windows · buckets` outputs sums every window.
pub fn sort_all_windows(digits: &[u32], n: usize, buckets: usize) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(digits.len() % n, 0, "digits must hold whole windows");
    let windows = digits.len() / n;
    let mut order = Vec::with_capacity(digits.len());
    let mut starts = Vec::with_capacity(windows * buckets + 1);
    for w in 0..windows {
        let (o, s) = counting_sort_by_digit(&digits[w * n..(w + 1) * n], buckets);
        let base = order.len() as u32;
        starts.extend(s[..buckets].iter().map(|x| base + x));
        order.extend_from_slice(&o);
    }
    starts.push(order.len() as u32);
    (order, starts)
}

/// The longest chain one thread adds in an MSM's bucket sums, and in every
/// level of the reduction above them. MEASURED on an RTX 5080 (C21,
/// `groth16-cuda-msm-bench`, 2^20 uniform scalars, 22 windows, the same
/// kernel): one thread per bucket ran at 4.5 M additions/s, because the top
/// window — two bits of a 254-bit scalar — puts a quarter of all points into
/// each of three buckets and the launch waits for those three threads;
/// pieces of at most 64 ran at 1.8 G/s (32: 1.7 G/s; 256: 1.0 G/s; 1024:
/// 0.6 G/s). A skewed witness (small values) fattens low windows the same
/// way; the plan bounds every chain regardless.
pub const CHUNK: u32 = 64;

/// Split every range of `starts` (contiguous ranges over one sequence, as
/// `sort_all_windows` lays them out) into pieces of at most `chunk` entries;
/// an empty range stays one empty piece. Returns the pieces, as `starts`
/// over the same sequence, and the groups: for each original range the
/// range of piece indices that belong to it (`groups.len() == starts.len()`).
pub fn chunk_ranges(starts: &[u32], chunk: u32) -> (Vec<u32>, Vec<u32>) {
    assert!(chunk > 0, "a piece holds at least one entry");
    let mut pieces = Vec::with_capacity(starts.len());
    let mut groups = Vec::with_capacity(starts.len());
    for w in starts.windows(2) {
        let (from, to) = (w[0], w[1]);
        assert!(from <= to, "ranges are non-decreasing");
        groups.push(pieces.len() as u32);
        let mut at = from;
        loop {
            pieces.push(at);
            if to - at <= chunk {
                break;
            }
            at += chunk;
        }
    }
    groups.push(pieces.len() as u32);
    pieces.push(*starts.last().expect("starts has a terminator"));
    (pieces, groups)
}

/// The launch plan that sums every range of `starts` with no thread adding
/// more than `chunk` entries: `levels[0]` are ranges over the original
/// sequence (for `bucket_sum`); `levels[k]`, `k ≥ 1`, ranges over the
/// outputs of level `k − 1` (for `jacobian_sum`); the last level has one
/// range per original range, in order. One level when no range exceeds
/// `chunk`; a range of `m` entries needs `⌈log_chunk m⌉` levels.
pub fn plan_ranges(starts: &[u32], chunk: u32) -> Vec<Vec<u32>> {
    let mut levels = Vec::new();
    let (mut pieces, mut groups) = chunk_ranges(starts, chunk);
    loop {
        levels.push(pieces);
        if groups.windows(2).all(|w| w[1] - w[0] == 1) {
            return levels;
        }
        (pieces, groups) = chunk_ranges(&groups, chunk);
    }
}

/// Window sums from the flat bucket sums (`windows · buckets` entries).
pub fn reduce_all_windows<F: Field>(sums: &[Jacobian<F>], buckets: usize) -> Vec<Jacobian<F>> {
    sums.chunks_exact(buckets).map(reduce_buckets).collect()
}

/// A multi-scalar multiplication over every window at once: one digits
/// launch, one host sort, one bucket-sum launch over pieces of at most
/// [`CHUNK`] points, the levels of `jacobian_sum` the plan needs (two or
/// three for the production circuit), then the reduction and Horner on the
/// host.
pub fn msm<L: Launcher, F: Field + Send + Sync>(
    l: &L,
    points: &[Affine<F>],
    scalars: &[Fr],
) -> Jacobian<F> {
    assert_eq!(points.len(), scalars.len());
    let canonical: Vec<[u64; 4]> = scalars.iter().map(Fr::to_canonical).collect();
    let n = points.len();
    let w = WINDOW_BITS;
    let buckets = (1usize << w) - 1;
    let windows = 256u32.div_ceil(w) as usize;
    let mut digits = vec![0u32; windows * n];
    l.map(&mut digits, |i| kernels::digit_all(i, &canonical, w));
    let (order, starts) = sort_all_windows(&digits, n, buckets);
    let plan = plan_ranges(&starts, CHUNK);
    let mut sums = vec![Jacobian::<F>::INFINITY; plan[0].len() - 1];
    l.map(&mut sums, |b| {
        kernels::bucket_sum(b, points, &order, &plan[0])
    });
    for level in &plan[1..] {
        let mut next = vec![Jacobian::<F>::INFINITY; level.len() - 1];
        l.map(&mut next, |b| kernels::jacobian_sum(b, &sums, level));
        sums = next;
    }
    debug_assert_eq!(sums.len(), windows * buckets);
    horner(&reduce_all_windows(&sums, buckets), w)
}

/// The arm's prover: the canonical kernel sequence on a launcher, the
/// shared assembly on the host.
pub fn prove<L: Launcher>(
    l: &L,
    zkey: &Zkey,
    witness: &[Fr],
    r: &Fr,
    s: &Fr,
) -> Result<Proof, ProveError> {
    let groups = CoefficientGroups::from_zkey(zkey);
    prove_grouped(l, zkey, &groups, witness, r, s)
}

/// [`prove`] with the coefficient groups supplied by the caller (who may have
/// dropped the zkey's flat coefficient list to halve that memory).
pub fn prove_grouped<L: Launcher>(
    l: &L,
    zkey: &Zkey,
    groups: &CoefficientGroups,
    witness: &[Fr],
    r: &Fr,
    s: &Fr,
) -> Result<Proof, ProveError> {
    if witness.len() != zkey.num_vars {
        return Err(ProveError::WitnessLength {
            expected: zkey.num_vars,
            found: witness.len(),
        });
    }
    if witness[0] != Fr::ONE {
        return Err(ProveError::WitnessConstant);
    }
    let n = zkey.domain_size;
    let tables = Tables::new(n);

    let a_h = scatter(
        l,
        &groups.a,
        &groups.a_starts,
        &groups.a_constraint,
        witness,
        n,
    );
    let b_h = scatter(
        l,
        &groups.b,
        &groups.b_starts,
        &groups.b_constraint,
        witness,
        n,
    );
    let mut c_h = vec![Fr::ZERO; n];
    l.map(&mut c_h, |i| kernels::pointwise_mul(i, &a_h, &b_h));

    let mut scratch = vec![Fr::ZERO; n];
    let a_c = h_to_coset_with(l, a_h, &mut scratch, &tables);
    let b_c = h_to_coset_with(l, b_h, &mut scratch, &tables);
    let c_c = h_to_coset_with(l, c_h, &mut scratch, &tables);
    let mut quotient = scratch;
    l.map(&mut quotient, |i| {
        kernels::pointwise_mul_sub(i, &a_c, &b_c, &c_c)
    });
    drop((a_c, b_c, c_c));

    let msms = Msms {
        h: msm(l, &zkey.h, &quotient),
        a: msm(l, &zkey.a, witness),
        b1: msm(l, &zkey.b1, witness),
        b2: msm(l, &zkey.b2, witness),
        c: msm(l, &zkey.c, &witness[zkey.num_public + 1..]),
    };
    Ok(assemble(zkey, &msms, r, s))
}

#[cfg(test)]
mod tests {
    use ark_ec::{AffineRepr as _, CurveGroup as _, PrimeGroup as _};
    use ark_ff::{BigInteger as _, PrimeField as _};
    use ark_groth16::Groth16;
    use risc0_groth16_core::{
        ec::{g1_generator, G1Affine, G2Affine},
        field::Fp,
        fp2::Fp2,
        ntt::{self, lg2},
        zkey::parse_wtns,
    };

    use super::*;
    use crate::launch::{CpuLauncher, SerialLauncher};

    const ZKEY: &[u8] =
        include_bytes!("../../../groth16_proof/circom-compat/test/data/multiplier2_final.zkey");
    const WTNS: &[u8] =
        include_bytes!("../../../groth16_proof/circom-compat/test/data/multiplier2.wtns");

    fn fp_to_ark(v: &Fp) -> ark_bn254::Fq {
        ark_bn254::Fq::from_le_bytes_mod_order(&v.to_le_bytes())
    }
    fn fp_from_ark(v: ark_bn254::Fq) -> Fp {
        Fp::from_le_bytes(&v.into_bigint().to_bytes_le().try_into().unwrap()).unwrap()
    }
    fn fr_to_ark(v: &Fr) -> ark_bn254::Fr {
        ark_bn254::Fr::from_le_bytes_mod_order(&v.to_le_bytes())
    }
    fn fr_from_ark(v: ark_bn254::Fr) -> Fr {
        Fr::from_le_bytes(&v.into_bigint().to_bytes_le().try_into().unwrap()).unwrap()
    }
    fn g1_to_ark(p: &G1Affine) -> ark_bn254::G1Affine {
        if p.infinity {
            ark_bn254::G1Affine::identity()
        } else {
            ark_bn254::G1Affine::new_unchecked(fp_to_ark(&p.x), fp_to_ark(&p.y))
        }
    }
    fn g1_from_ark(p: ark_bn254::G1Affine) -> G1Affine {
        match p.xy() {
            None => G1Affine::INFINITY,
            Some((x, y)) => G1Affine::new(fp_from_ark(x), fp_from_ark(y)),
        }
    }
    fn g2_to_ark(p: &G2Affine) -> ark_bn254::G2Affine {
        if p.infinity {
            ark_bn254::G2Affine::identity()
        } else {
            let c = |v: &Fp2| ark_bn254::Fq2::new(fp_to_ark(&v.c0), fp_to_ark(&v.c1));
            ark_bn254::G2Affine::new_unchecked(c(&p.x), c(&p.y))
        }
    }
    struct Scalars(u64);
    impl Scalars {
        fn next_fr(&mut self) -> ark_bn254::Fr {
            let mut b = [0u8; 32];
            for chunk in b.chunks_mut(8) {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                chunk.copy_from_slice(&x.to_le_bytes());
            }
            ark_bn254::Fr::from_le_bytes_mod_order(&b)
        }
    }

    fn verify(z: &Zkey, p: &Proof, public: &[Fr]) -> bool {
        let vk = ark_groth16::VerifyingKey::<ark_bn254::Bn254> {
            alpha_g1: g1_to_ark(&z.vk.alpha_g1),
            beta_g2: g2_to_ark(&z.vk.beta_g2),
            gamma_g2: g2_to_ark(&z.vk.gamma_g2),
            delta_g2: g2_to_ark(&z.vk.delta_g2),
            gamma_abc_g1: z.ic.iter().map(g1_to_ark).collect(),
        };
        let pvk = ark_groth16::prepare_verifying_key(&vk);
        let proof = ark_groth16::Proof {
            a: g1_to_ark(&p.a),
            b: g2_to_ark(&p.b),
            c: g1_to_ark(&p.c),
        };
        let inputs: Vec<ark_bn254::Fr> = public.iter().map(fr_to_ark).collect();
        Groth16::<ark_bn254::Bn254>::verify_proof(&pvk, &proof, &inputs).unwrap()
    }

    #[test]
    fn kernel_ntt_matches_the_core_transform() {
        let n = 64;
        let mut s = Scalars(0x3131);
        let data: Vec<Fr> = (0..n).map(|_| fr_from_ark(s.next_fr())).collect();
        let t = Tables::new(n);
        let mine = h_to_coset(&SerialLauncher, data.clone(), &t);
        let mut expected = data.clone();
        ntt::h_to_coset(&mut expected, &Fr::two_adic_root(lg2(n) + 1));
        assert_eq!(mine, expected);
        let par = h_to_coset(&CpuLauncher { threads: 5 }, data, &t);
        assert_eq!(par, expected, "chunked launcher agrees with serial");
    }

    #[test]
    fn a_plan_bounds_every_chain_and_ends_one_range_per_bucket() {
        // buckets of 0, 1, 64, 65, 1000 and 70_000 entries, chunk 64
        let sizes = [0u32, 1, 64, 65, 1000, 70_000];
        let mut starts = vec![0u32];
        for s in sizes {
            starts.push(starts.last().unwrap() + s);
        }
        let plan = plan_ranges(&starts, 64);
        assert_eq!(
            plan.len(),
            3,
            "70_000 → 1094 → 18 → 1: the pieces, then two levels above them"
        );
        for level in &plan {
            assert!(
                level.windows(2).all(|w| w[1] - w[0] <= 64),
                "no chain above 64"
            );
        }
        assert_eq!(
            plan.last().unwrap().len(),
            sizes.len() + 1,
            "one range per bucket at the end"
        );
        // the sums agree with the direct ones
        let seq: Vec<Jacobian<Fp>> = (0..*starts.last().unwrap())
            .map(|i| {
                g1_generator()
                    .to_jacobian()
                    .mul(&Fr::from_u64(u64::from(i) % 7 + 1))
            })
            .collect();
        let direct: Vec<Jacobian<Fp>> = (0..sizes.len())
            .map(|b| kernels::jacobian_sum(b, &seq, &starts))
            .collect();
        let mut sums: Vec<Jacobian<Fp>> = (0..plan[0].len() - 1)
            .map(|b| kernels::jacobian_sum(b, &seq, &plan[0]))
            .collect();
        for level in &plan[1..] {
            sums = (0..level.len() - 1)
                .map(|b| kernels::jacobian_sum(b, &sums, level))
                .collect();
        }
        assert_eq!(sums, direct);
        // a plan with nothing to split is one level, the ranges themselves
        assert_eq!(plan_ranges(&[0, 3, 3, 10], 64), vec![vec![0, 3, 3, 10]]);
    }

    #[test]
    fn kernel_msm_matches_the_core_msm() {
        let mut s = Scalars(0x4242);
        let g = ark_bn254::G1Projective::generator();
        let pts: Vec<G1Affine> = (0..100)
            .map(|_| g1_from_ark((g * s.next_fr()).into_affine()))
            .collect();
        let ks: Vec<Fr> = (0..100).map(|_| fr_from_ark(s.next_fr())).collect();
        let expected = risc0_groth16_core::msm::msm_naive(&pts, &ks);
        assert_eq!(msm(&CpuLauncher { threads: 3 }, &pts, &ks), expected);
        assert_eq!(msm(&SerialLauncher, &pts, &ks), expected);
    }

    #[test]
    fn fixture_proof_through_the_kernels_verifies_and_matches_core_for_fixed_blinding() {
        let z = Zkey::parse(ZKEY).unwrap();
        let w = parse_wtns(WTNS).unwrap();
        let (r, s) = (Fr::from_u64(5), Fr::from_u64(9));
        let proof = prove(&CpuLauncher::default(), &z, &w, &r, &s).unwrap();
        assert!(verify(&z, &proof, &w[1..=z.num_public]));
        let core = risc0_groth16_core::prover::prove(&z, &w, &r, &s).unwrap();
        assert_eq!(
            proof, core,
            "same blinding ⇒ the two pipelines produce the identical proof"
        );
        let mut bad = w.clone();
        bad[1] = Fr::from_u64(34);
        assert!(!verify(
            &z,
            &prove(&SerialLauncher, &z, &bad, &r, &s).unwrap(),
            &bad[1..=z.num_public]
        ));
    }

    #[test]
    fn coefficient_groups_cover_every_coefficient_once() {
        let z = Zkey::parse(ZKEY).unwrap();
        let g = CoefficientGroups::from_zkey(&z);
        assert_eq!(g.a.len() + g.b.len(), z.coefficients.len());
        assert_eq!(g.a_starts.len(), g.a_constraint.len() + 1);
        assert_eq!(g.b_starts.len(), g.b_constraint.len() + 1);
        assert_eq!(*g.a_starts.last().unwrap() as usize, g.a.len());
        assert!(
            g.a_constraint.windows(2).all(|p| p[0] < p[1]),
            "groups ascend by constraint"
        );
    }
}
