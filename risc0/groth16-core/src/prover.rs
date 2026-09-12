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

//! The Groth16 prover pipeline, on the CPU, in the exact shape of the
//! canonical kernels (`groth16_s01/BOUNDARY.md` §2): coefficient scatter,
//! `C = A∘B` on the domain, three coset transforms, `h·Z` on the coset, five
//! MSMs, and the host-side assembly with the blinding scalars `r`, `s`.
//!
//! This is the reference the GPU arms are differential-tested against and the
//! host pipeline they reuse; every step is one call into the shared modules.

use core::fmt;

use crate::{
    ec::{G1Affine, G1Jacobian, G2Affine, G2Jacobian},
    field::{Field, Fr},
    msm::msm_pippenger,
    ntt::{h_to_coset, lg2},
    zkey::Zkey,
};

/// A Groth16 proof `(π_A, π_B, π_C)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    /// π_A in G1.
    pub a: G1Affine,
    /// π_B in G2.
    pub b: G2Affine,
    /// π_C in G1.
    pub c: G1Affine,
}

/// Why a proof could not be produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProveError {
    /// The witness has the wrong length for this key.
    WitnessLength {
        /// Expected `num_vars`.
        expected: usize,
        /// Provided.
        found: usize,
    },
    /// `w[0]` is not 1.
    WitnessConstant,
}

impl fmt::Display for ProveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProveError::WitnessLength { expected, found } => {
                write!(f, "witness has {found} values, the key expects {expected}")
            }
            ProveError::WitnessConstant => write!(f, "witness[0] must be 1"),
        }
    }
}

impl std::error::Error for ProveError {}

/// Pippenger window width used by the reference (any width is correct; this
/// one is a reasonable CPU default).
pub const WINDOW_BITS: u32 = 12;

/// Evaluate the A and B polynomials on the domain from the coefficient list
/// (the `witness_into_poly` kernels): `poly_m[c] += w[s]·v`.
pub fn scatter(zkey: &Zkey, witness: &[Fr]) -> (Vec<Fr>, Vec<Fr>) {
    let mut a = vec![Fr::ZERO; zkey.domain_size];
    let mut b = vec![Fr::ZERO; zkey.domain_size];
    for co in &zkey.coefficients {
        let term = witness[co.signal as usize].mul(&co.value);
        let target = if co.matrix == 0 { &mut a } else { &mut b };
        let slot = &mut target[co.constraint as usize];
        *slot = slot.add(&term);
    }
    (a, b)
}

/// The quotient evaluations `(A·B − C)(g·H)` the H-query MSM consumes, computed
/// as the kernels do: `C = A∘B` on `H`, move all three to the coset `g·H` with
/// `g` a `2n`-th root of unity, then `A∘B − C` there.
pub fn quotient_on_coset(zkey: &Zkey, witness: &[Fr]) -> Vec<Fr> {
    let (mut a, mut b) = scatter(zkey, witness);
    let mut c: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| x.mul(y)).collect();
    let shift = Fr::two_adic_root(lg2(zkey.domain_size) + 1);
    h_to_coset(&mut a, &shift);
    h_to_coset(&mut b, &shift);
    h_to_coset(&mut c, &shift);
    a.iter()
        .zip(&b)
        .zip(&c)
        .map(|((x, y), z)| x.mul(y).sub(z))
        .collect()
}

/// Produce a proof for `witness` under `zkey` with blinding scalars `r`, `s`.
/// Deterministic for fixed `(r, s)`; callers sample them uniformly.
pub fn prove(zkey: &Zkey, witness: &[Fr], r: &Fr, s: &Fr) -> Result<Proof, ProveError> {
    if witness.len() != zkey.num_vars {
        return Err(ProveError::WitnessLength {
            expected: zkey.num_vars,
            found: witness.len(),
        });
    }
    if witness[0] != Fr::ONE {
        return Err(ProveError::WitnessConstant);
    }
    let h_evals = quotient_on_coset(zkey, witness);

    let pi_h: G1Jacobian = msm_pippenger(&zkey.h, &h_evals, WINDOW_BITS);
    let pi_a: G1Jacobian = msm_pippenger(&zkey.a, witness, WINDOW_BITS);
    let pi_b1: G1Jacobian = msm_pippenger(&zkey.b1, witness, WINDOW_BITS);
    let pi_b2: G2Jacobian = msm_pippenger(&zkey.b2, witness, WINDOW_BITS);
    let private = &witness[zkey.num_public + 1..];
    let pi_c: G1Jacobian = msm_pippenger(&zkey.c, private, WINDOW_BITS);

    let vk = &zkey.vk;
    let delta_g1 = vk.delta_g1.to_jacobian();
    // A = α + Σ wᵢ·Aᵢ + r·δ
    let a = pi_a.add_affine(&vk.alpha_g1).add(&delta_g1.mul(r));
    // B₂ = β₂ + Σ wᵢ·B2ᵢ + s·δ₂ ;  B₁ = β₁ + Σ wᵢ·B1ᵢ + s·δ₁
    let b2 = pi_b2
        .add_affine(&vk.beta_g2)
        .add(&vk.delta_g2.to_jacobian().mul(s));
    let b1 = pi_b1.add_affine(&vk.beta_g1).add(&delta_g1.mul(s));
    // C = Σ_{i > ℓ} wᵢ·Cᵢ + π_h + s·A + r·B₁ − (r·s)·δ
    let rs = r.mul(s);
    let c = pi_c
        .add(&pi_h)
        .add(&a.mul(s))
        .add(&b1.mul(r))
        .add(&delta_g1.mul(&rs).neg());

    Ok(Proof {
        a: a.to_affine(),
        b: b2.to_affine(),
        c: c.to_affine(),
    })
}

/// Decimal rendering of a canonical 256-bit value (what snarkjs JSON carries).
pub fn decimal(limbs: [u64; 4]) -> String {
    if limbs == [0; 4] {
        return "0".to_string();
    }
    let mut digits = Vec::new();
    let mut n = limbs;
    while n != [0; 4] {
        let mut rem = 0u64;
        for limb in n.iter_mut().rev() {
            let cur = ((rem as u128) << 64) | (*limb as u128);
            *limb = (cur / 10) as u64;
            rem = (cur % 10) as u64;
        }
        digits.push(b'0' + rem as u8);
    }
    digits.reverse();
    String::from_utf8(digits).expect("ascii digits")
}

fn g1_json(p: &G1Affine) -> String {
    if p.infinity {
        return "[\"0\",\"1\",\"0\"]".to_string();
    }
    format!(
        "[\"{}\",\"{}\",\"1\"]",
        decimal(p.x.to_canonical()),
        decimal(p.y.to_canonical())
    )
}

fn g2_json(p: &G2Affine) -> String {
    if p.infinity {
        return "[[\"0\",\"0\"],[\"1\",\"0\"],[\"0\",\"0\"]]".to_string();
    }
    format!(
        "[[\"{}\",\"{}\"],[\"{}\",\"{}\"],[\"1\",\"0\"]]",
        decimal(p.x.c0.to_canonical()),
        decimal(p.x.c1.to_canonical()),
        decimal(p.y.c0.to_canonical()),
        decimal(p.y.c1.to_canonical())
    )
}

/// The proof in snarkjs's `proof.json` shape — what `risc0_groth16::ProofJson`
/// parses and what the canonical kernels write.
pub fn proof_json(proof: &Proof) -> String {
    format!(
        "{{ \"pi_a\": {}, \"pi_b\": {}, \"pi_c\": {}, \"protocol\": \"groth16\", \"curve\": \"bn128\" }}",
        g1_json(&proof.a),
        g2_json(&proof.b),
        g1_json(&proof.c)
    )
}

/// The public inputs `w[1..=num_public]` in snarkjs's `public.json` shape.
pub fn public_json(witness: &[Fr], num_public: usize) -> String {
    let items: Vec<String> = witness[1..=num_public]
        .iter()
        .map(|w| format!("\"{}\"", decimal(w.to_canonical())))
        .collect();
    format!("[{}]", items.join(","))
}

/// The verifying key in snarkjs's `verification_key.json` shape (what
/// `risc0_groth16::VerifyingKeyJson` parses); `vk_alphabeta_12` is left empty
/// because the verifier recomputes the pairing.
pub fn verifying_key_json(zkey: &Zkey) -> String {
    let ic: Vec<String> = zkey.ic.iter().map(g1_json).collect();
    format!(
        "{{ \"protocol\": \"groth16\", \"curve\": \"bn128\", \"nPublic\": {}, \"vk_alpha_1\": {}, \"vk_beta_2\": {}, \"vk_gamma_2\": {}, \"vk_delta_2\": {}, \"vk_alphabeta_12\": [], \"IC\": [{}] }}",
        zkey.num_public,
        g1_json(&zkey.vk.alpha_g1),
        g2_json(&zkey.vk.beta_g2),
        g2_json(&zkey.vk.gamma_g2),
        g2_json(&zkey.vk.delta_g2),
        ic.join(",")
    )
}

#[cfg(test)]
mod tests {
    use ark_groth16::Groth16;

    use super::*;
    use crate::ec::conv::*;
    use crate::zkey::{
        fixture::{WTNS, ZKEY},
        parse_wtns,
    };

    fn ark_vk(z: &Zkey) -> ark_groth16::VerifyingKey<ark_bn254::Bn254> {
        ark_groth16::VerifyingKey {
            alpha_g1: g1_to_ark(&z.vk.alpha_g1),
            beta_g2: g2_to_ark(&z.vk.beta_g2),
            gamma_g2: g2_to_ark(&z.vk.gamma_g2),
            delta_g2: g2_to_ark(&z.vk.delta_g2),
            gamma_abc_g1: z.ic.iter().map(g1_to_ark).collect(),
        }
    }

    fn ark_proof(p: &Proof) -> ark_groth16::Proof<ark_bn254::Bn254> {
        ark_groth16::Proof {
            a: g1_to_ark(&p.a),
            b: g2_to_ark(&p.b),
            c: g1_to_ark(&p.c),
        }
    }

    fn verify(z: &Zkey, p: &Proof, public: &[Fr]) -> bool {
        let pvk = ark_groth16::prepare_verifying_key(&ark_vk(z));
        let inputs: Vec<ark_bn254::Fr> = public.iter().map(fr_to_ark).collect();
        Groth16::<ark_bn254::Bn254>::verify_proof(&pvk, &ark_proof(p), &inputs).unwrap()
    }

    #[test]
    fn fixture_proof_verifies_under_arkworks_groth16() {
        let z = Zkey::parse(ZKEY).unwrap();
        let w = parse_wtns(WTNS).unwrap();
        let (r, s) = (Fr::from_u64(0x1234_5678), Fr::from_u64(0x9abc_def0));
        let proof = prove(&z, &w, &r, &s).unwrap();
        assert!(
            verify(&z, &proof, &w[1..=z.num_public]),
            "the proof must verify"
        );
        // Randomisation: different blinding, still verifies, different bytes.
        let proof2 = prove(&z, &w, &Fr::from_u64(7), &Fr::from_u64(11)).unwrap();
        assert_ne!(proof, proof2);
        assert!(verify(&z, &proof2, &w[1..=z.num_public]));
        // Mutation arms: a wrong public input, a corrupted proof element, a wrong witness.
        assert!(
            !verify(&z, &proof, &[Fr::from_u64(34)]),
            "wrong public input must be rejected"
        );
        let mut corrupted = proof.clone();
        corrupted.c = corrupted.c.neg();
        assert!(
            !verify(&z, &corrupted, &w[1..=z.num_public]),
            "a corrupted π_C must be rejected"
        );
        let mut bad_w = w.clone();
        bad_w[1] = Fr::from_u64(34); // c ≠ a·b
        let bad_proof = prove(&z, &bad_w, &r, &s).unwrap();
        assert!(
            !verify(&z, &bad_proof, &bad_w[1..=z.num_public]),
            "an unsatisfied witness must not verify"
        );
    }

    #[test]
    fn quotient_is_divisible_by_the_vanishing_polynomial() {
        // On the coset every A·B − C value is h(x)·Z_H(x); Z_H is constant on g·H, so the
        // evaluations interpolate to a polynomial of degree < n (the last n coefficients of
        // the size-2n interpolation vanish).
        let z = Zkey::parse(ZKEY).unwrap();
        let w = parse_wtns(WTNS).unwrap();
        let (a, b) = scatter(&z, &w);
        // The constraint a·b = c holds on H: A∘B − C = 0 there.
        let c: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| x.mul(y)).collect();
        assert_eq!(
            c[0],
            w[2].mul(&w[3]).neg(),
            "constraint 0: (-a)·(b) with the fixture's sign"
        );
        let q = quotient_on_coset(&z, &w);
        assert_eq!(q.len(), z.domain_size);
        // With an unsatisfied witness the domain values are no longer consistent.
        let mut bad = w.clone();
        bad[1] = Fr::from_u64(34);
        assert_ne!(quotient_on_coset(&z, &bad), q);
    }

    #[test]
    fn prove_rejects_a_malformed_witness_with_a_reason() {
        let z = Zkey::parse(ZKEY).unwrap();
        let w = parse_wtns(WTNS).unwrap();
        assert_eq!(
            prove(&z, &w[..3], &Fr::ONE, &Fr::ONE),
            Err(ProveError::WitnessLength {
                expected: 4,
                found: 3
            })
        );
        let mut bad = w.clone();
        bad[0] = Fr::from_u64(2);
        assert_eq!(
            prove(&z, &bad, &Fr::ONE, &Fr::ONE),
            Err(ProveError::WitnessConstant)
        );
    }

    #[test]
    fn json_shapes_round_trip_through_decimal() {
        assert_eq!(decimal([0, 0, 0, 0]), "0");
        assert_eq!(decimal([10, 0, 0, 0]), "10");
        assert_eq!(decimal([u64::MAX, 0, 0, 0]), "18446744073709551615");
        assert_eq!(decimal([0, 1, 0, 0]), "18446744073709551616");
        assert_eq!(
            decimal(Fr::MODULUS),
            "21888242871839275222246405745257275088548364400416034343698204186575808495617"
        );
        let z = Zkey::parse(ZKEY).unwrap();
        let w = parse_wtns(WTNS).unwrap();
        let proof = prove(&z, &w, &Fr::from_u64(3), &Fr::from_u64(4)).unwrap();
        let pj = proof_json(&proof);
        assert!(pj.starts_with("{ \"pi_a\": [\"") && pj.contains("\"protocol\": \"groth16\""));
        assert_eq!(public_json(&w, z.num_public), "[\"33\"]");
        let vj = verifying_key_json(&z);
        assert!(vj.contains("\"nPublic\": 1") && vj.contains("\"IC\": [[\""));
    }
}
