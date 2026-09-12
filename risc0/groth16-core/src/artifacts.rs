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

//! The two precomputed artifacts the canonical kernels consume beside the
//! zkey — `preprocessed_coeffs.bin` and `fuzzed_msm_results.bin` — in the
//! layouts MEASURED on the rzup `risc0-groth16` v0.1.0 set (BOUNDARY.md
//! §4.1): both are derivable from the zkey, so an arm may consume them for
//! speed or rebuild them; this module does both and checks they agree.

use core::fmt;

use crate::{
    coeff::GroupedCoeff,
    ec::Jacobian,
    field::{Field, Fp, Fr},
    fp2::Fp2,
    prover::CoefficientGroups,
};

/// Record size of `preprocessed_coeffs.bin`: `u32 m, c, s`, 4 bytes of
/// padding, then the 32-byte value (`v·R²`, as in the zkey).
pub const PCOEFF_RECORD: usize = 48;

/// Why an artifact could not be parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    /// The file is shorter than its header or its declared contents.
    Truncated,
    /// The declared counts do not add up to the file size.
    Size {
        /// Expected size in bytes.
        expected: usize,
        /// Actual size.
        found: usize,
    },
    /// A record's matrix id is not 0 or 1, or its indices exceed the zkey's.
    Record(usize),
    /// A value is not a scalar-field element.
    Value(usize),
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArtifactError::Truncated => write!(f, "artifact truncated"),
            ArtifactError::Size { expected, found } => {
                write!(
                    f,
                    "artifact size {found} does not match its header ({expected})"
                )
            }
            ArtifactError::Record(i) => write!(f, "artifact record {i} is malformed"),
            ArtifactError::Value(i) => write!(f, "artifact value {i} is not a field element"),
        }
    }
}

impl std::error::Error for ArtifactError {}

fn u64_at(b: &[u8], off: usize) -> Result<u64, ArtifactError> {
    let s = b.get(off..off + 8).ok_or(ArtifactError::Truncated)?;
    Ok(u64::from_le_bytes(s.try_into().expect("8 bytes")))
}

fn u32_at(b: &[u8], off: usize) -> Result<u32, ArtifactError> {
    let s = b.get(off..off + 4).ok_or(ArtifactError::Truncated)?;
    Ok(u32::from_le_bytes(s.try_into().expect("4 bytes")))
}

fn limbs_at(b: &[u8], off: usize) -> Result<[u64; 4], ArtifactError> {
    let mut l = [0u64; 4];
    for (i, v) in l.iter_mut().enumerate() {
        *v = u64_at(b, off + i * 8)?;
    }
    Ok(l)
}

/// The element `R⁻¹` (its Montgomery limbs are 1).
fn r_inverse() -> Fr {
    Fr::from_montgomery_limbs([1, 0, 0, 0])
}

/// Parse `preprocessed_coeffs.bin` into coefficient groups: header of four
/// `u64` (`count_a`, `count_b`, `unique_a`, `unique_b` — the index lists'
/// lengths, sentinel included), `count_a + count_b` 48-byte records (A list
/// then B list, each grouped by ascending constraint), then
/// `unique_a + unique_b` `u32` start indices.
pub fn parse_preprocessed_coeffs(
    bytes: &[u8],
    zkey_num_vars: usize,
    zkey_domain: usize,
) -> Result<CoefficientGroups, ArtifactError> {
    let count_a = u64_at(bytes, 0)? as usize;
    let count_b = u64_at(bytes, 8)? as usize;
    let unique_a = u64_at(bytes, 16)? as usize;
    let unique_b = u64_at(bytes, 24)? as usize;
    let expected = 32 + (count_a + count_b) * PCOEFF_RECORD + (unique_a + unique_b) * 4;
    if bytes.len() != expected {
        return Err(ArtifactError::Size {
            expected,
            found: bytes.len(),
        });
    }
    let r_inv = r_inverse();
    let read_list = |offset: usize,
                         count: usize,
                         matrix: u32|
     -> Result<(Vec<GroupedCoeff>, Vec<u32>), ArtifactError> {
        let mut coeffs = Vec::with_capacity(count);
        let mut constraints = Vec::new();
        for i in 0..count {
            let rec = offset + i * PCOEFF_RECORD;
            let m = u32_at(bytes, rec)?;
            let c = u32_at(bytes, rec + 4)?;
            let s = u32_at(bytes, rec + 8)?;
            if m != matrix || c as usize >= zkey_domain || s as usize >= zkey_num_vars {
                return Err(ArtifactError::Record(i));
            }
            let raw = limbs_at(bytes, rec + 16)?;
            if raw.iter().rev().cmp(Fr::MODULUS.iter().rev()) != core::cmp::Ordering::Less {
                return Err(ArtifactError::Value(i));
            }
            coeffs.push(GroupedCoeff {
                signal: s,
                value: Fr::from_montgomery_limbs(raw).mul(&r_inv),
            });
            constraints.push(c);
        }
        Ok((coeffs, constraints))
    };
    let (a, a_cons) = read_list(32, count_a, 0)?;
    let (b, b_cons) = read_list(32 + count_a * PCOEFF_RECORD, count_b, 1)?;
    let idx_base = 32 + (count_a + count_b) * PCOEFF_RECORD;
    let mut a_starts = Vec::with_capacity(unique_a);
    for i in 0..unique_a {
        a_starts.push(u32_at(bytes, idx_base + i * 4)?);
    }
    let mut b_starts = Vec::with_capacity(unique_b);
    for i in 0..unique_b {
        b_starts.push(u32_at(bytes, idx_base + (unique_a + i) * 4)?);
    }
    let group_constraints = |starts: &[u32], cons: &[u32]| -> Result<Vec<u32>, ArtifactError> {
        let mut out = Vec::with_capacity(starts.len().saturating_sub(1));
        for w in starts.windows(2) {
            let (from, to) = (w[0] as usize, w[1] as usize);
            if from >= to || to > cons.len() {
                return Err(ArtifactError::Record(from));
            }
            let c = cons[from];
            if cons[from..to].iter().any(|x| *x != c) {
                return Err(ArtifactError::Record(from));
            }
            out.push(c);
        }
        Ok(out)
    };
    let a_constraint = group_constraints(&a_starts, &a_cons)?;
    let b_constraint = group_constraints(&b_starts, &b_cons)?;
    Ok(CoefficientGroups {
        a,
        a_starts,
        a_constraint,
        b,
        b_starts,
        b_constraint,
    })
}

/// Write coefficient groups in the `preprocessed_coeffs.bin` layout (the
/// setup path's output), so an arm can produce the artifact for a new circuit
/// and the reader can be checked against the zkey.
pub fn write_preprocessed_coeffs(groups: &CoefficientGroups) -> Vec<u8> {
    let r2 = Fr::from_montgomery_limbs(crate::consts::FR_R2); // element R
    let mut out = Vec::with_capacity(
        32 + (groups.a.len() + groups.b.len()) * PCOEFF_RECORD
            + (groups.a_starts.len() + groups.b_starts.len()) * 4,
    );
    for v in [
        groups.a.len(),
        groups.b.len(),
        groups.a_starts.len(),
        groups.b_starts.len(),
    ] {
        out.extend_from_slice(&(v as u64).to_le_bytes());
    }
    let mut write_list = |coeffs: &[GroupedCoeff], starts: &[u32], cons: &[u32], matrix: u32| {
        for (g, w) in starts.windows(2).enumerate() {
            for c in &coeffs[w[0] as usize..w[1] as usize] {
                out.extend_from_slice(&matrix.to_le_bytes());
                out.extend_from_slice(&cons[g].to_le_bytes());
                out.extend_from_slice(&c.signal.to_le_bytes());
                out.extend_from_slice(&[0u8; 4]);
                // stored as v·R²: the element v·R in Montgomery limbs
                let raw = c.value.mul(&r2).montgomery_limbs();
                for l in raw {
                    out.extend_from_slice(&l.to_le_bytes());
                }
            }
        }
    };
    write_list(&groups.a, &groups.a_starts, &groups.a_constraint, 0);
    write_list(&groups.b, &groups.b_starts, &groups.b_constraint, 1);
    for s in groups.a_starts.iter().chain(&groups.b_starts) {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// `fuzzed_msm_results.bin`: five Jacobian points in sppark's layout
/// (`X, Y, Z` Montgomery limbs) — `a`, `b_g1`, `c`, `h` in G1 (96 bytes
/// each), then `b_g2` in G2 (192 bytes); 576 bytes. They hold the negated
/// MSMs of the deterministic "fuzz" scalars the canonical kernels add to the
/// witness; an arm that does not use the fuzz trick does not need them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuzzedMsmResults {
    /// −MSM(fuzz, A).
    pub a: Jacobian<Fp>,
    /// −MSM(fuzz, B1).
    pub b_g1: Jacobian<Fp>,
    /// −MSM(fuzz, C).
    pub c: Jacobian<Fp>,
    /// Unused by the canonical prover (never computed at setup).
    pub h: Jacobian<Fp>,
    /// −MSM(fuzz, B2).
    pub b_g2: Jacobian<Fp2>,
}

/// Size of `fuzzed_msm_results.bin`.
pub const FRES_SIZE: usize = 4 * 96 + 192;

/// Parse `fuzzed_msm_results.bin`.
pub fn parse_fuzzed_msm_results(bytes: &[u8]) -> Result<FuzzedMsmResults, ArtifactError> {
    if bytes.len() != FRES_SIZE {
        return Err(ArtifactError::Size {
            expected: FRES_SIZE,
            found: bytes.len(),
        });
    }
    let g1 = |off: usize| -> Result<Jacobian<Fp>, ArtifactError> {
        Ok(Jacobian {
            x: Fp::from_montgomery_limbs(limbs_at(bytes, off)?),
            y: Fp::from_montgomery_limbs(limbs_at(bytes, off + 32)?),
            z: Fp::from_montgomery_limbs(limbs_at(bytes, off + 64)?),
        })
    };
    let fp2 = |off: usize| -> Result<Fp2, ArtifactError> {
        Ok(Fp2::new(
            Fp::from_montgomery_limbs(limbs_at(bytes, off)?),
            Fp::from_montgomery_limbs(limbs_at(bytes, off + 32)?),
        ))
    };
    Ok(FuzzedMsmResults {
        a: g1(0)?,
        b_g1: g1(96)?,
        c: g1(192)?,
        h: g1(288)?,
        b_g2: Jacobian {
            x: fp2(384)?,
            y: fp2(448)?,
            z: fp2(512)?,
        },
    })
}

/// Whether a parsed Jacobian point is the identity.
pub fn is_identity<F: Field>(p: &Jacobian<F>) -> bool {
    p.z.is_zero()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkey::{fixture::ZKEY, Zkey};

    #[test]
    fn preprocessed_coeffs_round_trip_the_fixture_zkey() {
        let z = Zkey::parse(ZKEY).unwrap();
        let groups = CoefficientGroups::from_zkey(&z);
        let bytes = write_preprocessed_coeffs(&groups);
        assert_eq!(bytes.len() % 4, 0);
        let parsed = parse_preprocessed_coeffs(&bytes, z.num_vars, z.domain_size).unwrap();
        assert_eq!(parsed.a, groups.a);
        assert_eq!(parsed.b, groups.b);
        assert_eq!(parsed.a_starts, groups.a_starts);
        assert_eq!(parsed.b_starts, groups.b_starts);
        assert_eq!(parsed.a_constraint, groups.a_constraint);
        assert_eq!(parsed.b_constraint, groups.b_constraint);
        // The stored value is v·R²: the constant 1 reads back as R² raw, as in the zkey.
        let first_value_raw = limbs_at(&bytes, 32 + 16).unwrap();
        assert_eq!(
            first_value_raw,
            Fr::ONE
                .neg()
                .mul(&Fr::from_montgomery_limbs(crate::consts::FR_R2))
                .montgomery_limbs(),
            "fixture's first A coefficient is −1"
        );
        // Malformations are distinct errors.
        assert_eq!(
            parse_preprocessed_coeffs(&bytes[..20], 4, 4),
            Err(ArtifactError::Truncated)
        );
        assert!(matches!(
            parse_preprocessed_coeffs(&bytes[..bytes.len() - 4], 4, 4),
            Err(ArtifactError::Size { .. })
        ));
        let mut bad = bytes.clone();
        bad[32] = 7; // matrix id of the first record
        assert_eq!(
            parse_preprocessed_coeffs(&bad, 4, 4),
            Err(ArtifactError::Record(0))
        );
    }

    #[test]
    fn fuzzed_results_parse_and_reject_the_wrong_size() {
        let zero = [0u8; FRES_SIZE];
        let f = parse_fuzzed_msm_results(&zero).unwrap();
        assert!(is_identity(&f.a) && is_identity(&f.b_g2));
        assert!(matches!(
            parse_fuzzed_msm_results(&zero[..100]),
            Err(ArtifactError::Size { .. })
        ));
    }

    /// Against the production artifact set when it is present (`RISC0_GROTH16_ARTIFACTS`).
    #[test]
    #[ignore = "needs the rzup risc0-groth16 v0.1.0 artifact set on disk"]
    fn production_preprocessed_coeffs_agree_with_the_production_zkey() {
        let dir = std::env::var("RISC0_GROTH16_ARTIFACTS").expect("RISC0_GROTH16_ARTIFACTS");
        let zkey_bytes = std::fs::read(format!("{dir}/stark_verify_final.zkey")).unwrap();
        let z = Zkey::parse(&zkey_bytes).unwrap();
        let pc = std::fs::read(format!("{dir}/preprocessed_coeffs.bin")).unwrap();
        let parsed = parse_preprocessed_coeffs(&pc, z.num_vars, z.domain_size).unwrap();
        let groups = CoefficientGroups::from_zkey(&z);
        assert_eq!(parsed.a.len(), groups.a.len());
        assert_eq!(parsed.b.len(), groups.b.len());
        assert_eq!(parsed.a_constraint, groups.a_constraint);
        assert_eq!(parsed.b_constraint, groups.b_constraint);
        assert_eq!(
            parsed.a, groups.a,
            "A coefficients equal the zkey's, value for value"
        );
        assert_eq!(parsed.b, groups.b);
        let fres = std::fs::read(format!("{dir}/fuzzed_msm_results.bin")).unwrap();
        let f = parse_fuzzed_msm_results(&fres).unwrap();
        assert!(
            !is_identity(&f.a) && !is_identity(&f.b_g2),
            "production fuzz results are real points"
        );
    }
}
