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

//! The snarkjs artifact formats a Groth16 prover consumes: the `zkey` (proving
//! key) and the `wtns` (witness). Layouts and encodings are those MEASURED in
//! `groth16_s01/BOUNDARY.md` §4.1: ten sections in ascending order; points as
//! little-endian Montgomery limbs; coefficient values as `v·R²`; witness
//! values canonical.
//!
//! Every malformation is a distinct error — a prover must reject a bad
//! artifact with a reason, never produce something from it.

use core::fmt;

use crate::{
    ec::{g1_b, g2_b, Affine, G1Affine, G2Affine},
    field::{Field, Fp, Fr},
    fp2::Fp2,
};

/// The verifying key embedded in the zkey header (section 2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyingKey {
    /// α in G1.
    pub alpha_g1: G1Affine,
    /// β in G1 (used by the prover for π_B's G1 twin).
    pub beta_g1: G1Affine,
    /// β in G2.
    pub beta_g2: G2Affine,
    /// γ in G2.
    pub gamma_g2: G2Affine,
    /// δ in G1.
    pub delta_g1: G1Affine,
    /// δ in G2.
    pub delta_g2: G2Affine,
}

/// One entry of the zkey's coefficient section: `matrix ∈ {0 = A, 1 = B}`,
/// the constraint (row) index, the signal (witness) index, and the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coefficient {
    /// 0 for the A matrix, 1 for the B matrix.
    pub matrix: u32,
    /// Constraint index (a point of the evaluation domain).
    pub constraint: u32,
    /// Witness index.
    pub signal: u32,
    /// The coefficient as a field element (the file's `v·R²` decoded).
    pub value: Fr,
}

/// A parsed Groth16 zkey.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Zkey {
    /// Number of witness values (`w[0] = 1` included).
    pub num_vars: usize,
    /// Number of public inputs (`w[1..=num_public]`).
    pub num_public: usize,
    /// Size of the evaluation domain (a power of two).
    pub domain_size: usize,
    /// The verifying key.
    pub vk: VerifyingKey,
    /// The `IC` points (`num_public + 1` of them).
    pub ic: Vec<G1Affine>,
    /// The A/B coefficients, in file order (ascending constraint).
    pub coefficients: Vec<Coefficient>,
    /// Section 5: `num_vars` A-query points.
    pub a: Vec<G1Affine>,
    /// Section 6: `num_vars` B-query points in G1.
    pub b1: Vec<G1Affine>,
    /// Section 7: `num_vars` B-query points in G2.
    pub b2: Vec<G2Affine>,
    /// Section 8: `num_vars - num_public - 1` C-query points.
    pub c: Vec<G1Affine>,
    /// Section 9: `domain_size` H-query points.
    pub h: Vec<G1Affine>,
}

/// Why a zkey could not be parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZkeyError {
    /// The file does not start with `zkey`.
    Magic,
    /// The file ends before a declared structure.
    Truncated {
        /// Byte offset at which more data was needed.
        at: usize,
    },
    /// The file does not declare exactly ten sections.
    SectionCount(u32),
    /// A section id is out of order.
    SectionOrder {
        /// The id expected next.
        expected: u32,
        /// The id found.
        found: u32,
    },
    /// The header names a protocol other than Groth16 (1).
    Protocol(u32),
    /// The header's field moduli or sizes do not match BN254 / 32 bytes.
    Curve,
    /// A section's byte length is not a multiple of its element size.
    SectionSize {
        /// Section id.
        section: u32,
    },
    /// A point is not on its curve.
    PointOffCurve {
        /// Section id.
        section: u32,
        /// Index within the section.
        index: usize,
    },
    /// The point counts disagree with the header.
    Inconsistent(&'static str),
    /// A coefficient value is not below the scalar modulus.
    CoefficientValue(usize),
}

impl fmt::Display for ZkeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZkeyError::Magic => write!(f, "not a zkey file (bad magic)"),
            ZkeyError::Truncated { at } => write!(f, "zkey truncated at byte {at}"),
            ZkeyError::SectionCount(n) => write!(f, "zkey has {n} sections, expected 10"),
            ZkeyError::SectionOrder { expected, found } => {
                write!(
                    f,
                    "zkey section {found} found where {expected} was expected"
                )
            }
            ZkeyError::Protocol(p) => write!(f, "zkey protocol {p} is not groth16 (1)"),
            ZkeyError::Curve => write!(f, "zkey is not over BN254 with 32-byte elements"),
            ZkeyError::SectionSize { section } => {
                write!(
                    f,
                    "zkey section {section} has a size that is not a whole number of elements"
                )
            }
            ZkeyError::PointOffCurve { section, index } => {
                write!(
                    f,
                    "zkey section {section} point {index} is not on the curve"
                )
            }
            ZkeyError::Inconsistent(what) => write!(f, "zkey is inconsistent: {what}"),
            ZkeyError::CoefficientValue(i) => {
                write!(f, "zkey coefficient {i} is not a field element")
            }
        }
    }
}

impl std::error::Error for ZkeyError {}

/// BN254's base-field modulus q as 32 little-endian bytes.
fn q_bytes() -> [u8; 32] {
    limbs_to_bytes(&Fp::MODULUS)
}

/// BN254's scalar-field modulus r as 32 little-endian bytes.
fn r_bytes() -> [u8; 32] {
    limbs_to_bytes(&Fr::MODULUS)
}

fn limbs_to_bytes(limbs: &[u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, l) in limbs.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&l.to_le_bytes());
    }
    out
}

fn bytes_to_limbs(b: &[u8]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let mut w = [0u8; 8];
        w.copy_from_slice(&b[i * 8..i * 8 + 8]);
        *limb = u64::from_le_bytes(w);
    }
    limbs
}

/// The element `R⁻¹`, whose Montgomery limbs are `1`; multiplying by it turns
/// a raw `v·R²` coefficient (read as Montgomery limbs, i.e. the element `v·R`)
/// into the element `v`.
fn r_inverse() -> Fr {
    Fr::from_montgomery_limbs([1, 0, 0, 0])
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ZkeyError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(ZkeyError::Truncated { at: self.pos })?;
        if end > self.bytes.len() {
            return Err(ZkeyError::Truncated { at: self.pos });
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, ZkeyError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, ZkeyError> {
        let b = self.take(8)?;
        let mut w = [0u8; 8];
        w.copy_from_slice(b);
        Ok(u64::from_le_bytes(w))
    }
}

/// Decode a G1 point stored as two Montgomery-form coordinates (64 bytes).
pub fn g1_from_montgomery_bytes(b: &[u8]) -> G1Affine {
    let x = Fp::from_montgomery_limbs(bytes_to_limbs(&b[0..32]));
    let y = Fp::from_montgomery_limbs(bytes_to_limbs(&b[32..64]));
    if x.is_zero() && y.is_zero() {
        G1Affine::INFINITY
    } else {
        G1Affine::new(x, y)
    }
}

/// Decode a G2 point stored as four Montgomery-form coordinates (128 bytes):
/// `x.c0, x.c1, y.c0, y.c1`.
pub fn g2_from_montgomery_bytes(b: &[u8]) -> G2Affine {
    let x = Fp2::new(
        Fp::from_montgomery_limbs(bytes_to_limbs(&b[0..32])),
        Fp::from_montgomery_limbs(bytes_to_limbs(&b[32..64])),
    );
    let y = Fp2::new(
        Fp::from_montgomery_limbs(bytes_to_limbs(&b[64..96])),
        Fp::from_montgomery_limbs(bytes_to_limbs(&b[96..128])),
    );
    if x.is_zero() && y.is_zero() {
        G2Affine::INFINITY
    } else {
        G2Affine::new(x, y)
    }
}

fn read_points<F: Field>(
    section: u32,
    bytes: &[u8],
    elem: usize,
    b: &F,
    decode: impl Fn(&[u8]) -> Affine<F>,
) -> Result<Vec<Affine<F>>, ZkeyError> {
    if bytes.len() % elem != 0 {
        return Err(ZkeyError::SectionSize { section });
    }
    let mut out = Vec::with_capacity(bytes.len() / elem);
    for (index, chunk) in bytes.chunks_exact(elem).enumerate() {
        let p = decode(chunk);
        if !p.is_on_curve(b) {
            return Err(ZkeyError::PointOffCurve { section, index });
        }
        out.push(p);
    }
    Ok(out)
}

impl Zkey {
    /// Parse a zkey from its bytes, validating structure, curve membership of
    /// every point, and the consistency of the section sizes with the header.
    pub fn parse(bytes: &[u8]) -> Result<Self, ZkeyError> {
        let mut r = Reader { bytes, pos: 0 };
        if r.take(4)? != b"zkey" {
            return Err(ZkeyError::Magic);
        }
        let _version = r.u32()?;
        let num_sections = r.u32()?;
        if num_sections != 10 {
            return Err(ZkeyError::SectionCount(num_sections));
        }
        let mut sections: Vec<&[u8]> = Vec::with_capacity(10);
        for expected in 1..=10u32 {
            let id = r.u32()?;
            if id != expected {
                return Err(ZkeyError::SectionOrder {
                    expected,
                    found: id,
                });
            }
            let size = r.u64()? as usize;
            sections.push(r.take(size)?);
        }

        // 1: header — protocol id.
        let mut s1 = Reader {
            bytes: sections[0],
            pos: 0,
        };
        let protocol = s1.u32()?;
        if protocol != 1 {
            return Err(ZkeyError::Protocol(protocol));
        }

        // 2: groth16 header — q, r, sizes, verifying key.
        let mut s2 = Reader {
            bytes: sections[1],
            pos: 0,
        };
        let n8q = s2.u32()? as usize;
        let q = s2.take(n8q)?;
        let n8r = s2.u32()? as usize;
        let rr = s2.take(n8r)?;
        if n8q != 32 || n8r != 32 || q != q_bytes() || rr != r_bytes() {
            return Err(ZkeyError::Curve);
        }
        let num_vars = s2.u32()? as usize;
        let num_public = s2.u32()? as usize;
        let domain_size = s2.u32()? as usize;
        let g1 = |r: &mut Reader| -> Result<G1Affine, ZkeyError> {
            let p = g1_from_montgomery_bytes(r.take(64)?);
            if !p.is_on_curve(&g1_b()) {
                return Err(ZkeyError::PointOffCurve {
                    section: 2,
                    index: 0,
                });
            }
            Ok(p)
        };
        let g2 = |r: &mut Reader| -> Result<G2Affine, ZkeyError> {
            let p = g2_from_montgomery_bytes(r.take(128)?);
            if !p.is_on_curve(&g2_b()) {
                return Err(ZkeyError::PointOffCurve {
                    section: 2,
                    index: 0,
                });
            }
            Ok(p)
        };
        let vk = VerifyingKey {
            alpha_g1: g1(&mut s2)?,
            beta_g1: g1(&mut s2)?,
            beta_g2: g2(&mut s2)?,
            gamma_g2: g2(&mut s2)?,
            delta_g1: g1(&mut s2)?,
            delta_g2: g2(&mut s2)?,
        };
        if !domain_size.is_power_of_two() || num_public + 1 > num_vars {
            return Err(ZkeyError::Inconsistent("header sizes"));
        }

        // 3: IC.
        let ic = read_points(3, sections[2], 64, &g1_b(), g1_from_montgomery_bytes)?;
        if ic.len() != num_public + 1 {
            return Err(ZkeyError::Inconsistent("IC count"));
        }

        // 4: coefficients — u32 count, then (m, c, s, value) records of 44 bytes.
        let mut s4 = Reader {
            bytes: sections[3],
            pos: 0,
        };
        let count = s4.u32()? as usize;
        if sections[3].len() != 4 + count * 44 {
            return Err(ZkeyError::SectionSize { section: 4 });
        }
        let r_inv = r_inverse();
        let mut coefficients = Vec::with_capacity(count);
        for i in 0..count {
            let matrix = s4.u32()?;
            let constraint = s4.u32()?;
            let signal = s4.u32()?;
            let raw = bytes_to_limbs(s4.take(32)?);
            // The file stores v·R². Read as Montgomery limbs that is the element v·R; one
            // multiplication by R⁻¹ (whose limbs are 1) yields v.
            if raw.iter().rev().cmp(Fr::MODULUS.iter().rev()) != core::cmp::Ordering::Less {
                return Err(ZkeyError::CoefficientValue(i));
            }
            let value = Fr::from_montgomery_limbs(raw).mul(&r_inv);
            if matrix > 1 || constraint as usize >= domain_size || signal as usize >= num_vars {
                return Err(ZkeyError::Inconsistent("coefficient indices"));
            }
            coefficients.push(Coefficient {
                matrix,
                constraint,
                signal,
                value,
            });
        }

        // 5..9: query points.
        let a = read_points(5, sections[4], 64, &g1_b(), g1_from_montgomery_bytes)?;
        let b1 = read_points(6, sections[5], 64, &g1_b(), g1_from_montgomery_bytes)?;
        let b2 = read_points(7, sections[6], 128, &g2_b(), g2_from_montgomery_bytes)?;
        let c = read_points(8, sections[7], 64, &g1_b(), g1_from_montgomery_bytes)?;
        let h = read_points(9, sections[8], 64, &g1_b(), g1_from_montgomery_bytes)?;
        if a.len() != num_vars || b1.len() != num_vars || b2.len() != num_vars {
            return Err(ZkeyError::Inconsistent("A/B query counts"));
        }
        if c.len() != num_vars - num_public - 1 {
            return Err(ZkeyError::Inconsistent("C query count"));
        }
        if h.len() != domain_size {
            return Err(ZkeyError::Inconsistent("H query count"));
        }

        Ok(Zkey {
            num_vars,
            num_public,
            domain_size,
            vk,
            ic,
            coefficients,
            a,
            b1,
            b2,
            c,
            h,
        })
    }
}

/// Why a witness file could not be parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WtnsError {
    /// The file does not start with `wtns`.
    Magic,
    /// The file ends before a declared structure.
    Truncated,
    /// The header is not for BN254's scalar field with 32-byte elements.
    Field,
    /// A value is not below the scalar modulus.
    Value(usize),
    /// The declared count and the data section disagree.
    Count,
}

impl fmt::Display for WtnsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WtnsError::Magic => write!(f, "not a wtns file (bad magic)"),
            WtnsError::Truncated => write!(f, "wtns truncated"),
            WtnsError::Field => write!(f, "wtns is not over BN254's scalar field"),
            WtnsError::Value(i) => write!(f, "wtns value {i} is not a field element"),
            WtnsError::Count => write!(f, "wtns count does not match its data section"),
        }
    }
}

impl std::error::Error for WtnsError {}

/// Parse a snarkjs `wtns` file into canonical witness values.
pub fn parse_wtns(bytes: &[u8]) -> Result<Vec<Fr>, WtnsError> {
    let mut r = Reader { bytes, pos: 0 };
    let t = |e: ZkeyError| match e {
        ZkeyError::Truncated { .. } => WtnsError::Truncated,
        _ => WtnsError::Truncated,
    };
    if r.take(4).map_err(t)? != b"wtns" {
        return Err(WtnsError::Magic);
    }
    let _version = r.u32().map_err(t)?;
    let num_sections = r.u32().map_err(t)?;
    if num_sections != 2 {
        return Err(WtnsError::Count);
    }
    let _id1 = r.u32().map_err(t)?;
    let size1 = r.u64().map_err(t)? as usize;
    let mut s1 = Reader {
        bytes: r.take(size1).map_err(t)?,
        pos: 0,
    };
    let n8 = s1.u32().map_err(t)? as usize;
    let modulus = s1.take(n8).map_err(t)?;
    if n8 != 32 || modulus != r_bytes() {
        return Err(WtnsError::Field);
    }
    let count = s1.u32().map_err(t)? as usize;
    let _id2 = r.u32().map_err(t)?;
    let size2 = r.u64().map_err(t)? as usize;
    let data = r.take(size2).map_err(t)?;
    if data.len() != count * 32 {
        return Err(WtnsError::Count);
    }
    parse_witness_values(data).map_err(WtnsError::Value)
}

/// Decode `n` consecutive canonical little-endian 32-byte witness values (the
/// exact bytes the boundary receives through `ProverParams::witness`).
pub fn parse_witness_values(data: &[u8]) -> Result<Vec<Fr>, usize> {
    let mut out = Vec::with_capacity(data.len() / 32);
    for (i, chunk) in data.chunks_exact(32).enumerate() {
        let mut b = [0u8; 32];
        b.copy_from_slice(chunk);
        out.push(Fr::from_le_bytes(&b).ok_or(i)?);
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod fixture {
    //! The in-tree circom fixture (`groth16_proof/circom-compat/test/data`).
    pub const ZKEY: &[u8] =
        include_bytes!("../../../groth16_proof/circom-compat/test/data/multiplier2_final.zkey");
    pub const WTNS: &[u8] =
        include_bytes!("../../../groth16_proof/circom-compat/test/data/multiplier2.wtns");
}

#[cfg(test)]
mod tests {
    use super::fixture::{WTNS, ZKEY};
    use super::*;

    #[test]
    fn fixture_parses_with_the_measured_layout() {
        let z = Zkey::parse(ZKEY).unwrap();
        assert_eq!((z.num_vars, z.num_public, z.domain_size), (4, 1, 4));
        assert_eq!(z.ic.len(), 2);
        assert_eq!(
            (z.a.len(), z.b1.len(), z.b2.len(), z.c.len(), z.h.len()),
            (4, 4, 4, 2, 4)
        );
        assert_eq!(z.coefficients.len(), 4);
        // (m, c, s, value) as measured on the fixture: value ∈ {-1, 1}.
        let one = Fr::ONE;
        let expected = [
            (0, 0, 2, one.neg()),
            (1, 0, 3, one),
            (0, 1, 0, one),
            (0, 2, 1, one),
        ];
        for (co, (m, c, s, v)) in z.coefficients.iter().zip(expected) {
            assert_eq!(
                (co.matrix, co.constraint, co.signal, co.value),
                (m, c, s, v)
            );
        }
        assert!(z.vk.alpha_g1.is_on_curve(&g1_b()));
        assert!(z.vk.beta_g2.is_on_curve(&g2_b()));
        assert!(!z.vk.alpha_g1.infinity);
    }

    #[test]
    fn fixture_witness_is_canonical_and_matches_the_circuit() {
        let w = parse_wtns(WTNS).unwrap();
        let v: Vec<u64> = w.iter().map(|x| x.to_canonical()[0]).collect();
        assert_eq!(v, [1, 33, 3, 11], "1, c = a·b, a, b");
        assert_eq!(w[1], w[2].mul(&w[3]));
    }

    #[test]
    fn malformations_are_distinct_errors() {
        assert_eq!(Zkey::parse(b"zke"), Err(ZkeyError::Truncated { at: 0 }));
        let mut bad = ZKEY.to_vec();
        bad[0] = b'x';
        assert_eq!(Zkey::parse(&bad), Err(ZkeyError::Magic));
        let mut bad = ZKEY.to_vec();
        bad[8] = 9; // section count
        assert_eq!(Zkey::parse(&bad), Err(ZkeyError::SectionCount(9)));
        let mut bad = ZKEY.to_vec();
        bad[12] = 2; // first section id
        assert_eq!(
            Zkey::parse(&bad),
            Err(ZkeyError::SectionOrder {
                expected: 1,
                found: 2
            })
        );
        let mut bad = ZKEY.to_vec();
        bad[24] = 2; // protocol id (section 1 payload starts at 12 + 12)
        assert_eq!(Zkey::parse(&bad), Err(ZkeyError::Protocol(2)));
        let truncated = &ZKEY[..ZKEY.len() - 100];
        assert!(matches!(
            Zkey::parse(truncated),
            Err(ZkeyError::Truncated { .. })
        ));
        // Flip a byte of the first A-query point: it leaves the curve.
        let z = Zkey::parse(ZKEY).unwrap();
        let _ = z;
        let mut bad = ZKEY.to_vec();
        let a_offset = find_section(ZKEY, 5);
        bad[a_offset + 3] ^= 0x01;
        assert_eq!(
            Zkey::parse(&bad),
            Err(ZkeyError::PointOffCurve {
                section: 5,
                index: 0
            })
        );
        let mut bad_w = WTNS.to_vec();
        bad_w[0] = b'x';
        assert_eq!(parse_wtns(&bad_w), Err(WtnsError::Magic));
        assert_eq!(parse_wtns(&WTNS[..40]), Err(WtnsError::Truncated));
    }

    /// Byte offset of a section's payload.
    fn find_section(bytes: &[u8], wanted: u32) -> usize {
        let mut pos = 12;
        loop {
            let id = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            let size = u64::from_le_bytes(bytes[pos + 4..pos + 12].try_into().unwrap()) as usize;
            pos += 12;
            if id == wanted {
                return pos;
            }
            pos += size;
        }
    }
}
