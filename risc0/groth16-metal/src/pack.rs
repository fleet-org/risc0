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

//! Byte layouts the kernels read, produced from the shared Rust types:
//! field elements as 32 Montgomery bytes, G1 points as 64, G2 as 128 (the
//! zkey's own layouts), grouped coefficients as 48-byte records (signal at
//! 0, value at 16 — `preprocessed_coeffs.bin`'s record), and Jacobian
//! results back into `risc0-groth16-core` points.

use risc0_groth16_core::{
    coeff::GroupedCoeff,
    ec::{Affine, Jacobian},
    field::{Field, Fp, Fr},
    fp2::Fp2,
};

/// A field element's Montgomery limbs as 32 little-endian bytes.
pub fn fr_bytes(x: &Fr) -> [u8; 32] {
    limbs_bytes(&x.montgomery_limbs())
}

fn limbs_bytes(l: &[u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, v) in l.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
    out
}

fn limbs_from(b: &[u8]) -> [u64; 4] {
    let mut l = [0u64; 4];
    for (i, v) in l.iter_mut().enumerate() {
        let mut w = [0u8; 8];
        w.copy_from_slice(&b[i * 8..i * 8 + 8]);
        *v = u64::from_le_bytes(w);
    }
    l
}

/// Pack scalar-field elements (Montgomery form).
pub fn pack_fr(xs: &[Fr]) -> Vec<u8> {
    xs.iter().flat_map(fr_bytes).collect()
}

/// Pack canonical scalars (what the digit kernel reads).
pub fn pack_canonical(xs: &[Fr]) -> Vec<u8> {
    xs.iter()
        .flat_map(|x| limbs_bytes(&x.to_canonical()))
        .collect()
}

/// Pack G1 affine points as 64 bytes each; infinity as all zeros.
pub fn pack_g1(ps: &[Affine<Fp>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ps.len() * 64);
    for p in ps {
        if p.infinity {
            out.extend_from_slice(&[0u8; 64]);
        } else {
            out.extend_from_slice(&limbs_bytes(&p.x.montgomery_limbs()));
            out.extend_from_slice(&limbs_bytes(&p.y.montgomery_limbs()));
        }
    }
    out
}

/// Pack G2 affine points as 128 bytes each (`x.c0, x.c1, y.c0, y.c1`); infinity as zeros.
pub fn pack_g2(ps: &[Affine<Fp2>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ps.len() * 128);
    for p in ps {
        if p.infinity {
            out.extend_from_slice(&[0u8; 128]);
        } else {
            for c in [&p.x.c0, &p.x.c1, &p.y.c0, &p.y.c1] {
                out.extend_from_slice(&limbs_bytes(&c.montgomery_limbs()));
            }
        }
    }
    out
}

/// Pack grouped coefficients as 48-byte records: `signal` at 0, `value` at 16.
pub fn pack_coeffs(cs: &[GroupedCoeff]) -> Vec<u8> {
    let mut out = Vec::with_capacity(cs.len() * 48);
    for c in cs {
        out.extend_from_slice(&c.signal.to_le_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out.extend_from_slice(&fr_bytes(&c.value));
    }
    out
}

/// Read Jacobian G1 results (96 bytes each: X, Y, Z Montgomery).
pub fn unpack_jac_g1(bytes: &[u8]) -> Vec<Jacobian<Fp>> {
    bytes
        .chunks_exact(96)
        .map(|c| Jacobian {
            x: Fp::from_montgomery_limbs(limbs_from(&c[0..32])),
            y: Fp::from_montgomery_limbs(limbs_from(&c[32..64])),
            z: Fp::from_montgomery_limbs(limbs_from(&c[64..96])),
        })
        .collect()
}

/// Read Jacobian G2 results (192 bytes each: X, Y, Z in Fp2).
pub fn unpack_jac_g2(bytes: &[u8]) -> Vec<Jacobian<Fp2>> {
    let fp2 = |b: &[u8]| {
        Fp2::new(
            Fp::from_montgomery_limbs(limbs_from(&b[0..32])),
            Fp::from_montgomery_limbs(limbs_from(&b[32..64])),
        )
    };
    bytes
        .chunks_exact(192)
        .map(|c| Jacobian {
            x: fp2(&c[0..64]),
            y: fp2(&c[64..128]),
            z: fp2(&c[128..192]),
        })
        .collect()
}

/// Read scalar-field results (32 bytes each, Montgomery).
pub fn unpack_fr(bytes: &[u8]) -> Vec<Fr> {
    bytes
        .chunks_exact(32)
        .map(|c| Fr::from_montgomery_limbs(limbs_from(c)))
        .collect()
}

/// Whether a Jacobian result is the identity (Z = 0), matching the kernels' convention.
pub fn is_infinity<F: Field>(p: &Jacobian<F>) -> bool {
    p.z.is_zero()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_have_the_measured_layouts() {
        let one = Fr::ONE;
        assert_eq!(pack_fr(&[one]).len(), 32);
        assert_eq!(pack_fr(&[one])[..32], fr_bytes(&one));
        let c = GroupedCoeff {
            signal: 167,
            value: one,
        };
        let rec = pack_coeffs(&[c]);
        assert_eq!(rec.len(), 48);
        assert_eq!(u32::from_le_bytes(rec[0..4].try_into().unwrap()), 167);
        assert_eq!(rec[16..48], fr_bytes(&one));
        assert!(rec[4..16].iter().all(|b| *b == 0));
        let g = Affine::new(Fp::ONE, Fp::from_u64(2));
        let pg = pack_g1(&[g, Affine::INFINITY]);
        assert_eq!(pg.len(), 128);
        assert!(pg[64..].iter().all(|b| *b == 0), "infinity packs as zeros");
        assert_eq!(
            unpack_fr(&pack_fr(&[one, Fr::from_u64(7)])),
            vec![one, Fr::from_u64(7)]
        );
        let j = Jacobian {
            x: Fp::ONE,
            y: Fp::from_u64(2),
            z: Fp::from_u64(3),
        };
        let mut bytes = Vec::new();
        for v in [&j.x, &j.y, &j.z] {
            bytes.extend_from_slice(&limbs_bytes(&v.montgomery_limbs()));
        }
        assert_eq!(unpack_jac_g1(&bytes)[0], j);
    }

    #[test]
    fn shader_source_carries_both_constant_sets_and_every_kernel() {
        let src = crate::MSL_SOURCE;
        for needle in [
            "FP_MOD",
            "FR_MOD",
            "FP_INV",
            "FR_INV",
            "kernel void scatter_group",
            "kernel void pointwise_mul",
            "kernel void pointwise_mul_sub",
            "kernel void pointwise_scale",
            "kernel void bit_reverse",
            "kernel void ntt_stage",
            "kernel void digits",
            "kernel void bucket_sum_g1",
            "kernel void bucket_sum_g2",
        ] {
            assert!(src.contains(needle), "missing {needle}");
        }
    }
}
