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

//! The BN254 prime fields — `Fp` (base field, modulus q) and `Fr` (scalar
//! field, modulus r) — in Montgomery form with four little-endian 64-bit limbs
//! (R = 2^256).
//!
//! Plain Rust with `u128` products and no intrinsics, so the same code runs on
//! the host, compiles to PTX through cuda-oxide (which lowers `u128`
//! arithmetic), and serves the Metal host pipeline. Every operation returns a
//! fully reduced representative, so limb equality is field equality.

#![allow(clippy::needless_range_loop)]

use core::fmt;

use crate::consts;

/// What the curve and polynomial code needs from a field.
pub trait Field: Copy + Clone + PartialEq + Eq + fmt::Debug {
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;
    /// `self + other`.
    fn add(&self, other: &Self) -> Self;
    /// `self - other`.
    fn sub(&self, other: &Self) -> Self;
    /// `self * other`.
    fn mul(&self, other: &Self) -> Self;
    /// `-self`.
    fn neg(&self) -> Self;
    /// `self^2`.
    fn square(&self) -> Self {
        self.mul(self)
    }
    /// `2 * self`.
    fn double(&self) -> Self {
        self.add(self)
    }
    /// Whether `self == 0`.
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }
    /// `self^-1`, or `None` for zero.
    fn inverse(&self) -> Option<Self>;
}

#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let t = (a as u128) + (b as u128) * (c as u128) + (carry as u128);
    (t as u64, (t >> 64) as u64)
}

#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = (a as u128) + (b as u128) + (carry as u128);
    (t as u64, (t >> 64) as u64)
}

#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let t = (a as u128).wrapping_sub((b as u128) + (borrow as u128));
    (t as u64, ((t >> 64) as u64) & 1)
}

/// `a >= m` on little-endian limbs.
#[inline(always)]
fn geq(a: &[u64; 4], m: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] != m[i] {
            return a[i] > m[i];
        }
    }
    true
}

/// `a - m` on little-endian limbs (caller guarantees `a >= m`).
#[inline(always)]
fn sub_limbs(a: &[u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let mut r = [0u64; 4];
    let mut borrow = 0;
    for i in 0..4 {
        let (d, b) = sbb(a[i], m[i], borrow);
        r[i] = d;
        borrow = b;
    }
    r
}

/// Montgomery multiplication (CIOS): returns `a * b * R^-1 mod m`, reduced.
#[inline(always)]
fn mont_mul(a: &[u64; 4], b: &[u64; 4], m: &[u64; 4], inv: u64) -> [u64; 4] {
    let mut t = [0u64; 6];
    for i in 0..4 {
        let mut carry = 0;
        for j in 0..4 {
            let (lo, c) = mac(t[j], a[j], b[i], carry);
            t[j] = lo;
            carry = c;
        }
        let (lo, c) = adc(t[4], 0, carry);
        t[4] = lo;
        t[5] = c;

        let k = t[0].wrapping_mul(inv);
        let (_, mut carry) = mac(t[0], k, m[0], 0);
        for j in 1..4 {
            let (lo, c) = mac(t[j], k, m[j], carry);
            t[j - 1] = lo;
            carry = c;
        }
        let (lo, c) = adc(t[4], 0, carry);
        t[3] = lo;
        t[4] = t[5] + c;
    }
    let r = [t[0], t[1], t[2], t[3]];
    if t[4] != 0 || geq(&r, m) {
        sub_limbs(&r, m)
    } else {
        r
    }
}

macro_rules! prime_field {
    ($name:ident, $doc:literal, $modulus:expr, $r:expr, $r2:expr, $inv:expr, $pm2:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Default)]
        #[repr(C)]
        pub struct $name([u64; 4]);

        // `Hash` by hand too, over the same limbs, so it agrees with the hand-written `PartialEq`.
        impl core::hash::Hash for $name {
            #[inline]
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                self.0.hash(state);
            }
        }

        // Not derived: a derived `PartialEq` on `[u64; 4]` lowers to the `raw_eq` intrinsic, which
        // cuda-oxide's device backend does not support (MEASURED, C13). Limb-wise it is plain
        // integer arithmetic — and branch-free, which the host is happy with too.
        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                let mut acc = 0u64;
                let mut i = 0;
                while i < 4 {
                    acc |= self.0[i] ^ other.0[i];
                    i += 1;
                }
                acc == 0
            }
        }

        impl Eq for $name {}

        impl $name {
            /// The modulus, little-endian limbs.
            pub const MODULUS: [u64; 4] = $modulus;

            /// Wrap raw Montgomery-form limbs (what a snarkjs zkey stores for
            /// curve coordinates). The limbs must already be reduced.
            #[inline]
            pub const fn from_montgomery_limbs(limbs: [u64; 4]) -> Self {
                Self(limbs)
            }

            /// The Montgomery-form limbs (what a GPU kernel operates on).
            #[inline]
            pub const fn montgomery_limbs(&self) -> [u64; 4] {
                self.0
            }

            /// From a canonical (non-Montgomery) value `x < MODULUS`.
            #[inline]
            pub fn from_canonical(x: [u64; 4]) -> Self {
                debug_assert!(!geq(&x, &Self::MODULUS), "value not reduced");
                Self(mont_mul(&x, &$r2, &Self::MODULUS, $inv))
            }

            /// To the canonical (non-Montgomery) value.
            #[inline]
            pub fn to_canonical(&self) -> [u64; 4] {
                mont_mul(&self.0, &[1, 0, 0, 0], &Self::MODULUS, $inv)
            }

            /// From a small integer.
            #[inline]
            pub fn from_u64(v: u64) -> Self {
                Self::from_canonical([v, 0, 0, 0])
            }

            /// From 32 canonical little-endian bytes; `None` if not below the modulus.
            pub fn from_le_bytes(bytes: &[u8; 32]) -> Option<Self> {
                let mut limbs = [0u64; 4];
                for (i, limb) in limbs.iter_mut().enumerate() {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
                    *limb = u64::from_le_bytes(b);
                }
                if geq(&limbs, &Self::MODULUS) {
                    None
                } else {
                    Some(Self::from_canonical(limbs))
                }
            }

            /// To 32 canonical little-endian bytes.
            pub fn to_le_bytes(&self) -> [u8; 32] {
                let limbs = self.to_canonical();
                let mut out = [0u8; 32];
                for (i, limb) in limbs.iter().enumerate() {
                    out[i * 8..i * 8 + 8].copy_from_slice(&limb.to_le_bytes());
                }
                out
            }

            /// `self^exp` for a little-endian limb exponent.
            pub fn pow(&self, exp: &[u64; 4]) -> Self {
                let mut result = <Self as Field>::ONE;
                for limb in exp.iter().rev() {
                    for bit in (0..64).rev() {
                        result = result.square();
                        if (limb >> bit) & 1 == 1 {
                            result = result.mul(self);
                        }
                    }
                }
                result
            }
        }

        impl Field for $name {
            const ZERO: Self = Self([0, 0, 0, 0]);
            const ONE: Self = Self($r);

            #[inline]
            fn add(&self, other: &Self) -> Self {
                let mut r = [0u64; 4];
                let mut carry = 0;
                for i in 0..4 {
                    let (s, c) = adc(self.0[i], other.0[i], carry);
                    r[i] = s;
                    carry = c;
                }
                if carry != 0 || geq(&r, &Self::MODULUS) {
                    Self(sub_limbs(&r, &Self::MODULUS))
                } else {
                    Self(r)
                }
            }

            #[inline]
            fn sub(&self, other: &Self) -> Self {
                let mut r = [0u64; 4];
                let mut borrow = 0;
                for i in 0..4 {
                    let (d, b) = sbb(self.0[i], other.0[i], borrow);
                    r[i] = d;
                    borrow = b;
                }
                if borrow != 0 {
                    let mut carry = 0;
                    for i in 0..4 {
                        let (s, c) = adc(r[i], Self::MODULUS[i], carry);
                        r[i] = s;
                        carry = c;
                    }
                }
                Self(r)
            }

            #[inline]
            fn mul(&self, other: &Self) -> Self {
                Self(mont_mul(&self.0, &other.0, &Self::MODULUS, $inv))
            }

            #[inline]
            fn neg(&self) -> Self {
                if self.is_zero() {
                    *self
                } else {
                    Self(sub_limbs(&Self::MODULUS, &self.0))
                }
            }

            fn inverse(&self) -> Option<Self> {
                if self.is_zero() {
                    None
                } else {
                    Some(self.pow(&$pm2))
                }
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let c = self.to_canonical();
                write!(
                    f,
                    "{}(0x{:016x}{:016x}{:016x}{:016x})",
                    stringify!($name),
                    c[3],
                    c[2],
                    c[1],
                    c[0]
                )
            }
        }
    };
}

prime_field!(
    Fp,
    "The BN254 base field (modulus q), Montgomery form.",
    consts::FP_MODULUS,
    consts::FP_R,
    consts::FP_R2,
    consts::FP_INV,
    consts::FP_MODULUS_MINUS_2
);

prime_field!(
    Fr,
    "The BN254 scalar field (modulus r), Montgomery form.",
    consts::FR_MODULUS,
    consts::FR_R,
    consts::FR_R2,
    consts::FR_INV,
    consts::FR_MODULUS_MINUS_2
);

impl Fr {
    /// A primitive `2^k`-th root of unity, `k <= 28`.
    pub fn two_adic_root(k: u32) -> Self {
        assert!(
            k <= consts::FR_TWO_ADICITY,
            "k exceeds the two-adicity of r - 1"
        );
        let mut root = Self::from_canonical(consts::FR_TWO_ADIC_ROOT_CANONICAL);
        for _ in k..consts::FR_TWO_ADICITY {
            root = root.square();
        }
        root
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup as _, BigInteger as _, FftField as _, Field as _, PrimeField as _};

    use super::*;

    /// A deterministic generator so every run exercises the same values.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn bytes(&mut self) -> [u8; 32] {
            let mut b = [0u8; 32];
            for chunk in b.chunks_mut(8) {
                chunk.copy_from_slice(&self.next().to_le_bytes());
            }
            b
        }
    }

    fn limbs_of<F: ark_ff::PrimeField<BigInt = ark_ff::BigInt<4>>>(x: F) -> [u64; 4] {
        x.into_bigint().0
    }

    macro_rules! field_conformance {
        ($test:ident, $mine:ty, $ark:ty, $cfg:ty) => {
            #[test]
            fn $test() {
                // Constants: R^2 and the reduction constant must be arkworks' own.
                assert_eq!(<$mine>::MODULUS, <$cfg as ark_ff::MontConfig<4>>::MODULUS.0);
                assert_eq!(
                    <$mine as Field>::ONE.montgomery_limbs(),
                    <$ark>::ONE.0 .0,
                    "ONE must be R"
                );
                assert_eq!(
                    <$mine>::from_u64(1).montgomery_limbs(),
                    <$cfg as ark_ff::MontConfig<4>>::R.0,
                    "R must be arkworks' R"
                );
                assert_eq!(<$mine>::from_u64(1).to_canonical(), [1, 0, 0, 0]);
                let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
                for _ in 0..500 {
                    let ba = <$ark>::from_le_bytes_mod_order(&rng.bytes());
                    let bb = <$ark>::from_le_bytes_mod_order(&rng.bytes());
                    let a =
                        <$mine>::from_le_bytes(&ba.into_bigint().to_bytes_le().try_into().unwrap())
                            .unwrap();
                    let b =
                        <$mine>::from_le_bytes(&bb.into_bigint().to_bytes_le().try_into().unwrap())
                            .unwrap();
                    assert_eq!(a.to_canonical(), limbs_of(ba));
                    assert_eq!(a.add(&b).to_canonical(), limbs_of(ba + bb));
                    assert_eq!(a.sub(&b).to_canonical(), limbs_of(ba - bb));
                    assert_eq!(b.sub(&a).to_canonical(), limbs_of(bb - ba));
                    assert_eq!(a.mul(&b).to_canonical(), limbs_of(ba * bb));
                    assert_eq!(a.square().to_canonical(), limbs_of(ba.square()));
                    assert_eq!(a.double().to_canonical(), limbs_of(ba.double()));
                    assert_eq!(a.neg().to_canonical(), limbs_of(-ba));
                    assert_eq!(
                        a.inverse().unwrap().to_canonical(),
                        limbs_of(ba.inverse().unwrap())
                    );
                    let e = rng.next();
                    assert_eq!(a.pow(&[e, 0, 0, 0]).to_canonical(), limbs_of(ba.pow([e])));
                    // Montgomery limbs are what arkworks holds internally, and what the zkey stores.
                    assert_eq!(
                        a.montgomery_limbs(),
                        ba.0 .0,
                        "Montgomery representation must match arkworks"
                    );
                    assert_eq!(<$mine>::from_montgomery_limbs(ba.0 .0), a);
                    // Limb equality is field equality: every result is fully reduced.
                    assert!(!geq(&a.mul(&b).montgomery_limbs(), &<$mine>::MODULUS));
                    assert!(!geq(&a.add(&b).montgomery_limbs(), &<$mine>::MODULUS));
                    assert_eq!(<$mine>::from_le_bytes(&a.to_le_bytes()).unwrap(), a);
                }
            }
        };
    }

    field_conformance!(fp_matches_arkworks, Fp, ark_bn254::Fq, ark_bn254::FqConfig);
    field_conformance!(fr_matches_arkworks, Fr, ark_bn254::Fr, ark_bn254::FrConfig);

    #[test]
    fn edge_values_behave() {
        let p_minus_1 = Fp::from_canonical(sub_limbs(&Fp::MODULUS, &[1, 0, 0, 0]));
        assert_eq!(p_minus_1.add(&Fp::ONE), Fp::ZERO);
        assert_eq!(p_minus_1.mul(&p_minus_1), Fp::ONE);
        assert_eq!(p_minus_1.neg(), Fp::ONE);
        assert_eq!(Fp::ZERO.neg(), Fp::ZERO);
        assert_eq!(Fp::ZERO.inverse(), None);
        assert_eq!(Fp::ONE.inverse(), Some(Fp::ONE));
        assert_eq!(Fp::ZERO.sub(&Fp::ONE), p_minus_1);
        assert_eq!(Fr::from_u64(7).pow(&[0, 0, 0, 0]), Fr::ONE);
    }

    #[test]
    fn from_le_bytes_rejects_unreduced_values() {
        let mut modulus = [0u8; 32];
        for (i, limb) in Fp::MODULUS.iter().enumerate() {
            modulus[i * 8..i * 8 + 8].copy_from_slice(&limb.to_le_bytes());
        }
        assert!(
            Fp::from_le_bytes(&modulus).is_none(),
            "the modulus itself is not an element"
        );
        assert!(Fp::from_le_bytes(&[0xff; 32]).is_none());
        let mut p_minus_1 = modulus;
        p_minus_1[0] -= 1;
        assert!(Fp::from_le_bytes(&p_minus_1).is_some());
    }

    #[test]
    fn two_adic_roots_match_arkworks() {
        let root = Fr::two_adic_root(consts::FR_TWO_ADICITY);
        assert_eq!(
            root.to_canonical(),
            limbs_of(ark_bn254::Fr::TWO_ADIC_ROOT_OF_UNITY)
        );
        assert_eq!(
            Fr::from_canonical(consts::FR_GENERATOR_CANONICAL).to_canonical(),
            limbs_of(ark_bn254::Fr::GENERATOR)
        );
        for k in [0u32, 1, 2, 10, 28] {
            let w = Fr::two_adic_root(k);
            assert_eq!(w.pow(&[1u64 << k, 0, 0, 0]), Fr::ONE, "w^(2^k) == 1");
            if k > 0 {
                assert_ne!(w.pow(&[1u64 << (k - 1), 0, 0, 0]), Fr::ONE, "primitive");
            }
        }
        assert_eq!(
            Fr::two_adic_root(2).to_canonical(),
            limbs_of(ark_bn254::Fr::get_root_of_unity(4).unwrap())
        );
    }
}
