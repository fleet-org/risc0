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

//! Short-Weierstrass curve arithmetic for `y² = x³ + b` (a = 0), generic over
//! the coordinate field so that BN254's G1 (over `Fp`) and G2 (over `Fp2`)
//! share one implementation. Jacobian coordinates for accumulation, affine for
//! storage — the zkey stores affine points and an MSM adds affine points into
//! Jacobian buckets.

use core::fmt;

use crate::{
    consts,
    field::{Field, Fp, Fr},
    fp2::Fp2,
};

/// An affine point, or the point at infinity.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Affine<F: Field> {
    /// x coordinate (unspecified when `infinity`).
    pub x: F,
    /// y coordinate (unspecified when `infinity`).
    pub y: F,
    /// Whether this is the identity.
    pub infinity: bool,
}

/// A point in Jacobian coordinates `(X : Y : Z)` with `x = X/Z²`, `y = Y/Z³`;
/// the identity is any point with `Z = 0`.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Jacobian<F: Field> {
    /// X coordinate.
    pub x: F,
    /// Y coordinate.
    pub y: F,
    /// Z coordinate; zero for the identity.
    pub z: F,
}

impl<F: Field> Affine<F> {
    /// The point at infinity.
    pub const INFINITY: Self = Self {
        x: F::ZERO,
        y: F::ZERO,
        infinity: true,
    };

    /// A finite point.
    pub const fn new(x: F, y: F) -> Self {
        Self {
            x,
            y,
            infinity: false,
        }
    }

    /// `-P`.
    pub fn neg(&self) -> Self {
        if self.infinity {
            *self
        } else {
            Self::new(self.x, self.y.neg())
        }
    }

    /// Whether `y² = x³ + b` holds (the identity is on every curve).
    pub fn is_on_curve(&self, b: &F) -> bool {
        self.infinity || self.y.square() == self.x.square().mul(&self.x).add(b)
    }

    /// Lift to Jacobian coordinates.
    pub fn to_jacobian(&self) -> Jacobian<F> {
        if self.infinity {
            Jacobian::INFINITY
        } else {
            Jacobian {
                x: self.x,
                y: self.y,
                z: F::ONE,
            }
        }
    }
}

impl<F: Field> Jacobian<F> {
    /// The point at infinity.
    pub const INFINITY: Self = Self {
        x: F::ONE,
        y: F::ONE,
        z: F::ZERO,
    };

    /// Whether this is the identity.
    #[inline]
    pub fn is_infinity(&self) -> bool {
        self.z.is_zero()
    }

    /// `-P`.
    pub fn neg(&self) -> Self {
        Self {
            x: self.x,
            y: self.y.neg(),
            z: self.z,
        }
    }

    /// `2P` (dbl-2009-l, a = 0).
    pub fn double(&self) -> Self {
        if self.is_infinity() {
            return *self;
        }
        let a = self.x.square();
        let b = self.y.square();
        let c = b.square();
        let d = self.x.add(&b).square().sub(&a).sub(&c).double();
        let e = a.double().add(&a);
        let f = e.square();
        let x3 = f.sub(&d.double());
        let eight_c = c.double().double().double();
        let y3 = e.mul(&d.sub(&x3)).sub(&eight_c);
        let z3 = self.y.mul(&self.z).double();
        Self {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `P + Q` (add-2007-bl, with the doubling and inverse cases handled).
    pub fn add(&self, other: &Self) -> Self {
        if self.is_infinity() {
            return *other;
        }
        if other.is_infinity() {
            return *self;
        }
        let z1z1 = self.z.square();
        let z2z2 = other.z.square();
        let u1 = self.x.mul(&z2z2);
        let u2 = other.x.mul(&z1z1);
        let s1 = self.y.mul(&other.z).mul(&z2z2);
        let s2 = other.y.mul(&self.z).mul(&z1z1);
        let h = u2.sub(&u1);
        let rr = s2.sub(&s1).double();
        if h.is_zero() {
            return if rr.is_zero() {
                self.double()
            } else {
                Self::INFINITY
            };
        }
        let i = h.double().square();
        let j = h.mul(&i);
        let v = u1.mul(&i);
        let x3 = rr.square().sub(&j).sub(&v.double());
        let y3 = rr.mul(&v.sub(&x3)).sub(&s1.mul(&j).double());
        let z3 = self.z.add(&other.z).square().sub(&z1z1).sub(&z2z2).mul(&h);
        Self {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `P + Q` for an affine `Q` (madd-2007-bl); the MSM bucket step.
    pub fn add_affine(&self, other: &Affine<F>) -> Self {
        if other.infinity {
            return *self;
        }
        if self.is_infinity() {
            return other.to_jacobian();
        }
        let z1z1 = self.z.square();
        let u2 = other.x.mul(&z1z1);
        let s2 = other.y.mul(&self.z).mul(&z1z1);
        let h = u2.sub(&self.x);
        let rr = s2.sub(&self.y).double();
        if h.is_zero() {
            return if rr.is_zero() {
                self.double()
            } else {
                Self::INFINITY
            };
        }
        let hh = h.square();
        let i = hh.double().double();
        let j = h.mul(&i);
        let v = self.x.mul(&i);
        let x3 = rr.square().sub(&j).sub(&v.double());
        let y3 = rr.mul(&v.sub(&x3)).sub(&self.y.mul(&j).double());
        let z3 = self.z.add(&h).square().sub(&z1z1).sub(&hh);
        Self {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `k·P` for a canonical little-endian limb scalar (double-and-add).
    pub fn mul_limbs(&self, k: &[u64; 4]) -> Self {
        let mut acc = Self::INFINITY;
        for limb in k.iter().rev() {
            for bit in (0..64).rev() {
                acc = acc.double();
                if (limb >> bit) & 1 == 1 {
                    acc = acc.add(self);
                }
            }
        }
        acc
    }

    /// `k·P` for a scalar-field element.
    pub fn mul(&self, k: &Fr) -> Self {
        self.mul_limbs(&k.to_canonical())
    }

    /// Normalise to affine coordinates (one field inversion).
    pub fn to_affine(&self) -> Affine<F> {
        match self.z.inverse() {
            None => Affine::INFINITY,
            Some(zinv) => {
                let zinv2 = zinv.square();
                let zinv3 = zinv2.mul(&zinv);
                Affine::new(self.x.mul(&zinv2), self.y.mul(&zinv3))
            }
        }
    }
}

impl<F: Field> PartialEq for Jacobian<F> {
    /// Projective equality: the same affine point (both infinite, or
    /// `X1·Z2² = X2·Z1²` and `Y1·Z2³ = Y2·Z1³`).
    fn eq(&self, other: &Self) -> bool {
        match (self.is_infinity(), other.is_infinity()) {
            (true, true) => true,
            (true, false) | (false, true) => false,
            (false, false) => {
                let z1z1 = self.z.square();
                let z2z2 = other.z.square();
                self.x.mul(&z2z2) == other.x.mul(&z1z1)
                    && self.y.mul(&z2z2).mul(&other.z) == other.y.mul(&z1z1).mul(&self.z)
            }
        }
    }
}

impl<F: Field> Eq for Jacobian<F> {}

impl<F: Field> fmt::Debug for Affine<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.infinity {
            write!(f, "Affine(infinity)")
        } else {
            write!(f, "Affine({:?}, {:?})", self.x, self.y)
        }
    }
}

impl<F: Field> fmt::Debug for Jacobian<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Jacobian({:?})", self.to_affine())
    }
}

/// BN254 G1 in affine coordinates.
pub type G1Affine = Affine<Fp>;
/// BN254 G1 in Jacobian coordinates.
pub type G1Jacobian = Jacobian<Fp>;
/// BN254 G2 in affine coordinates.
pub type G2Affine = Affine<Fp2>;
/// BN254 G2 in Jacobian coordinates.
pub type G2Jacobian = Jacobian<Fp2>;

/// The G1 curve constant `b = 3`.
pub fn g1_b() -> Fp {
    Fp::from_canonical(consts::G1_B_CANONICAL)
}

/// The G2 twist constant `b' = 3/(9+u)`.
pub fn g2_b() -> Fp2 {
    Fp2::new(
        Fp::from_canonical(consts::G2_B_C0_CANONICAL),
        Fp::from_canonical(consts::G2_B_C1_CANONICAL),
    )
}

#[cfg(test)]
pub(crate) mod conv {
    //! Conversions to and from arkworks for tests.
    use ark_ec::AffineRepr as _;
    use ark_ff::{BigInteger as _, PrimeField as _};

    use super::*;

    pub fn fp_from_ark(v: ark_bn254::Fq) -> Fp {
        Fp::from_le_bytes(&v.into_bigint().to_bytes_le().try_into().unwrap()).unwrap()
    }
    pub fn fp_to_ark(v: &Fp) -> ark_bn254::Fq {
        ark_bn254::Fq::from_le_bytes_mod_order(&v.to_le_bytes())
    }
    pub fn fp2_from_ark(v: ark_bn254::Fq2) -> Fp2 {
        Fp2::new(fp_from_ark(v.c0), fp_from_ark(v.c1))
    }
    pub fn fp2_to_ark(v: &Fp2) -> ark_bn254::Fq2 {
        ark_bn254::Fq2::new(fp_to_ark(&v.c0), fp_to_ark(&v.c1))
    }
    pub fn fr_from_ark(v: ark_bn254::Fr) -> Fr {
        Fr::from_le_bytes(&v.into_bigint().to_bytes_le().try_into().unwrap()).unwrap()
    }
    pub fn fr_to_ark(v: &Fr) -> ark_bn254::Fr {
        ark_bn254::Fr::from_le_bytes_mod_order(&v.to_le_bytes())
    }
    pub fn g1_from_ark(p: ark_bn254::G1Affine) -> G1Affine {
        match p.xy() {
            None => G1Affine::INFINITY,
            Some((x, y)) => G1Affine::new(fp_from_ark(x), fp_from_ark(y)),
        }
    }
    pub fn g1_to_ark(p: &G1Affine) -> ark_bn254::G1Affine {
        if p.infinity {
            ark_bn254::G1Affine::identity()
        } else {
            ark_bn254::G1Affine::new_unchecked(fp_to_ark(&p.x), fp_to_ark(&p.y))
        }
    }
    pub fn g2_from_ark(p: ark_bn254::G2Affine) -> G2Affine {
        match p.xy() {
            None => G2Affine::INFINITY,
            Some((x, y)) => G2Affine::new(fp2_from_ark(x), fp2_from_ark(y)),
        }
    }
    pub fn g2_to_ark(p: &G2Affine) -> ark_bn254::G2Affine {
        if p.infinity {
            ark_bn254::G2Affine::identity()
        } else {
            ark_bn254::G2Affine::new_unchecked(fp2_to_ark(&p.x), fp2_to_ark(&p.y))
        }
    }
    pub fn g1_proj_to_ark(p: &G1Jacobian) -> ark_bn254::G1Affine {
        g1_to_ark(&p.to_affine())
    }
    pub fn g2_proj_to_ark(p: &G2Jacobian) -> ark_bn254::G2Affine {
        g2_to_ark(&p.to_affine())
    }

    /// Deterministic scalars for tests.
    pub struct Scalars(pub u64);
    impl Scalars {
        pub fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        pub fn next_fr(&mut self) -> ark_bn254::Fr {
            let mut b = [0u8; 32];
            for chunk in b.chunks_mut(8) {
                chunk.copy_from_slice(&self.next_u64().to_le_bytes());
            }
            ark_bn254::Fr::from_le_bytes_mod_order(&b)
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_ec::{
        short_weierstrass::SWCurveConfig as _, AffineRepr as _, CurveGroup as _, PrimeGroup as _,
    };
    use ark_ff::AdditiveGroup as _;

    use super::conv::*;
    use super::*;

    #[test]
    fn generators_are_on_curve_and_constants_match_arkworks() {
        let g1 = g1_from_ark(ark_bn254::G1Affine::generator());
        let g2 = g2_from_ark(ark_bn254::G2Affine::generator());
        assert!(g1.is_on_curve(&g1_b()));
        assert!(g2.is_on_curve(&g2_b()));
        assert!(
            !G1Affine::new(g1.x, g1.x).is_on_curve(&g1_b()),
            "a wrong point is off-curve"
        );
        assert!(G1Affine::INFINITY.is_on_curve(&g1_b()));
        assert_eq!(fp2_to_ark(&g2_b()), ark_bn254::g2::Config::COEFF_B);
        assert_eq!(fp_to_ark(&g1_b()), ark_bn254::g1::Config::COEFF_B);
    }

    #[test]
    fn g1_arithmetic_matches_arkworks() {
        let mut s = Scalars(0x0123_4567_89ab_cdef);
        let g = ark_bn254::G1Projective::generator();
        for _ in 0..40 {
            let (ka, kb) = (s.next_fr(), s.next_fr());
            let (pa, pb) = (g * ka, g * kb);
            let (ma, mb) = (
                g1_from_ark(pa.into_affine()).to_jacobian(),
                g1_from_ark(pb.into_affine()).to_jacobian(),
            );
            assert_eq!(g1_proj_to_ark(&ma.add(&mb)), (pa + pb).into_affine());
            assert_eq!(
                g1_proj_to_ark(&ma.add_affine(&g1_from_ark(pb.into_affine()))),
                (pa + pb).into_affine()
            );
            assert_eq!(g1_proj_to_ark(&ma.double()), pa.double().into_affine());
            assert_eq!(
                g1_proj_to_ark(&ma.add(&ma)),
                pa.double().into_affine(),
                "add(P, P) doubles"
            );
            assert_eq!(g1_proj_to_ark(&ma.neg()), (-pa).into_affine());
            assert_eq!(
                g1_proj_to_ark(&ma.add(&ma.neg())),
                ark_bn254::G1Affine::identity()
            );
            assert!(ma.add(&ma.neg()).is_infinity());
            assert_eq!(
                g1_proj_to_ark(&ma.add(&G1Jacobian::INFINITY)),
                pa.into_affine()
            );
            assert_eq!(
                g1_proj_to_ark(&G1Jacobian::INFINITY.add_affine(&g1_from_ark(pa.into_affine()))),
                pa.into_affine()
            );
            let k = s.next_fr();
            assert_eq!(
                g1_proj_to_ark(&ma.mul(&fr_from_ark(k))),
                (pa * k).into_affine()
            );
            assert_eq!(
                ma,
                ma.double().add(&ma.neg()),
                "projective equality ignores Z"
            );
        }
        assert_eq!(
            g1_proj_to_ark(&g1_from_ark(g.into_affine()).to_jacobian().mul(&Fr::ZERO)),
            ark_bn254::G1Affine::identity()
        );
    }

    #[test]
    fn g2_arithmetic_matches_arkworks() {
        let mut s = Scalars(0xfedc_ba98_7654_3210);
        let g = ark_bn254::G2Projective::generator();
        for _ in 0..20 {
            let (ka, kb) = (s.next_fr(), s.next_fr());
            let (pa, pb) = (g * ka, g * kb);
            let (ma, mb) = (
                g2_from_ark(pa.into_affine()).to_jacobian(),
                g2_from_ark(pb.into_affine()).to_jacobian(),
            );
            assert_eq!(g2_proj_to_ark(&ma.add(&mb)), (pa + pb).into_affine());
            assert_eq!(
                g2_proj_to_ark(&ma.add_affine(&g2_from_ark(pb.into_affine()))),
                (pa + pb).into_affine()
            );
            assert_eq!(g2_proj_to_ark(&ma.double()), pa.double().into_affine());
            assert_eq!(g2_proj_to_ark(&ma.add(&ma)), pa.double().into_affine());
            assert!(ma.add(&ma.neg()).is_infinity());
            let k = s.next_fr();
            assert_eq!(
                g2_proj_to_ark(&ma.mul(&fr_from_ark(k))),
                (pa * k).into_affine()
            );
        }
    }

    #[test]
    fn affine_roundtrip_and_infinity() {
        let g = g1_from_ark(ark_bn254::G1Affine::generator());
        assert_eq!(g.to_jacobian().to_affine(), g);
        assert_eq!(G1Jacobian::INFINITY.to_affine(), G1Affine::INFINITY);
        assert_eq!(G1Affine::INFINITY.to_jacobian(), G1Jacobian::INFINITY);
        assert_eq!(G1Affine::INFINITY.neg(), G1Affine::INFINITY);
        let three_g = g.to_jacobian().double().add_affine(&g);
        assert_eq!(three_g, g.to_jacobian().mul(&Fr::from_u64(3)));
    }
}
