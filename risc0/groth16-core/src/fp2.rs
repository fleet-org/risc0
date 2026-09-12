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

//! The quadratic extension `Fp2 = Fp[u] / (u^2 + 1)` that BN254's G2 lives on.

use core::fmt;

use crate::field::{Field, Fp};

/// An element `c0 + c1·u` with `u^2 = -1`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Fp2 {
    /// The real part.
    pub c0: Fp,
    /// The coefficient of `u`.
    pub c1: Fp,
}

impl Fp2 {
    /// Build from the two coordinates.
    pub const fn new(c0: Fp, c1: Fp) -> Self {
        Self { c0, c1 }
    }

    /// Multiply by the base-field element `k`.
    pub fn scale(&self, k: &Fp) -> Self {
        Self::new(self.c0.mul(k), self.c1.mul(k))
    }
}

impl Field for Fp2 {
    const ZERO: Self = Self::new(Fp::ZERO, Fp::ZERO);
    const ONE: Self = Self::new(Fp::ONE, Fp::ZERO);

    #[inline]
    fn add(&self, other: &Self) -> Self {
        Self::new(self.c0.add(&other.c0), self.c1.add(&other.c1))
    }

    #[inline]
    fn sub(&self, other: &Self) -> Self {
        Self::new(self.c0.sub(&other.c0), self.c1.sub(&other.c1))
    }

    #[inline]
    fn mul(&self, other: &Self) -> Self {
        // (a0 + a1 u)(b0 + b1 u) = (a0 b0 - a1 b1) + (a0 b1 + a1 b0) u
        let a0b0 = self.c0.mul(&other.c0);
        let a1b1 = self.c1.mul(&other.c1);
        let a0b1 = self.c0.mul(&other.c1);
        let a1b0 = self.c1.mul(&other.c0);
        Self::new(a0b0.sub(&a1b1), a0b1.add(&a1b0))
    }

    #[inline]
    fn square(&self) -> Self {
        // (a0 + a1 u)^2 = (a0 - a1)(a0 + a1) + 2 a0 a1 u
        let c0 = self.c0.sub(&self.c1).mul(&self.c0.add(&self.c1));
        let c1 = self.c0.mul(&self.c1).double();
        Self::new(c0, c1)
    }

    #[inline]
    fn neg(&self) -> Self {
        Self::new(self.c0.neg(), self.c1.neg())
    }

    fn inverse(&self) -> Option<Self> {
        // (a0 + a1 u)^-1 = (a0 - a1 u) / (a0^2 + a1^2)
        let norm = self.c0.square().add(&self.c1.square());
        let inv = norm.inverse()?;
        Some(Self::new(self.c0.mul(&inv), self.c1.mul(&inv).neg()))
    }
}

impl fmt::Debug for Fp2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp2({:?} + {:?} u)", self.c0, self.c1)
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup as _, BigInteger as _, Field as _, PrimeField as _};

    use super::*;

    fn to_ark(x: &Fp2) -> ark_bn254::Fq2 {
        let c = |v: &Fp| ark_bn254::Fq::from_le_bytes_mod_order(&v.to_le_bytes());
        ark_bn254::Fq2::new(c(&x.c0), c(&x.c1))
    }

    fn from_ark(x: &ark_bn254::Fq2) -> Fp2 {
        let c = |v: ark_bn254::Fq| {
            Fp::from_le_bytes(&v.into_bigint().to_bytes_le().try_into().unwrap()).unwrap()
        };
        Fp2::new(c(x.c0), c(x.c1))
    }

    #[test]
    fn fp2_matches_arkworks() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let rand_fp = |next: &mut dyn FnMut() -> u64| {
            let mut b = [0u8; 32];
            for chunk in b.chunks_mut(8) {
                chunk.copy_from_slice(&next().to_le_bytes());
            }
            ark_bn254::Fq::from_le_bytes_mod_order(&b)
        };
        for _ in 0..300 {
            let a = ark_bn254::Fq2::new(rand_fp(&mut next), rand_fp(&mut next));
            let b = ark_bn254::Fq2::new(rand_fp(&mut next), rand_fp(&mut next));
            let (ma, mb) = (from_ark(&a), from_ark(&b));
            assert_eq!(to_ark(&ma.add(&mb)), a + b);
            assert_eq!(to_ark(&ma.sub(&mb)), a - b);
            assert_eq!(to_ark(&ma.mul(&mb)), a * b);
            assert_eq!(to_ark(&ma.square()), a.square());
            assert_eq!(ma.square(), ma.mul(&ma), "square agrees with mul");
            assert_eq!(to_ark(&ma.neg()), -a);
            assert_eq!(to_ark(&ma.double()), a.double());
            assert_eq!(to_ark(&ma.inverse().unwrap()), a.inverse().unwrap());
            assert_eq!(ma.mul(&ma.inverse().unwrap()), Fp2::ONE);
            let k = rand_fp(&mut next);
            let mk = from_ark(&ark_bn254::Fq2::new(k, ark_bn254::Fq::ZERO)).c0;
            assert_eq!(
                to_ark(&ma.scale(&mk)),
                a * ark_bn254::Fq2::new(k, ark_bn254::Fq::ZERO)
            );
        }
        assert_eq!(Fp2::ZERO.inverse(), None);
        assert_eq!(to_ark(&Fp2::ONE), ark_bn254::Fq2::ONE);
        // u^2 = -1
        let u = Fp2::new(Fp::ZERO, Fp::ONE);
        assert_eq!(u.square(), Fp2::ONE.neg());
    }
}
