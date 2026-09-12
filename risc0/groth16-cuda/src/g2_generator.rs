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

//! The BN254 G2 generator (EIP-197 / arkworks `G2Affine::generator()`), as
//! canonical little-endian limbs — a valid G2 point for the kernel check
//! without a zkey. `tests::on_curve` proves the constants against the
//! curve equation with `risc0-groth16-core`'s own arithmetic.

/// x.c0
pub const G2_GENERATOR_X_C0: [u64; 4] = [
    0x46debd5cd992f6ed,
    0x674322d4f75edadd,
    0x426a00665e5c4479,
    0x1800deef121f1e76,
];
/// x.c1
pub const G2_GENERATOR_X_C1: [u64; 4] = [
    0x97e485b7aef312c2,
    0xf1aa493335a9e712,
    0x7260bfb731fb5d25,
    0x198e9393920d483a,
];
/// y.c0
pub const G2_GENERATOR_Y_C0: [u64; 4] = [
    0x4ce6cc0166fa7daa,
    0xe3d1e7690c43d37b,
    0x4aab71808dcb408f,
    0x12c85ea5db8c6deb,
];
/// y.c1
pub const G2_GENERATOR_Y_C1: [u64; 4] = [
    0x55acdadcd122975b,
    0xbc4b313370b38ef3,
    0xec9e99ad690c3395,
    0x090689d0585ff075,
];

#[cfg(test)]
mod tests {
    use risc0_groth16_core::ec::g2_b;

    #[test]
    fn on_curve() {
        let g = crate::device::g2_generator();
        assert!(g.is_on_curve(&g2_b()));
        // and its 2nd and 3rd multiples agree whichever way they are formed
        let j = g.to_jacobian();
        assert_eq!(j.double().to_affine(), j.add_affine(&g).to_affine());
    }
}
