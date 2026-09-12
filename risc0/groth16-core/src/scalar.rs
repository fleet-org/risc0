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

//! Scalar digit decomposition for windowed multi-scalar multiplication —
//! `no_std`, because a kernel computes digits on the device.

/// The `w`-bit digit `window` of a canonical scalar (little-endian limbs).
#[inline]
pub fn digit(scalar: &[u64; 4], window: u32, w: u32) -> u64 {
    let bit = (window * w) as usize;
    let limb = bit / 64;
    if limb >= 4 {
        return 0;
    }
    let shift = bit % 64;
    let mut d = scalar[limb] >> shift;
    if shift + (w as usize) > 64 && limb + 1 < 4 {
        d |= scalar[limb + 1] << (64 - shift);
    }
    d & ((1u64 << w) - 1)
}
