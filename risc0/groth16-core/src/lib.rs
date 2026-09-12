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

//! Shared arithmetic and formats for the GROTH16 s01 prover arms.
//!
//! Everything a Groth16 prover computes below the `risc0_groth16_sys::prove`
//! boundary — BN254 field and curve arithmetic, the NTT/LDE sequence, the
//! Pippenger MSM, and the zkey / witness / artifact layouts — lives here once,
//! in `no_std` Rust, so that the cuda-oxide kernels (Rust → PTX), the Metal
//! host pipeline and a CPU reference backend cannot drift from each other.
//! The reference backend is how this code is proven on a CPU, against
//! arkworks and the in-tree `multiplier2` fixture, before any GPU host runs it.
//!
//! Encodings (MEASURED on the fixture, see `groth16_s01/BOUNDARY.md` §4.1):
//! zkey points are little-endian Montgomery limbs; zkey coefficient values are
//! `v·R²`; witnesses are canonical little-endian.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

pub mod consts;
pub mod ec;
pub mod field;
pub mod fp2;
#[cfg(feature = "std")]
pub mod msm;
pub mod ntt;
#[cfg(feature = "std")]
pub mod prover;
#[cfg(feature = "std")]
pub mod zkey;
