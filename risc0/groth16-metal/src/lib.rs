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

//! The Metal arm (GROTH16 s01/4b): the Groth16 prover stage for Apple
//! Silicon on macOS 13+, where `risc0-groth16` has no GPU path at all.
//!
//! Shape: the kernels (`kernels.metal`) mirror the cuda-oxide arm's kernel
//! bodies one to one; the host pipeline ([`device`], macOS/aarch64 only)
//! mirrors `risc0-groth16-oxide`'s pipeline — same buffers, same launch
//! sequence, same host-side sort / reduction / assembly from
//! `risc0-groth16-core`. Only the launcher and the shader language differ,
//! which is what keeps the arm substitutable behind the boundary.
//!
//! The shader source is compiled at run time (`new_library_with_source`),
//! so no Metal toolchain is needed to build the crate, Linux CI can
//! type-check the host side, and the macOS deployment floor is a run-time
//! property: the pipeline uses no feature beyond Metal 2 (macOS 13 supports
//! Metal 3; MSL 3.0 is the language version requested).
//!
//! Records ([`pack`]) are the exact byte layouts the kernels read; they are
//! pure Rust and tested on every host.

#![deny(missing_docs)]

pub mod pack;

pub mod device;

/// The complete shader source: constants, then kernels.
pub const MSL_SOURCE: &str = concat!(
    include_str!("consts.metal"),
    "\n",
    include_str!("kernels.metal")
);

/// Pippenger window width used by this arm (bucket sums are one thread each).
pub const WINDOW_BITS: u32 = 12;
