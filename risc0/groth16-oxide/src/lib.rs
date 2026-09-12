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

//! The CUDA arm (GROTH16 s01/4) in the shape cuda-oxide compiles: every
//! kernel is a plain Rust function of *one thread's work* over the shared
//! arithmetic in `risc0-groth16-core`, and the pipeline — buffers, the
//! per-window digit sort, the launch sequence, the assembly — is host Rust.
//!
//! Why this shape: cuda-oxide consumes Rust, not CUDA C++ (`CUDA_OXIDE_PIN.md`
//! §3), so the kernel bodies are the same code on the CPU and on the GPU. The
//! [`launch::Launcher`] trait is the seam: [`launch::CpuLauncher`] runs the
//! grid on the host and lets the whole pipeline be proven against the
//! upstream verifier before a CUDA host exists; the cuda-oxide launcher (a
//! `#[cuda_module]` whose `#[kernel]` wrappers call these bodies, built with
//! `cargo oxide` on a CUDA host) is the production one. Neither launcher is
//! registered as a backend by this crate: registration is the boundary's job
//! (`risc0-groth16-sys`), and only the real device launcher may claim
//! `BackendKind::CudaOxide`.
//!
//! Zero-conversion inputs (BOUNDARY.md §4.1): witness limbs are the canonical
//! values as stored, coefficient values are `v·R²` as stored, points are
//! Montgomery as stored. `Fr::from_montgomery_limbs(w) * Fr::from_montgomery_limbs(vR²)`
//! is the element `w·v`, bit-identical to the canonical kernels' scatter.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

pub mod abi;
#[cfg(feature = "std")]
pub mod check;
pub mod kernels;
#[cfg(feature = "std")]
pub mod launch;
#[cfg(feature = "std")]
pub mod pipeline;
pub mod schedule;
