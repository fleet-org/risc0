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

//! GROTH16 s01/4 — the CUDA arm's host side.
//!
//! The arm has two halves, joined by [`risc0_groth16_oxide::abi`]:
//!
//! - the **device module**: `#[kernel]` wrappers around
//!   [`risc0_groth16_oxide::kernels`] (the per-output-index bodies the CPU
//!   launcher has already proven on the production circuit), compiled by
//!   `cargo oxide` to PTX in `groth16_s01/cuda-kernels`;
//! - this crate, the **host**: loads that module through `cuda-core`, the
//!   driver-API runtime cuda-oxide itself uses, and runs the prover pipeline
//!   — the same sequence as `risc0_groth16_oxide::pipeline`, with device
//!   buffers where the CPU launcher has slices.
//!
//! What runs where: scatter, NTT stages, pointwise passes, MSM digits and
//! bucket sums on the device; the counting sort between digits and bucket
//! sums, the bucket reduction, Horner, and the final assembly on the host —
//! exactly the split of the Metal arm, so the two arms differ only in the
//! device API.
//!
//! Where the module comes from: [`ModuleSource::from_env`] reads
//! `RISC0_GROTH16_CUDA_MODULE` (a `.ptx`, a `.cubin`/`.fatbin`, or a
//! `cargo oxide` build product carrying an artifact bundle); unset, the
//! module embedded in the running executable is used.
//!
//! Status: type-checked without a GPU (CI does the same with the rootless
//! CUDA headers). Every launch is UNVERIFIED until `groth16-cuda-kernel-check`
//! has run on a CUDA host — that binary is the first thing to run there.

#![deny(missing_docs)]

pub mod device;
pub mod g2_generator;
pub mod module;

pub use device::{CudaProver, KernelCheck};
pub use module::ModuleSource;
use risc0_groth16_core::{
    field::Fr,
    prover::{CoefficientGroups, Proof},
    zkey::Zkey,
};

/// Produce a proof on the GPU named by the environment
/// (`RISC0_GROTH16_CUDA_MODULE`, device 0), grouping the zkey's coefficients
/// first.
pub fn prove(zkey: &Zkey, witness: &[Fr], r: &Fr, s: &Fr) -> anyhow::Result<Proof> {
    let groups = CoefficientGroups::from_zkey(zkey);
    prove_grouped(zkey, &groups, witness, r, s)
}

/// [`prove`] with the coefficient groups already built (the caller may then
/// drop `zkey.coefficients`, the largest part of a parsed zkey).
pub fn prove_grouped(
    zkey: &Zkey,
    groups: &CoefficientGroups,
    witness: &[Fr],
    r: &Fr,
    s: &Fr,
) -> anyhow::Result<Proof> {
    let prover = CudaProver::new(ModuleSource::from_env()?)?;
    prover.prove_grouped(zkey, groups, witness, r, s)
}
