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

//! `BackendKind::MetalCpu`: the Metal arm's shaders compiled as C++ and run
//! on the CPU (`risc0_groth16_metal::device::HostMslProver`), behind the same
//! boundary — the Metal counterpart of `oxide-cpu`: it proves the exact
//! shader source on a machine without a Metal device, through the harness
//! and the upstream verifier. A testing kind, never the default, not a
//! production path. Available on every target but macOS (there the real
//! `metal` kind runs the same pipeline on the device).

use anyhow::{anyhow, Context as _};
use risc0_groth16_core::{
    prover::{proof_json, public_json, CoefficientGroups},
    zkey::{parse_witness_values, Zkey},
};
use risc0_groth16_metal::device::{HostBackend, HostMslProver, ResidentZkey};

use super::{reference::random_scalar, resident, BackendKind, Groth16Backend};
use crate::{ProverParams, SetupParams};

/// The Metal shaders on the CPU.
pub struct MetalCpu;

impl Groth16Backend for MetalCpu {
    fn kind(&self) -> BackendKind {
        BackendKind::MetalCpu
    }

    fn prove(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()> {
        let path = setup.srs_path.as_path();
        let prepare = || -> anyhow::Result<Prepared> {
            let mut zkey = {
                let zkey_bytes = std::fs::read(path)
                    .with_context(|| format!("reading zkey {}", path.display()))?;
                Zkey::parse(&zkey_bytes).context("parsing zkey")?
            };
            let groups = CoefficientGroups::from_zkey(&zkey);
            zkey.coefficients = Vec::new();
            let device = HostMslProver::new()?;
            let resident = device.prepare(&zkey, &groups);
            Ok(Prepared { device, resident })
        };
        let p = if resident::enabled() {
            CACHE.get_or_prepare(resident::Key::of(path)?, prepare)?
        } else {
            std::sync::Arc::new(prepare()?)
        };
        let (num_vars, num_public) = (p.resident.num_vars(), p.resident.num_public());
        let witness_bytes = unsafe { std::slice::from_raw_parts(prover.witness, num_vars * 32) };
        let witness = parse_witness_values(witness_bytes)
            .map_err(|i| anyhow!("witness value {i} is not a field element"))?;
        let (r, s) = (random_scalar()?, random_scalar()?);
        let proof = p
            .device
            .prove_resident(&p.resident, &witness, &r, &s)
            .context("metal shaders on the CPU")?;
        std::fs::write(
            prover.public_path.as_path(),
            public_json(&witness, num_public),
        )
        .context("writing public.json")?;
        std::fs::write(prover.proof_path.as_path(), proof_json(&proof))
            .context("writing proof.json")?;
        Ok(())
    }
}

/// The CPU executor and the zkey resident in its buffers.
struct Prepared {
    device: HostMslProver,
    resident: ResidentZkey<HostBackend>,
}

// SAFETY: the host buffers are plain `Vec<u8>`s behind `UnsafeCell`; the cache hands out an
// `Arc` and the prover runs sequentially per call, never sharing a buffer across threads
// mid-run (the same invariant the Metal device backend relies on).
unsafe impl Send for Prepared {}
unsafe impl Sync for Prepared {}

static CACHE: resident::Cache<Prepared> = resident::Cache::new();

#[cfg(test)]
mod tests {
    use crate::backend::fixture;

    /// The Metal shaders on the CPU through the boundary: the same three
    /// properties as the reference backend's, on any Linux box.
    #[test]
    fn metal_cpu_through_the_boundary() {
        fixture::three_properties("metal-cpu");
    }
}
