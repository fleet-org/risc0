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

//! The cuda-oxide arm's kernel bodies behind the boundary, launched on the
//! host (`risc0_groth16_oxide::launch::CpuLauncher`). Same inputs, same
//! outputs and the same host pipeline as the device launcher will use; only
//! the launcher differs. Proves the kernels on a CPU; not a production path.

use anyhow::{anyhow, Context as _};
use risc0_groth16_core::{
    prover::{proof_json, public_json, CoefficientGroups},
    zkey::{parse_witness_values, Zkey},
};
use risc0_groth16_oxide::{launch::CpuLauncher, pipeline};

use super::{reference::random_scalar, BackendKind, Groth16Backend};
use crate::{ProverParams, SetupParams};

/// The host-launched cuda-oxide pipeline.
pub struct OxideCpu;

impl Groth16Backend for OxideCpu {
    fn kind(&self) -> BackendKind {
        BackendKind::OxideCpu
    }

    fn prove(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()> {
        // Parse, then drop the file bytes: the zkey is multi-GB and the proof holds memory long.
        let mut zkey = {
            let zkey_bytes = std::fs::read(setup.srs_path.as_path())
                .with_context(|| format!("reading zkey {}", setup.srs_path.as_path().display()))?;
            Zkey::parse(&zkey_bytes).context("parsing zkey")?
        };
        // Group the coefficients for the scatter kernel and drop the flat list (1.4 GB on the
        // production circuit): the pipeline needs only the groups.
        let groups = CoefficientGroups::from_zkey(&zkey);
        zkey.coefficients = Vec::new();
        // SAFETY: the boundary contract (BOUNDARY.md §4.1): `witness` points at `num_vars`
        // consecutive 32-byte canonical field elements, `num_vars` from the zkey header.
        let witness_bytes =
            unsafe { std::slice::from_raw_parts(prover.witness, zkey.num_vars * 32) };
        let witness = parse_witness_values(witness_bytes)
            .map_err(|i| anyhow!("witness value {i} is not a field element"))?;
        let (r, s) = (random_scalar()?, random_scalar()?);
        let proof =
            pipeline::prove_grouped(&CpuLauncher::default(), &zkey, &groups, &witness, &r, &s)
                .context("oxide (host-launched) prover")?;
        std::fs::write(
            prover.public_path.as_path(),
            public_json(&witness, zkey.num_public),
        )
        .context("writing public.json")?;
        std::fs::write(prover.proof_path.as_path(), proof_json(&proof))
            .context("writing proof.json")?;
        Ok(())
    }
}
