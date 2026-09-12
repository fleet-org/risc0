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

//! `BackendKind::CudaOxide`: the CUDA arm (GROTH16 s01/4) behind the
//! boundary — the zkey and witness are read exactly as the reference does,
//! the proof is produced by `risc0-groth16-cuda` on device 0 from the module
//! named by `RISC0_GROTH16_CUDA_MODULE`, and the JSON files are written where
//! the canonical path writes them.

use anyhow::{anyhow, Context as _};
use risc0_groth16_core::{
    prover::{proof_json, public_json, CoefficientGroups},
    zkey::{parse_witness_values, Zkey},
};

use super::{reference::random_scalar, BackendKind, Groth16Backend};
use crate::{ProverParams, SetupParams};

/// The CUDA arm.
pub struct CudaOxide;

impl Groth16Backend for CudaOxide {
    fn kind(&self) -> BackendKind {
        BackendKind::CudaOxide
    }

    fn prove(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()> {
        let mut zkey = {
            let zkey_bytes = std::fs::read(setup.srs_path.as_path())
                .with_context(|| format!("reading zkey {}", setup.srs_path.as_path().display()))?;
            Zkey::parse(&zkey_bytes).context("parsing zkey")?
        };
        let groups = CoefficientGroups::from_zkey(&zkey);
        zkey.coefficients = Vec::new();
        let witness_bytes =
            unsafe { std::slice::from_raw_parts(prover.witness, zkey.num_vars * 32) };
        let witness = parse_witness_values(witness_bytes)
            .map_err(|i| anyhow!("witness value {i} is not a field element"))?;
        let (r, s) = (random_scalar()?, random_scalar()?);
        let proof = risc0_groth16_cuda::prove_grouped(&zkey, &groups, &witness, &r, &s)
            .context("cuda-oxide prover")?;
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
