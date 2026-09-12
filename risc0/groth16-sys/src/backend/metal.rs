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

//! `BackendKind::Metal`: the Metal arm (GROTH16 s01/4b) behind the boundary
//! — the zkey and witness are read exactly as the reference does, the proof
//! is produced by `risc0-groth16-metal` on the system's default Metal device,
//! and the JSON files are written where the canonical path writes them.
//! Compiled only for macOS on Apple Silicon with the `metal` feature; on any
//! other build the kind is simply unavailable (never a silent fallback).

use anyhow::{anyhow, Context as _};
use risc0_groth16_core::{
    prover::{proof_json, public_json},
    zkey::{parse_witness_values, Zkey},
};

use super::{reference::random_scalar, BackendKind, Groth16Backend};
use crate::{ProverParams, SetupParams};

/// The Metal arm.
pub struct Metal;

impl Groth16Backend for Metal {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn prove(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()> {
        let zkey = {
            let zkey_bytes = std::fs::read(setup.srs_path.as_path())
                .with_context(|| format!("reading zkey {}", setup.srs_path.as_path().display()))?;
            Zkey::parse(&zkey_bytes).context("parsing zkey")?
        };
        let witness_bytes =
            unsafe { std::slice::from_raw_parts(prover.witness, zkey.num_vars * 32) };
        let witness = parse_witness_values(witness_bytes)
            .map_err(|i| anyhow!("witness value {i} is not a field element"))?;
        let (r, s) = (random_scalar()?, random_scalar()?);
        let proof =
            risc0_groth16_metal::device::prove(&zkey, &witness, &r, &s).context("metal prover")?;
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
