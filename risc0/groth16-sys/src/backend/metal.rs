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
    prover::{proof_json, public_json, CoefficientGroups},
    zkey::{parse_witness_values, Zkey},
};
use risc0_groth16_metal::device::{MetalBackend, MetalProver, ResidentZkey};

use super::{reference::random_scalar, resident, BackendKind, Groth16Backend};
use crate::{ProverParams, SetupParams};

/// The Metal arm.
pub struct Metal;

impl Groth16Backend for Metal {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn prove(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()> {
        let path = setup.srs_path.as_path();
        let prepare = || -> anyhow::Result<Prepared> {
            let mut stamp = resident::Stamp::new();
            let mut zkey = {
                let zkey_bytes = std::fs::read(path)
                    .with_context(|| format!("reading zkey {}", path.display()))?;
                stamp.mark("zkey read");
                Zkey::parse(&zkey_bytes).context("parsing zkey")?
            };
            stamp.mark("zkey parse");
            let groups = CoefficientGroups::from_zkey(&zkey);
            zkey.coefficients = Vec::new();
            stamp.mark("coefficient groups");
            let device = MetalProver::new()?;
            let resident = device.prepare_owned(zkey, groups);
            stamp.mark("prepare (upload, resident)");
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
        let mut stamp = resident::Stamp::new();
        let proof = p
            .device
            .prove_resident(&p.resident, &witness, &r, &s)
            .context("metal prover")?;
        stamp.mark("prove (witness in, proof out)");
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

/// The device, its pipelines, and the zkey resident on it.
struct Prepared {
    device: MetalProver,
    resident: ResidentZkey<MetalBackend>,
}

static CACHE: resident::Cache<Prepared> = resident::Cache::new();
