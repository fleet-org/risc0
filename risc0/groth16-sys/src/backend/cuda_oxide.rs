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
//! boundary — the zkey is parsed and uploaded ONCE per process and kept on
//! device 0 (`resident`; `RISC0_GROTH16_RESIDENT=0` for per-call uploads),
//! the witness is read exactly as the reference reads it, the proof is
//! produced by `risc0-groth16-cuda` from the module named by
//! `RISC0_GROTH16_CUDA_MODULE`, and the JSON files are written where the
//! canonical path writes them.

use anyhow::{anyhow, Context as _};
use risc0_groth16_core::{
    prover::{proof_json, public_json, CoefficientGroups},
    zkey::{parse_witness_values, Zkey},
};

use risc0_groth16_cuda::{CudaProver, ModuleSource, ResidentZkey};

use super::{reference::random_scalar, resident, BackendKind, Groth16Backend};
use crate::{ProverParams, SetupParams};

/// The CUDA arm.
pub struct CudaOxide;

impl Groth16Backend for CudaOxide {
    fn kind(&self) -> BackendKind {
        BackendKind::CudaOxide
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
            let device = CudaProver::new(ModuleSource::from_env()?)?;
            let resident = device.prepare(&zkey, &groups)?;
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
            .context("cuda-oxide prover")?;
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

/// The device, its module, and the zkey resident on it.
struct Prepared {
    device: CudaProver,
    resident: ResidentZkey,
}

static CACHE: resident::Cache<Prepared> = resident::Cache::new();
