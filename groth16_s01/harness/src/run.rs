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

//! Run one implementation behind the boundary and collect its seal. The
//! registry's own selection is recorded and returned, so a report can never
//! attribute a result to an implementation that did not run.

use std::{path::Path, time::Instant};

use anyhow::{Context as _, Result};
use risc0_groth16::{ProofJson, Seal};
use risc0_groth16_sys::{backend::compiled_in, BackendKind, ProverParams, SetupParams};

/// What a run produced.
pub struct RunResult {
    /// The kind the registry selected and ran — confirmed, not assumed.
    pub kind: BackendKind,
    /// The 256-byte seal (`Seal::to_vec()`), the form `Groth16Receipt` carries.
    pub seal: Vec<u8>,
    /// `proof.json` as written by the backend.
    pub proof_json: String,
    /// `public.json` as written by the backend.
    pub public_json: String,
    /// Wall-clock behind the boundary.
    pub seconds: f64,
}

/// Which backends this build of the harness can run.
pub fn available() -> Vec<BackendKind> {
    compiled_in().available()
}

/// Run `kind` on `witness` with the artifact set at `artifacts`, working in `work`.
pub fn run(kind: BackendKind, witness: &[u8], artifacts: &Path, work: &Path) -> Result<RunResult> {
    let registry = compiled_in();
    let selected = registry
        .select(Some(kind.name()))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let confirmed = selected.kind();
    anyhow::ensure!(
        confirmed == kind,
        "registry selected {confirmed}, asked for {kind}"
    );
    let setup = SetupParams::new(artifacts).context("setup params")?;
    let prover = ProverParams::new(work, witness.as_ptr()).context("prover params")?;
    let t = Instant::now();
    selected
        .prove(&prover, &setup)
        .with_context(|| format!("backend {kind}"))?;
    let seconds = t.elapsed().as_secs_f64();
    let proof_json = std::fs::read_to_string(prover.proof_path.as_path()).context("proof.json")?;
    let public_json =
        std::fs::read_to_string(prover.public_path.as_path()).context("public.json")?;
    let proof: ProofJson = serde_json::from_str(&proof_json).context("parsing proof.json")?;
    let seal: Seal = proof.try_into().context("proof.json → Seal")?;
    Ok(RunResult {
        kind: confirmed,
        seal: seal.to_vec(),
        proof_json,
        public_json,
        seconds,
    })
}
