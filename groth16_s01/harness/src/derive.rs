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

//! From the stage input to the boundary input, exactly as the canonical path
//! does it (`risc0-groth16` `prove/cuda.rs`): `identity_p254` on the succinct
//! receipt, the seal to its JSON form, and circom witness generation with the
//! `stark_verify_graph.bin` of the artifact set. Everything above the boundary
//! is upstream code; the harness only sequences it.

use std::{path::Path, time::Instant};

use anyhow::{anyhow, Context as _, Result};
use risc0_groth16_core::zkey::parse_wtns;
use risc0_zkvm::{get_prover_server, ProverOpts, Receipt, ReceiptClaim, SuccinctReceipt};

/// The boundary input for one case.
pub struct Derived {
    /// The identity_p254 receipt (its claim is what the seal must verify against).
    pub ident: SuccinctReceipt<ReceiptClaim>,
    /// The identity_p254 seal bytes.
    pub seal_bytes: Vec<u8>,
    /// The circom inputs JSON (`{"iop": [...]}`).
    pub inputs_json: String,
    /// The witness as the boundary receives it: `num_vars` × 32 canonical LE bytes.
    pub witness: Vec<u8>,
    /// Number of witness values.
    pub num_vars: usize,
    /// Wall-clock of identity_p254 on this host.
    pub identity_seconds: f64,
    /// Wall-clock of witness generation on this host.
    pub witness_seconds: f64,
}

/// Derive the boundary input from a succinct stage input.
pub fn derive(input: &Receipt, artifacts: &Path) -> Result<Derived> {
    let succinct = input
        .inner
        .succinct()
        .map_err(|e| anyhow!("stage input is not a succinct receipt: {e}"))?;
    let t = Instant::now();
    let prover = get_prover_server(&ProverOpts::default()).context("prover server")?;
    let ident = prover.identity_p254(succinct).context("identity_p254")?;
    let identity_seconds = t.elapsed().as_secs_f64();
    let seal_bytes = ident.get_seal_bytes();
    let inputs_json = risc0_groth16::prove::to_json(&seal_bytes).context("seal to json")?;

    let t = Instant::now();
    let graph = std::fs::read(artifacts.join("stark_verify_graph.bin"))
        .with_context(|| format!("{}/stark_verify_graph.bin", artifacts.display()))?;
    let wtns = circom_witnesscalc::calc_witness(&inputs_json, &graph)
        .map_err(|e| anyhow!("witness generation: {e}"))?;
    let values = parse_wtns(&wtns).map_err(|e| anyhow!("witness file: {e}"))?;
    let witness_seconds = t.elapsed().as_secs_f64();
    let witness: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(Derived {
        ident,
        seal_bytes,
        inputs_json,
        num_vars: values.len(),
        witness,
        identity_seconds,
        witness_seconds,
    })
}
