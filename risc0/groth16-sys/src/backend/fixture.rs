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

//! Test-only: the in-tree circom fixture through the boundary for ANY kind,
//! judged by the upstream verifier — the same three properties for every
//! backend (verifies; an unsatisfied witness is rejected by the verifier; a
//! non-field witness value is rejected before proving).

use risc0_groth16::{ProofJson, PublicInputsJson, Verifier, VerifyingKeyJson};
use risc0_groth16_core::{
    prover::verifying_key_json,
    zkey::{parse_wtns, Zkey},
};

use crate::{ProverParams, SetupParams};

const ZKEY: &[u8] =
    include_bytes!("../../../../groth16_proof/circom-compat/test/data/multiplier2_final.zkey");
const WTNS: &[u8] =
    include_bytes!("../../../../groth16_proof/circom-compat/test/data/multiplier2.wtns");

/// The fixture witness as the boundary receives it (32-byte little-endian values).
pub fn witness_bytes() -> Vec<u8> {
    parse_wtns(WTNS)
        .unwrap()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// Run `kind` through the boundary on the fixture and return the upstream
/// verifier's verdict on what it wrote.
pub fn prove_through_boundary(kind: &str, witness_bytes: &[u8]) -> anyhow::Result<()> {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("stark_verify_final.zkey"), ZKEY).unwrap();
    let setup = SetupParams::new(dir.path()).unwrap();
    let prover = ProverParams::new(dir.path(), witness_bytes.as_ptr()).unwrap();
    crate::backend::compiled_in().prove(Some(kind), &prover, &setup)?;
    let proof: ProofJson =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("proof.json")).unwrap())
            .unwrap();
    let public = PublicInputsJson {
        values: serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("public.json")).unwrap(),
        )
        .unwrap(),
    };
    let zkey = Zkey::parse(ZKEY).unwrap();
    let vk: VerifyingKeyJson = serde_json::from_str(&verifying_key_json(&zkey)).unwrap();
    Verifier::from_json(proof, public, vk)?.verify()
}

/// The three properties, for `kind`.
pub fn three_properties(kind: &str) {
    prove_through_boundary(kind, &witness_bytes()).expect("the proof must verify");
    let mut bytes = witness_bytes();
    bytes[32] = 34; // c = 34 while a·b = 33
    let err = prove_through_boundary(kind, &bytes).expect_err("must not verify");
    assert!(err.to_string().contains("Invalid proof"), "{err:#}");
    let mut bytes = witness_bytes();
    bytes[32..64].copy_from_slice(&[0xff; 32]);
    let err = prove_through_boundary(kind, &bytes).expect_err("must be rejected");
    assert!(err.to_string().contains("witness value 1"), "{err:#}");
}
