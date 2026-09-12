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

//! The oracle: the canonical upstream verifier, unmodified. A seal is judged
//! by assembling the `Groth16Receipt` bento's SNARK task would hold and
//! calling the same `verify_integrity_with_context` it calls
//! (`[BENTO-SNARK-005]`), with the default verifier context — control root,
//! BN254 identity control id and verifying key all upstream constants.

use risc0_zkvm::{
    sha::{Digest, Digestible as _},
    Groth16Receipt, Groth16ReceiptVerifierParameters, MaybePruned, ReceiptClaim, VerifierContext,
};

/// The oracle's identity for the report.
pub fn name() -> String {
    format!(
        "risc0-groth16 {} `Verifier` via risc0-zkvm {} `Groth16Receipt::verify_integrity_with_context`",
        risc0_groth16_version(),
        risc0_zkvm::VERSION
    )
}

fn risc0_groth16_version() -> &'static str {
    // The crate does not export its version; the workspace pins it (Cargo.lock: risc0-groth16 3.0.3
    // at the base tag). Reported from the build-time environment of this crate's dependency graph.
    "3.0.3 (workspace at the milestone base tag v3.0.4)"
}

/// Build the receipt the stage would emit for `seal` under `claim`.
pub fn receipt(seal: &[u8], claim: &MaybePruned<ReceiptClaim>) -> Groth16Receipt<ReceiptClaim> {
    Groth16Receipt::new(
        seal.to_vec(),
        claim.clone(),
        Groth16ReceiptVerifierParameters::default().digest(),
    )
}

/// Verify `seal` against `claim` with the upstream verifier.
pub fn verify(seal: &[u8], claim: &MaybePruned<ReceiptClaim>) -> Result<(), String> {
    receipt(seal, claim)
        .verify_integrity_with_context(&VerifierContext::default())
        .map_err(|e| format!("{e:?}"))
}

/// The claim digest — the value the five public inputs are derived from.
pub fn claim_digest(claim: &MaybePruned<ReceiptClaim>) -> Digest {
    claim.digest()
}
