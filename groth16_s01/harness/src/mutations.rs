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

//! The mutation arms of fleet-org/risc0#7. Each arm feeds the oracle
//! something it must reject and records whether it did. An arm that cannot be
//! attempted says why; it never counts as killed.

use std::path::Path;

use risc0_zkvm::{MaybePruned, Receipt, ReceiptClaim};
use serde::{Deserialize, Serialize};

use crate::{oracle, run, Arm, BackendKind};

/// Results of the four arms for one case.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Arms {
    /// A byte of the rewrite's seal is corrupted → must be rejected.
    pub bit_flip: Arm,
    /// The rewrite's seal is verified against another case's claim → must be rejected.
    pub cross_claim: Arm,
    /// The rewrite is disabled and the control implementation answers → the
    /// suite still passes, and the registry reports the control kind, not the rewrite.
    pub canonical_answers: Arm,
    /// The malformed input is fed → rejected before the boundary, with its error class.
    pub malformed_input: Arm,
}

impl Arms {
    /// No arm survived.
    pub fn all_killed_or_not_run(&self) -> bool {
        ![
            &self.bit_flip,
            &self.cross_claim,
            &self.canonical_answers,
            &self.malformed_input,
        ]
        .iter()
        .any(|a| matches!(a, Arm::Survived(_)))
    }
}

/// Arm: flip one bit in each of π_A, π_B and π_C; every variant must be rejected.
pub fn bit_flip(seal: &[u8], claim: &MaybePruned<ReceiptClaim>) -> Arm {
    if seal.len() != 256 {
        return Arm::NotRun(format!("seal is {} bytes, expected 256", seal.len()));
    }
    for (name, offset) in [("pi_a", 31usize), ("pi_b", 64 + 31), ("pi_c", 192 + 31)] {
        let mut corrupted = seal.to_vec();
        corrupted[offset] ^= 0x01;
        if oracle::verify(&corrupted, claim).is_ok() {
            return Arm::Survived(format!("a bit flipped in {name} still verified"));
        }
    }
    Arm::Killed("one bit flipped in each of pi_a, pi_b, pi_c: all rejected".into())
}

/// Arm: the seal against another statement's public inputs. Prefers another
/// case's claim; when none is given, or it is the same statement (two runs of
/// one guest with the same journal share a claim digest — the claim does not
/// encode work), falls back to a FOREIGN claim derived from the case's own
/// claim digest with one bit flipped, and says so in the result.
pub fn cross_claim(
    seal: &[u8],
    own: &MaybePruned<ReceiptClaim>,
    other: Option<&MaybePruned<ReceiptClaim>>,
) -> Arm {
    let own_digest = oracle::claim_digest(own);
    let (foreign, label): (MaybePruned<ReceiptClaim>, String) = match other {
        Some(other) if oracle::claim_digest(other) != own_digest => {
            (other.clone(), "another case's claim".into())
        }
        Some(_) => {
            let mut d = own_digest;
            d.as_mut_bytes()[0] ^= 0x01;
            (
                MaybePruned::Pruned(d),
                "the other case has the same claim digest; used the own digest with bit 0 flipped"
                    .into(),
            )
        }
        None => {
            let mut d = own_digest;
            d.as_mut_bytes()[0] ^= 0x01;
            (
                MaybePruned::Pruned(d),
                "no second case; used the own digest with bit 0 flipped".into(),
            )
        }
    };
    match oracle::verify(seal, &foreign) {
        Ok(()) => Arm::Survived(format!("verified against a foreign claim ({label})")),
        Err(e) => Arm::Killed(format!("rejected against {label}: {e}")),
    }
}

/// Arm: disable the rewrite and let the control implementation answer. The
/// suite must still pass with the control, and the registry must report the
/// control kind; selecting a kind this build lacks must be an error, not a
/// silent substitution.
pub fn canonical_answers(
    control: BackendKind,
    rewrite: BackendKind,
    witness: &[u8],
    artifacts: &Path,
    claim: &MaybePruned<ReceiptClaim>,
) -> Arm {
    if control == rewrite {
        return Arm::NotRun(format!("control and rewrite are the same kind ({control})"));
    }
    let available = run::available();
    if !available.contains(&control) {
        return Arm::NotRun(format!(
            "control `{control}` is not compiled into this harness (available: {})",
            available
                .iter()
                .map(|k| k.name())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    // The substitution check: a registry without the rewrite must refuse it.
    let mut without = risc0_groth16_sys::Registry::new();
    for kind in &available {
        if *kind != rewrite {
            without = match *kind {
                BackendKind::Reference => without.with(Box::new(reference_backend())),
                other => {
                    return Arm::NotRun(format!(
                        "cannot rebuild a registry with `{other}` in this harness"
                    ))
                }
            };
        }
    }
    if without.select(Some(rewrite.name())).is_ok() {
        return Arm::Survived(format!("a registry without `{rewrite}` still selected it"));
    }
    let work = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => return Arm::NotRun(format!("tempdir: {e}")),
    };
    match run::run(control, witness, artifacts, work.path()) {
        Err(e) => Arm::NotRun(format!("control run failed: {e:#}")),
        Ok(res) => {
            if res.kind != control {
                return Arm::Survived(format!(
                    "asked for `{control}`, registry ran `{}`",
                    res.kind
                ));
            }
            match oracle::verify(&res.seal, claim) {
                Ok(()) => Arm::Killed(format!(
                    "control `{control}` answered and verified; registry reported `{}`; a registry without `{rewrite}` refuses it",
                    res.kind
                )),
                Err(e) => Arm::Survived(format!("control `{control}` produced a seal the oracle rejects: {e}")),
            }
        }
    }
}

fn reference_backend() -> risc0_groth16_sys::backend::reference::Reference {
    risc0_groth16_sys::backend::reference::Reference
}

/// Arm: a truncated stage input. Rejection happens before the boundary, in
/// deserialization, so both implementations fail identically by construction;
/// the error class is what the report records.
pub fn malformed_input(input_bytes: &[u8]) -> Arm {
    let truncated = &input_bytes[..input_bytes.len() / 2];
    match bincode::deserialize::<Receipt>(truncated) {
        Ok(_) => Arm::Survived("a truncated receipt deserialized".into()),
        Err(e) => Arm::Killed(format!("rejected before the boundary (bincode): {e}")),
    }
}
