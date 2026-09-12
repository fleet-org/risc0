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

//! GROTH16 s01/5 — the differential correctness harness.
//!
//! The oracle is the **canonical upstream verifier, unmodified**: a
//! `Groth16Receipt` assembled from the seal under test and the input receipt's
//! claim, checked by `verify_integrity_with_context` — the same call bento's
//! SNARK task makes on its own output. Nothing here re-implements a check.
//!
//! Per corpus case and per arm, the harness records (three-state, never
//! collapsing "could not run" into "failed"):
//!
//! 1. the rewritten implementation's seal verifies (assertion 1);
//! 2. it verifies against the same claim — hence the same public inputs — as
//!    the canonical output (assertion 2);
//! 3. the canonical output still verifies in the same run (assertion 3, the
//!    control);
//! 4. a corrupted seal is rejected (assertion 4, the mutation);
//!
//! plus the mutation arms of fleet-org/risc0#7: cross-claim rejection, the
//! "canonical answers" control with the selected kind reported, and the
//! malformed-input rejection with its error class.

pub mod case;
pub mod derive;
pub mod mutations;
pub mod oracle;
pub mod report;
pub mod run;
pub mod synth;

pub use risc0_groth16_sys::BackendKind;

/// A fact the harness could establish, could not establish, or could not
/// attempt — kept distinct so a report never reads "could not look" as
/// "nothing there".
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Tri {
    /// Established true.
    Yes,
    /// Established false, with the evidence.
    No(String),
    /// Not attempted, with the reason.
    NotRun(String),
}

impl Tri {
    /// `Yes` counts as a pass; anything else does not.
    pub fn passed(&self) -> bool {
        matches!(self, Tri::Yes)
    }

    /// Short cell text for a table.
    pub fn cell(&self) -> String {
        match self {
            Tri::Yes => "yes".into(),
            Tri::No(why) => format!("**no** ({why})"),
            Tri::NotRun(why) => format!("not run ({why})"),
        }
    }
}

/// Outcome of a mutation arm: the harness killed the mutant (the oracle
/// rejected it), the mutant survived (a defect), or the arm could not run.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Arm {
    /// The corrupted or misdirected input was rejected, as required.
    Killed(String),
    /// It was accepted: the check cannot fail and is decoration.
    Survived(String),
    /// The arm could not be attempted.
    NotRun(String),
}

impl Arm {
    /// Short cell text for a table.
    pub fn cell(&self) -> String {
        match self {
            Arm::Killed(how) => format!("killed ({how})"),
            Arm::Survived(how) => format!("**SURVIVED** ({how})"),
            Arm::NotRun(why) => format!("not run ({why})"),
        }
    }
}
