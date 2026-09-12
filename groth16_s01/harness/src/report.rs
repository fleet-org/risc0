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

//! The per-case report row of fleet-org/risc0#7 and its rendering.

use serde::{Deserialize, Serialize};

use crate::{mutations::Arms, Tri};

/// One corpus case, one arm.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Row {
    /// Case id.
    pub case_id: String,
    /// Which implementation answered as the rewrite (as selected and confirmed by the registry).
    pub rewrite_kind: String,
    /// Assertion 3: the canonical output still verifies.
    pub canonical_verified: Tri,
    /// Assertion 1: the rewrite's output verifies.
    pub rewrite_verified: Tri,
    /// Assertion 2: the same public inputs (same claim digest) as the canonical output.
    pub public_inputs_equal: Tri,
    /// The mutation arms.
    pub arms: Arms,
    /// Seconds to derive the boundary input (identity_p254 + witness generation) on this host.
    pub derive_seconds: f64,
    /// Seconds the rewrite spent behind the boundary on this host.
    pub prove_seconds: f64,
    /// Verifier crate and version used as the oracle.
    pub oracle: String,
}

impl Row {
    /// Whether every assertion that ran passed and every arm that ran was killed.
    pub fn passed(&self) -> bool {
        self.rewrite_verified.passed()
            && !matches!(self.canonical_verified, Tri::No(_))
            && !matches!(self.public_inputs_equal, Tri::No(_))
            && self.arms.all_killed_or_not_run()
    }
}

/// Render rows as the markdown table the issue asks for.
pub fn markdown(rows: &[Row]) -> String {
    let mut out = String::new();
    out.push_str("| case | rewrite | canonical verifies (3) | rewrite verifies (1) | same public inputs (2) | bit-flip rejected (4) | cross-claim rejected | canonical answers | malformed input | derive s | prove s |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|---:|---:|\n");
    for r in rows {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {:.1} |\n",
            r.case_id,
            r.rewrite_kind,
            r.canonical_verified.cell(),
            r.rewrite_verified.cell(),
            r.public_inputs_equal.cell(),
            r.arms.bit_flip.cell(),
            r.arms.cross_claim.cell(),
            r.arms.canonical_answers.cell(),
            r.arms.malformed_input.cell(),
            r.derive_seconds,
            r.prove_seconds,
        ));
    }
    let passed = rows.iter().filter(|r| r.passed()).count();
    out.push_str(&format!(
        "\n**{passed}/{} cases pass** (a case passes when every assertion that ran holds and every arm that ran was killed; `not run` cells are reported, not counted as passes). Oracle: {}.\n",
        rows.len(),
        rows.first().map(|r| r.oracle.as_str()).unwrap_or("-")
    ));
    out
}
