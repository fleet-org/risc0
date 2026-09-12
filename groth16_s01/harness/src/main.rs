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

//! `groth16-s01-harness synth <case-dir> [iterations]`
//!     prove the in-tree loop guest to a succinct receipt and write a synthetic case.
//! `groth16-s01-harness run <case-dir> <artifacts-dir> <rewrite-kind> [--control <kind>] [--other <case-dir>] [--report <path>]`
//!     derive the boundary input, run the rewrite (and the control), judge with the
//!     upstream verifier, run the mutation arms, write the report (markdown + json).

use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};
use groth16_s01_harness::{
    case::Case,
    derive, mutations, oracle,
    report::{markdown, Row},
    run, synth, BackendKind, Tri,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("synth") => {
            let dir = PathBuf::from(args.get(1).context("synth <case-dir> [iterations]")?);
            let iterations: u32 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(0);
            let t = std::time::Instant::now();
            let receipt = synth::succinct_receipt(iterations)?;
            let secs = t.elapsed().as_secs_f64();
            Case::save_synthetic(
                &dir,
                &receipt,
                &format!("loop guest, {iterations} iterations, proven on the harness host in {secs:.1} s"),
            )?;
            println!("synthetic case written to {} ({secs:.1} s)", dir.display());
            Ok(())
        }
        Some("run") => run_cmd(&args[1..]),
        _ => bail!("usage: synth <case-dir> [iterations] | run <case-dir> <artifacts-dir> <rewrite-kind> [--control <kind>] [--other <case-dir>] [--report <path>]"),
    }
}

fn run_cmd(args: &[String]) -> Result<()> {
    let case_dir = PathBuf::from(args.first().context("run <case-dir> …")?);
    let artifacts = PathBuf::from(args.get(1).context("… <artifacts-dir> …")?);
    let rewrite = BackendKind::parse(args.get(2).context("… <rewrite-kind>")?)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut control = BackendKind::Canonical;
    let mut other: Option<PathBuf> = None;
    let mut report: Option<PathBuf> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--control" => {
                control = BackendKind::parse(args.get(i + 1).context("--control <kind>")?)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                i += 2;
            }
            "--other" => {
                other = Some(PathBuf::from(
                    args.get(i + 1).context("--other <case-dir>")?,
                ));
                i += 2;
            }
            "--report" => {
                report = Some(PathBuf::from(args.get(i + 1).context("--report <path>")?));
                i += 2;
            }
            a => bail!("unknown argument {a}"),
        }
    }

    let case = Case::load(&case_dir)?;
    let other_claim = other
        .as_deref()
        .map(Case::load)
        .transpose()?
        .map(|c| c.input.claim())
        .transpose()?;
    let row = run_case(&case, &artifacts, rewrite, control, other_claim.as_ref())?;
    let md = markdown(std::slice::from_ref(&row));
    println!("{md}");
    if let Some(path) = report {
        std::fs::write(&path, &md)?;
        std::fs::write(
            path.with_extension("json"),
            serde_json::to_string_pretty(&row)?,
        )?;
        println!("report written to {}", path.display());
    }
    if !row.passed() {
        bail!("case {} did not pass", row.case_id);
    }
    Ok(())
}

fn run_case(
    case: &Case,
    artifacts: &Path,
    rewrite: BackendKind,
    control: BackendKind,
    other_claim: Option<&risc0_zkvm::MaybePruned<risc0_zkvm::ReceiptClaim>>,
) -> Result<Row> {
    println!(
        "case {}: deriving the boundary input (identity_p254 + witness generation)…",
        case.id
    );
    let derived = derive::derive(&case.input, artifacts)?;
    println!(
        "  identity_p254 {:.1} s, witness generation {:.1} s, {} witness values",
        derived.identity_seconds, derived.witness_seconds, derived.num_vars
    );
    let claim = derived.ident.claim.clone();

    // Assertion 3 — the control: the canonical output from the corpus still verifies.
    let canonical_verified = match &case.canonical_output {
        None => {
            Tri::NotRun("the case carries no canonical output (synthetic or not captured)".into())
        }
        Some(receipt) => match receipt.inner.groth16() {
            Err(e) => Tri::No(format!("canonical output is not a groth16 receipt: {e:?}")),
            Ok(g) => match oracle::verify(&g.seal, &g.claim) {
                Ok(()) => Tri::Yes,
                Err(e) => Tri::No(format!("canonical output rejected by the oracle: {e}")),
            },
        },
    };

    println!("  running the rewrite `{rewrite}` behind the boundary…");
    let work = tempfile::tempdir()?;
    let res = run::run(rewrite, &derived.witness, artifacts, work.path())?;
    println!("  `{}` answered in {:.1} s", res.kind, res.seconds);

    // Assertion 1.
    let rewrite_verified = match oracle::verify(&res.seal, &claim) {
        Ok(()) => Tri::Yes,
        Err(e) => Tri::No(format!("oracle rejected the rewrite's seal: {e}")),
    };

    // Assertion 2 — same public inputs: the claim the canonical output verifies against
    // must be the claim the rewrite's seal verifies against.
    let public_inputs_equal = match &case.canonical_output {
        None => Tri::NotRun("no canonical output to compare claims with".into()),
        Some(receipt) => match receipt.inner.groth16() {
            Err(e) => Tri::No(format!("{e:?}")),
            Ok(g) => {
                if oracle::claim_digest(&g.claim) == oracle::claim_digest(&claim) {
                    Tri::Yes
                } else {
                    Tri::No("claim digests differ: the rewrite proved a different statement".into())
                }
            }
        },
    };

    let arms = mutations::Arms {
        bit_flip: mutations::bit_flip(&res.seal, &claim),
        cross_claim: mutations::cross_claim(&res.seal, &claim, other_claim),
        canonical_answers: mutations::canonical_answers(
            control,
            rewrite,
            &derived.witness,
            artifacts,
            &claim,
        ),
        malformed_input: mutations::malformed_input(&case.input_bytes),
    };

    Ok(Row {
        case_id: case.id.clone(),
        rewrite_kind: res.kind.to_string(),
        canonical_verified,
        rewrite_verified,
        public_inputs_equal,
        arms,
        derive_seconds: derived.identity_seconds + derived.witness_seconds,
        prove_seconds: res.seconds,
        oracle: oracle::name(),
    })
}
