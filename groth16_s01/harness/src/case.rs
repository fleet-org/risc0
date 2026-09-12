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

//! A corpus case on disk, in the layout `groth16_s01/CORPUS.md` §2 produces:
//! `input.stark.bincode` (the stage input, a succinct `Receipt`), optionally
//! `canonical.groth16.bincode` (the canonical stage output), and `task.json`.
//! A synthetic case (built by `synth`) has the same layout with
//! `"synthetic": true` and no canonical output.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use risc0_zkvm::Receipt;

/// A loaded case.
#[derive(Debug)]
pub struct Case {
    /// Directory name, the case id.
    pub id: String,
    /// The directory.
    pub dir: PathBuf,
    /// The stage input, as bento stored it.
    pub input: Receipt,
    /// The raw bytes of the stage input (for the malformed-input arm).
    pub input_bytes: Vec<u8>,
    /// The canonical stage output, when the corpus carries one.
    pub canonical_output: Option<Receipt>,
    /// Where `canonical_output` came from: `corpus` (production) or
    /// `rehearsal:<kind>` (produced locally by that backend to exercise the
    /// canonical code paths before the corpus exists). Reported verbatim.
    pub canonical_provenance: Option<String>,
}

impl Case {
    /// Load a case directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let id = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "case".into());
        let input_bytes = std::fs::read(dir.join("input.stark.bincode"))
            .with_context(|| format!("{}: input.stark.bincode", dir.display()))?;
        let input: Receipt = bincode::deserialize(&input_bytes).context("input receipt")?;
        let mut canonical_output = None;
        let mut canonical_provenance = None;
        for f in [
            "canonical.groth16.bincode",
            "canonical.blake3_groth16.bincode",
        ] {
            let p = dir.join(f);
            if p.exists() {
                let b = std::fs::read(&p)?;
                canonical_output =
                    Some(bincode::deserialize(&b).with_context(|| format!("{}", p.display()))?);
                canonical_provenance = Some("corpus".to_string());
                break;
            }
        }
        if canonical_output.is_none() {
            let rehearsal = std::fs::read_dir(dir)?
                .flatten()
                .map(|e| e.path())
                .find(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("rehearsal.") && n.ends_with(".bincode"))
                });
            if let Some(p) = rehearsal {
                let kind = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .trim_start_matches("rehearsal.")
                    .trim_end_matches(".bincode")
                    .to_string();
                let b = std::fs::read(&p)?;
                canonical_output =
                    Some(bincode::deserialize(&b).with_context(|| format!("{}", p.display()))?);
                canonical_provenance = Some(format!("rehearsal:{kind}"));
            }
        }
        Ok(Self {
            id,
            dir: dir.to_path_buf(),
            input,
            input_bytes,
            canonical_output,
            canonical_provenance,
        })
    }

    /// Store a locally produced stage output as a REHEARSAL canonical for this case
    /// (`rehearsal.<kind>.bincode`). It exercises the canonical code paths; the report
    /// labels it, so it is never mistaken for a production output.
    pub fn save_rehearsal(dir: &Path, kind: &str, output: &Receipt) -> Result<()> {
        std::fs::write(
            dir.join(format!("rehearsal.{kind}.bincode")),
            bincode::serialize(output)?,
        )?;
        Ok(())
    }

    /// Write a synthetic case (no canonical output) to `dir`.
    pub fn save_synthetic(dir: &Path, input: &Receipt, note: &str) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("input.stark.bincode"), bincode::serialize(input)?)?;
        let task = serde_json::json!({
            "synthetic": true,
            "compress_type": "Groth16",
            "note": note,
            "canonical_wall_clock_s": null,
            "notes": "synthetic case: no canonical output; canonical_wall_clock_s null because the case was never run by the production stage",
        });
        std::fs::write(dir.join("task.json"), serde_json::to_string_pretty(&task)?)?;
        Ok(())
    }
}
