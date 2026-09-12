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

//! The CPU reference backend: `risc0-groth16-core`'s pipeline behind the
//! boundary, consuming the exact inputs the canonical kernels consume (the
//! zkey at `srs_path`, the witness pointer) and writing the exact outputs
//! (`proof.json`, `public.json`).
//!
//! It does not read `preprocessed_coeffs.bin` or `fuzzed_msm_results.bin`:
//! both are derivable from the zkey and only speed up the canonical kernels.
//! Wall-clock is not a goal here — correctness and shared code are.

use anyhow::{anyhow, Context as _};
use risc0_groth16_core::{
    field::Fr,
    prover::{proof_json, prove, public_json},
    zkey::{parse_witness_values, Zkey},
};

use super::{BackendKind, Groth16Backend};
use crate::{ProverParams, SetupParams};

/// The CPU reference prover.
pub struct Reference;

/// A uniformly random scalar: 31 random bytes, as the canonical kernels draw
/// `r` and `s` (`randombytes_buf(&r, sizeof(fr_t) - 1)`), so the value is
/// below the modulus by construction.
pub(crate) fn random_scalar() -> anyhow::Result<Fr> {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes[..31]).map_err(|e| anyhow!("randomness unavailable: {e}"))?;
    Fr::from_le_bytes(&bytes).ok_or_else(|| anyhow!("31 random bytes exceeded the modulus"))
}

impl Groth16Backend for Reference {
    fn kind(&self) -> BackendKind {
        BackendKind::Reference
    }

    fn prove(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()> {
        // Parse, then drop the file bytes: the zkey is multi-GB and the proof holds memory long.
        let zkey = {
            let zkey_bytes = std::fs::read(setup.srs_path.as_path())
                .with_context(|| format!("reading zkey {}", setup.srs_path.as_path().display()))?;
            Zkey::parse(&zkey_bytes).context("parsing zkey")?
        };
        // SAFETY: the boundary contract (BOUNDARY.md §4.1) is that `witness` points at
        // `num_vars` consecutive 32-byte canonical field elements, `num_vars` being the
        // zkey header's value — exactly what the canonical kernels dereference.
        let witness_bytes =
            unsafe { std::slice::from_raw_parts(prover.witness, zkey.num_vars * 32) };
        let witness = parse_witness_values(witness_bytes)
            .map_err(|i| anyhow!("witness value {i} is not a field element"))?;
        let (r, s) = (random_scalar()?, random_scalar()?);
        let proof = prove(&zkey, &witness, &r, &s).context("reference prover")?;
        std::fs::write(
            prover.public_path.as_path(),
            public_json(&witness, zkey.num_public),
        )
        .context("writing public.json")?;
        std::fs::write(prover.proof_path.as_path(), proof_json(&proof))
            .context("writing proof.json")?;
        Ok(())
    }
}

// The Reference kind is registered only under `reference`; other arms compile this module for
// `random_scalar` alone, so its tests must not run there.
#[cfg(all(test, feature = "reference"))]
mod tests {
    use super::*;
    use crate::{backend::fixture, ProverParams, Registry, SetupParams};

    #[test]
    fn reference_backend_output_verifies_under_the_upstream_verifier() {
        fixture::prove_through_boundary("reference", &fixture::witness_bytes())
            .expect("the reference proof must verify");
    }

    #[test]
    fn an_unsatisfied_witness_yields_a_proof_the_verifier_rejects() {
        let mut bytes = fixture::witness_bytes();
        bytes[32] = 34; // c = 34 while a·b = 33
        let err =
            fixture::prove_through_boundary("reference", &bytes).expect_err("must not verify");
        assert!(err.to_string().contains("Invalid proof"), "{err:#}");
    }

    #[test]
    fn a_non_field_witness_value_is_rejected_before_proving() {
        let mut bytes = fixture::witness_bytes();
        bytes[32..64].copy_from_slice(&[0xff; 32]);
        let err =
            fixture::prove_through_boundary("reference", &bytes).expect_err("must be rejected");
        assert!(err.to_string().contains("witness value 1"), "{err:#}");
    }

    #[test]
    fn a_missing_zkey_is_a_distinct_error() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = fixture::witness_bytes();
        let setup = SetupParams::new(dir.path()).unwrap();
        let prover = ProverParams::new(dir.path(), bytes.as_ptr()).unwrap();
        let err = Registry::new()
            .with(Box::new(Reference))
            .prove(Some("reference"), &prover, &setup)
            .expect_err("no zkey");
        assert!(err.to_string().contains("reading zkey"), "{err:#}");
    }
}
