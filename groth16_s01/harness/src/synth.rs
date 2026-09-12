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

//! A synthetic stage input: a succinct receipt for the in-tree `loop` guest,
//! proven on this host. It exercises the real `stark_verify` circuit end to
//! end without production access; it is not a corpus case (no canonical
//! output, no production provenance) and the report says so.

use std::sync::LazyLock;

use anyhow::{Context as _, Result};
use risc0_binfmt::ProgramBinary;
use risc0_zkos_v1compat::V1COMPAT_ELF;
use risc0_zkvm::{get_prover_server, ExecutorEnv, ProverOpts, Receipt};

/// The `loop` guest from `risc0/zkvm/examples`, wrapped like the datasheet does.
static LOOP_ELF: LazyLock<Vec<u8>> = LazyLock::new(|| {
    const LOOP_ELF: &[u8] = include_bytes!("../../../risc0/zkvm/examples/loop.bin");
    ProgramBinary::new(LOOP_ELF, V1COMPAT_ELF).encode()
});

/// Prove the loop guest for `iterations` iterations and compress to a
/// succinct receipt — the form bento's SNARK task receives.
pub fn succinct_receipt(iterations: u32) -> Result<Receipt> {
    let env = ExecutorEnv::builder()
        .write_slice(&iterations.to_le_bytes())
        .build()
        .context("executor env")?;
    let prover = get_prover_server(&ProverOpts::succinct()).context("prover server")?;
    let info = prover
        .prove(env, &LOOP_ELF)
        .context("proving the loop guest")?;
    anyhow::ensure!(
        info.receipt.inner.succinct().is_ok(),
        "expected a succinct receipt from ProverOpts::succinct()"
    );
    Ok(info.receipt)
}
