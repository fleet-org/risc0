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

//! `groth16-cuda-kernel-check`: load the device module named by
//! `RISC0_GROTH16_CUDA_MODULE` (or embedded in this executable), run every
//! kernel of the ABI on a small input, and compare with the Rust bodies.
//! Exit status 0 only when every kernel agrees. Run this first on a CUDA host.

use risc0_groth16_cuda::{CudaProver, ModuleSource};

fn main() -> anyhow::Result<()> {
    let source = ModuleSource::from_env()?;
    let prover = CudaProver::new(source)?;
    println!(
        "device: {} · module: {}",
        prover.device_name(),
        prover.module_source()
    );
    let checks = prover.kernel_check()?;
    let bad = checks.iter().filter(|c| !c.ok).count();
    for c in &checks {
        println!(
            "{:<18} {}{}",
            c.kernel,
            if c.ok { "ok" } else { "DIFFERS" },
            if c.detail.is_empty() {
                String::new()
            } else {
                format!(" — {}", c.detail)
            }
        );
    }
    if bad > 0 {
        anyhow::bail!(
            "{bad} of {} kernels differ from risc0_groth16_oxide::kernels",
            checks.len()
        );
    }
    println!(
        "all {} kernels agree with risc0_groth16_oxide::kernels",
        checks.len()
    );
    Ok(())
}
