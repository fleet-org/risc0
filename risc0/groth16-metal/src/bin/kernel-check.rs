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

//! `groth16-metal-kernel-check`: every kernel, the MSM and a fixture proof
//! against the Rust bodies — on the system default Metal device (macOS), or
//! on the CPU through the shaders compiled as C++ (everywhere else). Exit
//! status 0 only when every check agrees. Run this first on a Mac.

use risc0_groth16_metal::device::Backend as _;

fn main() -> anyhow::Result<()> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let prover = risc0_groth16_metal::device::MetalProver::new()?;
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let prover = risc0_groth16_metal::device::HostMslProver::new()?;
    println!("{}: shaders compiled", prover.backend().describe());
    let checks = prover.kernel_check()?;
    let failed = checks.iter().filter(|c| !c.ok).count();
    for c in &checks {
        println!(
            "{:<24} {}{}",
            c.kernel,
            if c.ok { "ok" } else { "MISMATCH" },
            if c.detail.is_empty() {
                String::new()
            } else {
                format!("  ({})", c.detail)
            }
        );
    }
    if failed > 0 {
        anyhow::bail!("{failed} check(s) disagree with the Rust bodies");
    }
    println!(
        "all {} checks agree with risc0_groth16_oxide::kernels and the core prover",
        checks.len()
    );
    Ok(())
}
