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

//! `groth16-metal-kernel-check`: compile the shaders on this Mac and compare
//! every kernel against the Rust bodies on small inputs. Run this first.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> anyhow::Result<()> {
    let prover = risc0_groth16_metal::device::MetalProver::new()?;
    println!("{prover:?}: shaders compiled");
    let mut failed = 0;
    for c in prover.kernel_check()? {
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
        if !c.ok {
            failed += 1;
        }
    }
    if failed > 0 {
        anyhow::bail!("{failed} kernel(s) disagree with the Rust bodies");
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("groth16-metal-kernel-check runs only on macOS / Apple Silicon");
    std::process::exit(2);
}
