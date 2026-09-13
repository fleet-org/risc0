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

//! `BackendKind::CudaOxide`: the CUDA arm (GROTH16 s01/4) behind the
//! boundary — the zkey is parsed and uploaded ONCE per process and kept on
//! device 0 (`resident`; `RISC0_GROTH16_RESIDENT=0` for per-call uploads),
//! the witness is read exactly as the reference reads it, the proof is
//! produced by `risc0-groth16-cuda` from the module named by
//! `RISC0_GROTH16_CUDA_MODULE`, and the JSON files are written where the
//! canonical path writes them.

use anyhow::{anyhow, Context as _};
use risc0_groth16_core::{
    prover::{proof_json, public_json, CoefficientGroups},
    zkey::{parse_witness_values, Zkey},
};

use risc0_groth16_cuda::{budget, CudaProver, ModuleSource, ResidentZkey};

use super::{reference::random_scalar, resident, BackendKind, Groth16Backend};
use crate::{ProverParams, SetupParams};

/// The CUDA arm.
pub struct CudaOxide;

impl Groth16Backend for CudaOxide {
    fn kind(&self) -> BackendKind {
        BackendKind::CudaOxide
    }

    fn prove(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()> {
        let path = setup.srs_path.as_path();
        let key = resident::Key::of(path)?;
        // The path is chosen by `RISC0_GROTH16_RESIDENT`: `0`/`false`/`off`
        // forces streaming, anything else forces resident, and UNSET is AUTO —
        // the memory budget (`budget::Budget`) against the device's free memory
        // decides, so the arm fits a shared or small device without a flag.
        match resident::choice() {
            resident::Choice::Force(false) => {
                let mut stamp = resident::Stamp::new();
                let (zkey, groups) = load_zkey(path, &mut stamp)?;
                let device = CudaProver::new(ModuleSource::from_env()?)?;
                run_streaming(prover, &device, &zkey, &groups, &mut stamp)
            }
            resident::Choice::Force(true) => {
                let p = CACHE.get_or_prepare(key, || prepare(path))?;
                run_resident(prover, &p)
            }
            resident::Choice::Auto => {
                // A resident zkey for this path already up: keep using it.
                if let Some(p) = CACHE.peek(&key) {
                    return run_resident(prover, &p);
                }
                // Probe once: parse the dimensions, read the device's free
                // memory, and let the budget choose — then run the chosen path
                // with what we already parsed (no second parse or upload).
                let mut stamp = resident::Stamp::new();
                let (zkey, groups) = load_zkey(path, &mut stamp)?;
                let device = CudaProver::new(ModuleSource::from_env()?)?;
                let budget = budget::Budget::new(budget::Dims::from_zkey(&zkey));
                // The MINIMUM over several readings, not one: on a shared device
                // the free line swings as the co-tenant allocates.
                let free = device.min_free_device_bytes(4);
                let mode = free
                    .and_then(|f| budget.choose(f, 0.9))
                    .unwrap_or(budget::Mode::Streaming);
                eprintln!(
                    "[groth16-backend] budget: resident peak {:.2} GiB, streaming peak {:.2} GiB; \
                     device free(min) {}; auto -> {:?}",
                    budget.resident_peak_bytes() as f64 / GIB,
                    budget.streaming_peak_bytes() as f64 / GIB,
                    free.map(|f| format!("{:.2} GiB", f as f64 / GIB))
                        .unwrap_or_else(|| "unknown".into()),
                    mode,
                );
                match mode {
                    budget::Mode::Streaming => {
                        run_streaming(prover, &device, &zkey, &groups, &mut stamp)
                    }
                    // A shared device's free reading can be stale between the
                    // probe and the upload; fall back to streaming on an
                    // out-of-memory (the bbstark P5.2 adaptive pattern), so AUTO
                    // still produces a proof when the co-tenant took the window.
                    budget::Mode::Resident => {
                        match try_resident(prover, device, zkey, groups, key, &mut stamp) {
                            Ok(()) => Ok(()),
                            Err(e) if is_oom(&e) => {
                                CACHE.evict();
                                eprintln!(
                                    "[groth16-backend] resident path out of memory ({e:#}); \
                                     falling back to streaming"
                                );
                                let mut stamp = resident::Stamp::new();
                                let (zkey, groups) = load_zkey(path, &mut stamp)?;
                                let device = CudaProver::new(ModuleSource::from_env()?)?;
                                run_streaming(prover, &device, &zkey, &groups, &mut stamp)
                            }
                            Err(e) => Err(e),
                        }
                    }
                }
            }
        }
    }
}

/// Upload the zkey, cache it, and prove resident — the resident attempt AUTO
/// makes before falling back to streaming on out-of-memory. Consumes `zkey`.
fn try_resident(
    prover: &ProverParams,
    device: CudaProver,
    zkey: Zkey,
    groups: CoefficientGroups,
    key: resident::Key,
    stamp: &mut resident::Stamp,
) -> anyhow::Result<()> {
    let resident = device.prepare_owned(zkey, groups)?;
    stamp.mark("prepare (upload, resident)");
    let p = CACHE.get_or_prepare(key, move || Ok(Prepared { device, resident }))?;
    run_resident(prover, &p)
}

/// Whether an error is a device out-of-memory (the shared-tenant race), so AUTO
/// falls back to streaming rather than failing the proof.
fn is_oom(e: &anyhow::Error) -> bool {
    let s = format!("{e:#}").to_ascii_lowercase();
    s.contains("out of memory") || s.contains("outofmemory") || s.contains("drivererror(2")
}

/// One GiB, for the budget log line.
const GIB: f64 = (1usize << 30) as f64;

/// Read and parse the zkey, take the coefficient groups, and drop the raw
/// coefficient vector — the work both paths share before they diverge.
fn load_zkey(
    path: &std::path::Path,
    stamp: &mut resident::Stamp,
) -> anyhow::Result<(Zkey, CoefficientGroups)> {
    let mut zkey = {
        let zkey_bytes =
            std::fs::read(path).with_context(|| format!("reading zkey {}", path.display()))?;
        stamp.mark("zkey read");
        Zkey::parse(&zkey_bytes).context("parsing zkey")?
    };
    stamp.mark("zkey parse");
    let groups = CoefficientGroups::from_zkey(&zkey);
    zkey.coefficients = Vec::new();
    stamp.mark("coefficient groups");
    Ok((zkey, groups))
}

/// Parse, upload once and cache — the resident path's `prepare` (used when the
/// operator forces resident; the AUTO path builds `Prepared` from its probe).
fn prepare(path: &std::path::Path) -> anyhow::Result<Prepared> {
    let mut stamp = resident::Stamp::new();
    let (zkey, groups) = load_zkey(path, &mut stamp)?;
    let device = CudaProver::new(ModuleSource::from_env()?)?;
    let resident = device.prepare_owned(zkey, groups)?;
    stamp.mark("prepare (upload, resident)");
    Ok(Prepared { device, resident })
}

/// The streaming proof: nothing resident, the parsed zkey on the host for the
/// call, every phase uploaded and freed on the device.
fn run_streaming(
    prover: &ProverParams,
    device: &CudaProver,
    zkey: &Zkey,
    groups: &CoefficientGroups,
    stamp: &mut resident::Stamp,
) -> anyhow::Result<()> {
    let witness_bytes = unsafe { std::slice::from_raw_parts(prover.witness, zkey.num_vars * 32) };
    let witness = parse_witness_values(witness_bytes)
        .map_err(|i| anyhow!("witness value {i} is not a field element"))?;
    let (r, s) = (random_scalar()?, random_scalar()?);
    let proof = device
        .prove_streaming(zkey, groups, &witness, &r, &s)
        .context("cuda-oxide prover (streaming)")?;
    stamp.mark("prove (streaming)");
    std::fs::write(
        prover.public_path.as_path(),
        public_json(&witness, zkey.num_public),
    )
    .context("writing public.json")?;
    std::fs::write(prover.proof_path.as_path(), proof_json(&proof))
        .context("writing proof.json")?;
    Ok(())
}

/// The resident proof: the witness in, the proof out, against the cached zkey.
fn run_resident(prover: &ProverParams, p: &Prepared) -> anyhow::Result<()> {
    let (num_vars, num_public) = (p.resident.num_vars(), p.resident.num_public());
    let witness_bytes = unsafe { std::slice::from_raw_parts(prover.witness, num_vars * 32) };
    let witness = parse_witness_values(witness_bytes)
        .map_err(|i| anyhow!("witness value {i} is not a field element"))?;
    let (r, s) = (random_scalar()?, random_scalar()?);
    let mut stamp = resident::Stamp::new();
    let proof = p
        .device
        .prove_resident(&p.resident, &witness, &r, &s)
        .context("cuda-oxide prover")?;
    stamp.mark("prove (witness in, proof out)");
    std::fs::write(
        prover.public_path.as_path(),
        public_json(&witness, num_public),
    )
    .context("writing public.json")?;
    std::fs::write(prover.proof_path.as_path(), proof_json(&proof))
        .context("writing proof.json")?;
    Ok(())
}

/// The device, its module, and the zkey resident on it.
struct Prepared {
    device: CudaProver,
    resident: ResidentZkey,
}

static CACHE: resident::Cache<Prepared> = resident::Cache::new();

#[cfg(test)]
mod tests {
    use crate::backend::fixture;

    /// Why the device test cannot run here, or `None` when it can.
    fn not_here() -> Option<String> {
        let env = risc0_groth16_cuda::ModuleSource::ENV;
        if std::env::var_os(env).is_none() {
            return Some(format!("{env} unset: no device module to load"));
        }
        if !std::path::Path::new("/dev/nvidiactl").exists() {
            return Some("no /dev/nvidiactl: no CUDA device in this environment".into());
        }
        None
    }

    /// The same three properties as the reference backend's, through the
    /// boundary on the GPU — runnable only where a device and the module
    /// exist (no CI runner has one); elsewhere it says why it did not run
    /// rather than pass vacuously.
    #[test]
    fn cuda_oxide_through_the_boundary_on_the_device() {
        if let Some(why) = not_here() {
            eprintln!("cuda-oxide device test NOT RUN: {why}");
            return;
        }
        fixture::three_properties("cuda-oxide");
    }
}
