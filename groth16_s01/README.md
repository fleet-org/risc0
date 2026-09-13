# GROTH16 s01 — milestone workspace in the `fleet-org/risc0` fork

[Milestone](https://github.com/fleet-org/risc0/milestone/1) · entry issue
[#1@fleet-org/risc0](https://github.com/fleet-org/risc0/issues/1) · plan:
`TODO/features/groth16-gpu-rewrite-s01.md` in `fleet-org/fleet`.

Base branch `integration/staging-platform` = upstream tag `v3.0.4` (`d7ee368e`), chosen because
bento's lockfile pins `risc0-groth16 3.0.3` + `risc0-groth16-sys 0.1.0` — the crates at that tag
(DEF-G16-001).

| file                                       | issue      | what                                                                                         |
| ------------------------------------------ | ---------- | -------------------------------------------------------------------------------------------- |
| [`CUDA_OXIDE_PIN.md`](./CUDA_OXIDE_PIN.md) | s01/1 · #2 | what cuda-oxide is, six answers + verdict                                                    |
| [`BOUNDARY.md`](./BOUNDARY.md)             | s01/2 · #3 | call graph, kernel inventory, the named boundary (`risc0_groth16_sys::prove`), data contract |
| [`CORPUS.md`](./CORPUS.md)                 | s01/3 · #4 | capture recipe, coverage axes, manifest schema, freeze rules                                 |

| [`HARNESS.md`](./HARNESS.md) | s01/5 · #7 | the differential harness: oracle, assertions, mutation
| [`WORKAROUNDS.md`](./WORKAROUNDS.md) | all | the register of workarounds, stubs and shortcuts for
expert review (W-01…) | arms, measured results (`harness/`, `reports/`) | |
[`../handoffs/cs-groth16-risc0-groth16-snark-m1.handback.md`](../handoffs/cs-groth16-risc0-groth16-snark-m1.handback.md)
| handback | what landed where, what remains and who holds the key, what the artifacts do not show |

Code map:

| crate                      | role                                                                                                                                                                                                                                           | proven by                                                                                                                                                                                        |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `risc0/groth16-sys`        | the boundary (`Groth16Backend`, `Registry`, `RISC0_GROTH16_BACKEND`) + backends: `canonical` (cuda), `reference`, `oxide-cpu`, `metal-cpu` (testing kinds)                                                                                     | unit tests with mutation arms; `reference`/`oxide-cpu` outputs verified by the upstream verifier · `backend/resident.rs`: one prepared zkey per arm per process, keyed by the file (DEF-G16-014) |
| `risc0/groth16-core`       | shared `no_std` arithmetic (Fp/Fr/Fp2, G1/G2), NTT, MSM, zkey/wtns formats, the prover pipeline and assembly, the arms' shared helpers                                                                                                         | conformance vs arkworks; fixture proof verifies                                                                                                                                                  |
| `risc0/groth16-oxide`      | the CUDA arm's kernel bodies as plain Rust + a launcher seam (`CpuLauncher` proves them; the cuda-oxide `#[cuda_module]` launcher is the device one)                                                                                           | byte-identical proof to core for fixed blinding; real-circuit run through the harness as `oxide-cpu`                                                                                             |
| `risc0/groth16-metal`      | the Metal arm: MSL kernels mirroring the oxide bodies, one host pipeline over a `Backend` — the Metal device on macOS, the shaders compiled as C++ and run on the CPU everywhere else (`msl-host/`); `groth16-metal-kernel-check` runs on both | shaders verified on the CPU down to a fixture proof (C14); the Mac confirms the Metal compiler and device                                                                                        |
| `risc0/groth16-cuda`       | the CUDA arm's host side: loads the cuda-oxide device module via `cuda-core`, runs the pipeline on device buffers, `groth16-cuda-kernel-check`; type-checks without a GPU (CUDA headers only)                                                  | `BackendKind::CudaOxide` under feature `cuda-oxide`; device half pending E2                                                                                                                      |
| `groth16_s01/cuda-kernels` | the device half: `#[kernel]` wrappers around the oxide bodies to the shared ABI (`risc0_groth16_oxide::abi`); built by `cargo oxide` on a CUDA host only (excluded from the workspace)                                                         | template — two UNVERIFIED points named in its `lib.rs`                                                                                                                                           |
| `groth16_s01/scripts`      | `precommit.sh` (the commit gate: hooks, CI, humans) · `cuda-headers.sh` (rootless CUDA 13 headers for GPU-less type-checks)                                                                                                                    |                                                                                                                                                                                                  |
| `groth16_s01/harness`      | s01/5                                                                                                                                                                                                                                          | run on the real circuit (see HARNESS.md)                                                                                                                                                         |

Every issue reference is repo-qualified; every source link is a full-SHA permalink.

**Status:** C6 — `risc0-groth16-oxide` (CUDA arm kernel bodies, CPU-proven) and
`risc0-groth16-metal` (Metal arm, darwin type-checked) exist; C5 — the s01/5 harness ran the
reference arm on the real circuit; C4 — `risc0/groth16-core` (shared `no_std` arithmetic, NTT, MSM,
zkey/wtns formats, reference prover; every module conformance-tested against arkworks) and the
`reference` backend in `risc0-groth16-sys`, verified end-to-end on the in-tree fixture. C3 — the
boundary module `risc0/groth16-sys/src/backend.rs` (`Groth16Backend`, `BackendKind`, `Registry`,
`RISC0_GROTH16_BACKEND`) with the canonical backend registered under `cuda`; CI in
`.github/workflows/groth16-s01.yml`.
