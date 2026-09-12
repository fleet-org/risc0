# GROTH16 s01 — milestone workspace in the `fleet-org/risc0` fork

[Milestone](https://github.com/fleet-org/risc0/milestone/1) · entry issue [#1@fleet-org/risc0](https://github.com/fleet-org/risc0/issues/1) · plan: `TODO/features/groth16-gpu-rewrite-s01.md` in `fleet-org/fleet`.

Base branch `integration/staging-platform` = upstream tag `v3.0.4` (`d7ee368e`), chosen because bento's lockfile pins `risc0-groth16 3.0.3` + `risc0-groth16-sys 0.1.0` — the crates at that tag (DEF-G16-001).

| file | issue | what |
|---|---|---|
| [`CUDA_OXIDE_PIN.md`](./CUDA_OXIDE_PIN.md) | s01/1 · #2 | what cuda-oxide is, six answers + verdict |
| [`BOUNDARY.md`](./BOUNDARY.md) | s01/2 · #3 | call graph, kernel inventory, the named boundary (`risc0_groth16_sys::prove`), data contract |
| [`CORPUS.md`](./CORPUS.md) | s01/3 · #4 | capture recipe, coverage axes, manifest schema, freeze rules |

Code lands under `risc0/groth16-sys/` (boundary trait + backends) and, later, `groth16_s01/harness/` (s01/5). Every issue reference is repo-qualified; every source link is a full-SHA permalink.

**Status:** C4 — `risc0/groth16-core` (shared `no_std` arithmetic, NTT, MSM, zkey/wtns formats, reference prover; every module conformance-tested against arkworks) and the `reference` backend in `risc0-groth16-sys`, verified end-to-end on the in-tree fixture. C3 — the boundary module `risc0/groth16-sys/src/backend.rs` (`Groth16Backend`, `BackendKind`, `Registry`, `RISC0_GROTH16_BACKEND`) with the canonical backend registered under `cuda`; CI in `.github/workflows/groth16-s01.yml`.
