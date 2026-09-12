---
kind: handback
from: cs-groth16-risc0-groth16-snark-m1 (owner session, fleet-org/risc0 milestone 1)
to: root-CP, and any successor session on GROTH16 s01 (arm sessions on a CUDA host or a Mac)
refs:
  - https://github.com/fleet-org/risc0/milestone/1
  - https://github.com/fleet-org/risc0/issues/1 (entry; checkpoints C1–C6 as comments)
  - https://github.com/fleet-org/risc0/pull/9 (docs) · /pull/10 (boundary) · /pull/11 (core +
    reference) · /pull/13 (harness + arms) · /pull/14 (rehearsal + kernel-check)
---

# GROTH16 s01 — handback after the first arc (2026-09-12)

## What this session owned and what landed

| checkpoint | what                                                                                                                                                                                                                                                                          | where                                                                     |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| C1         | goal opened; plan corrections reported; base branch cut at tag `v3.0.4` (DEF-G16-001)                                                                                                                                                                                         | comment on #1; `integration/staging-platform`                             |
| C2         | s01/1 cuda-oxide pin, s01/2 boundary map, s01/3 corpus recipe                                                                                                                                                                                                                 | `groth16_s01/{CUDA_OXIDE_PIN,BOUNDARY,CORPUS}.md`; comments on #2, #3, #4 |
| C3         | the boundary in code: `Groth16Backend`, `Registry`, `RISC0_GROTH16_BACKEND`, canonical backend under `cuda`                                                                                                                                                                   | `risc0/groth16-sys/src/backend.rs`; CI job `groth16-s01`                  |
| C4         | `risc0-groth16-core` (shared `no_std` arithmetic, NTT, MSM, zkey/wtns, prover) + `reference` backend                                                                                                                                                                          | verified by arkworks and the in-tree fixture                              |
| C5         | s01/5 harness, run on the real `stark_verify` circuit with the reference arm                                                                                                                                                                                                  | `groth16_s01/harness`, `HARNESS.md`, `reports/`                           |
| C6         | CUDA arm kernel bodies (`risc0-groth16-oxide`, proven on the CPU launcher on the real circuit as `oxide-cpu`), Metal arm (`risc0-groth16-metal`, darwin type-checked), production artifact readers                                                                            | same PR; comments on #5, #6                                               |
| C7         | harness rehearsal mode (assertions 2/3 exercised and shown to fail), Metal per-kernel check binary, the commit gate `groth16_s01/scripts/precommit.sh`                                                                                                                        | PRs #14, #15                                                              |
| C15        | the first hardware run: the session recycled onto an RTX 5080; driver user-space library fetched rootless; `groth16-cuda-kernel-check` 13/13 for both modules; the production circuit through the GPU in the harness                                                          | comments on #1/#5; PR #23                                                 |
| C14        | the Metal shaders run on the CPU: the prover generic over a `Backend`; `HostBackend` compiles the shader source as C++ (build script + stub `<metal_stdlib>`) and runs it per index; every kernel, both MSMs and a fixture proof verified on Linux                            | PR #22                                                                    |
| C13        | the CUDA device module compiled to PTX without a GPU (nightly type-check + clippy; cargo-oxide backend + llc; ptxas for sm_89 and sm_120; kernels and parameter layouts checked against the ABI); a `raw_eq` limitation fixed in core; PTX published as the `ptx-c13` release | PR #21                                                                    |
| C12        | the MSM over every window at once (`digits_all` + one flat `bucket_sum` launch; host sort laid out flat) in the CPU pipeline, the CUDA host and the Metal host; CPU-verified by the byte-identical-proof test                                                                 | PR #20                                                                    |
| C11        | resident zkey for both arms: `prepare` once (points, grouped coefficients, NTT tables on the device), `prove_resident` per call; process-wide cache in the backends keyed by the zkey file; `RISC0_GROTH16_RESIDENT=0` opts out                                               | PR #19                                                                    |
| C8         | CUDA arm host side (`risc0-groth16-cuda`, `cuda-oxide` backend), the shared ABI and the coset launch schedule as data (which also fixed a ping-pong bug in the never-run Metal transform), the device-module template, GPU-less type-checks in CI                             | PR #16                                                                    |

Every claim in those documents is marked MEASURED / INFERRED / UNVERIFIED; every source link is a
full-SHA permalink.

## What remains, and who holds the key

| item                                                 | blocked on                                                            | what unblocks it                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| s01/3 corpus (assertions 2 and 3 on production data) | production access (E1)                                                | run `CORPUS.md` §2 where the agent's `DATABASE_URL`/`S3_*` are set; freeze as a GitHub Release on this fork                                                                                                                                                                                                                                                                                        |
| s01/4 device half of the CUDA arm                    | a CUDA 13 host with the nightly + `cargo oxide` (E2)                  | `groth16_s01/cuda-kernels/README.md` in order: `cargo oxide build --arch <sm>` (product `risc0_groth16_cuda_kernels.ptx`), run `groth16-cuda-kernel-check` (names the first kernel that differs from the Rust bodies), `cargo test -p risc0-groth16-sys --features cuda-oxide`, then `harness run … cuda-oxide --control canonical`; the host side, ABI and schedule are in place and type-checked |
| s01/4b Metal execution                               | a macOS 13+ Apple Silicon host (E3)                                   | `cargo run -p risc0-groth16-metal --bin groth16-metal-kernel-check` first (names the first kernel that differs from the Rust bodies), then `cargo test -p risc0-groth16-sys --features metal`, then `cargo run -p groth16-s01-harness --features metal -- run <case> metal --control reference`; the arm is selectable behind the boundary since C9                                                |
| s01/6 on-chain e2e + timing                          | the deployment (and fleet-org/fleet#1950 or production authorization) | out of any CRCS's reach; needs root-CP                                                                                                                                                                                                                                                                                                                                                             |

## What the artifacts do not show (learned the hard way)

- **The fork's `main` is not the target.** bento pins `risc0-groth16 3.0.3` from crates.io; the
  editable surface is `risc0-groth16-sys` at tag `v3.0.4`, reached from bento by
  `[patch.crates-io]`. Anyone working from `main` (5.0.0, no `bento/`) is on a different product.
- **Both circuits share the boundary.** `Groth16` and `Blake3Groth16` differ only in the artifact
  directory handed to `risc0_groth16_sys::prove`; production uses the blake3 variant (INFERRED from
  the fleet's template env). The blake3 artifacts (2.4 GB) were not exercised here.
- **Encodings:** zkey points Montgomery LE; coefficient values `v·R²`; witness canonical. A port
  that gets any of the three wrong produces a proof that fails to verify with no other symptom.
- **CRCS toolchain:** no C compiler, no `protoc`, no libclang, no xz in the container; the
  relocations that worked are recorded in the catalog draft `groth16-s01-fork-baseline`
  (I-ASM-006..008); the session's `buildenv.sh` in its home directory is the sourced form. The
  memory guard counts page cache: keep `evict-cache.py` running beside big reads or builds.
- **Every commit passes `groth16_s01/scripts/precommit.sh`** (privacy, links, format, license,
  clippy, message shape); install it once with `--install`. CI runs the same script on the PR range.
- **Orchestration written blind must be data with one executor that is tested.** The Metal coset
  transform chose its own ping-pong buffers and read the pre-scale buffer into the forward NTT;
  nothing could catch it without a Mac. The launch order is now `risc0_groth16_oxide::schedule`, run
  by the CPU pipeline under the byte-identical-proof test, so the Metal and CUDA provers only follow
  it.
- **A claim does not encode work**: two runs of the same guest with the same journal are the same
  statement; the cross-claim arm needs a different image id or journal, or its labeled derived-claim
  fallback.
- **The `canonical answers` arm needs a control that differs from the rewrite and is compiled in**;
  on a CUDA host use `--control canonical`, elsewhere `--control reference`.

## How to resume

The session container's ledger holds `next_action_on_resume`; this file and the milestone comments
hold everything a successor needs without it. Start from `groth16_s01/README.md` (code map), then
`BOUNDARY.md` §3.1, then run
`cargo test -p risc0-groth16-core -p risc0-groth16-oxide -p risc0-groth16-sys --features risc0-groth16-sys/reference,risc0-groth16-sys/oxide-cpu`
to confirm the contract before touching an arm.
