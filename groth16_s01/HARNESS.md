# GROTH16 s01/5 — the differential correctness harness

**Issue:** [#7@fleet-org/risc0](https://github.com/fleet-org/risc0/issues/7) · **Crate:**
`groth16_s01/harness` (`groth16-s01-harness`) · **Oracle:** the canonical upstream verifier,
unmodified — `risc0_zkvm::Groth16Receipt::verify_integrity_with_context` with the default
`VerifierContext` (the call bento's SNARK task makes on its own output, `[BENTO-SNARK-005]`), which
wraps `risc0_groth16::Verifier`.

## The correctness argument, as the harness checks it

Groth16 proofs are randomized, so nothing is compared byte for byte. Per corpus case and per arm the
harness establishes, three-state (`yes` / `no` with evidence / `not run` with reason — never
collapsing "could not look" into "nothing there"):

| #   | assertion                                                          | how                                                                                                                                                                                   |
| --- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | the rewrite's seal verifies                                        | `oracle::verify(seal, claim)` where `claim` is the identity_p254 receipt's claim derived from the stage input                                                                         |
| 2   | it verifies against the same public inputs as the canonical output | the canonical output's claim digest equals the derived claim's digest (the five public inputs are a function of the claim digest, the control root and the BN254 identity control id) |
| 3   | the canonical output still verifies in the same run (control)      | `oracle::verify` on the corpus's `canonical.*.bincode` seal and claim                                                                                                                 |
| 4   | a corrupted seal is rejected (mutation)                            | one bit flipped in each of π_A, π_B, π_C; all three variants must be rejected                                                                                                         |

Mutation arms beyond assertion 4 (`mutations.rs`):

| arm               | what must happen                                                                    | how the harness knows it is not fooling itself                                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| cross-claim       | the rewrite's seal verified against another case's claim is rejected                | skipped with a reason when the other case has the same claim digest                                                                                            |
| canonical answers | the rewrite is disabled, the control implementation answers, the suite still passes | the registry's _confirmed_ kind is recorded (`RunResult::kind`), and a registry without the rewrite must refuse to select it — a substitution cannot be silent |
| malformed input   | the truncated stage input is rejected                                               | rejection happens before the boundary (deserialization), so both implementations fail identically by construction; the error class is recorded                 |

## Running it

```
# a synthetic stage input (the in-tree loop guest proven on this host; not a corpus case)
groth16-s01-harness synth cases/synthetic-loop-0 0

# one case, rewrite = <kind>, control = canonical (default) or --control reference
groth16-s01-harness run cases/<case> <artifacts-dir> <kind> [--control <kind>] [--other cases/<other>] [--report reports/<case>.md]

# before the corpus exists: store a LABELED stand-in canonical output, produced locally by <kind>
groth16-s01-harness rehearse cases/<case> <artifacts-dir> <kind>
```

`<artifacts-dir>` holds the rzup `risc0-groth16` v0.1.0 set (`stark_verify_final.zkey`,
`stark_verify_graph.bin`, `preprocessed_coeffs.bin`, `fuzzed_msm_results.bin`). The harness derives
the boundary input exactly as the canonical path does (`identity_p254` → seal → `to_json` → circom
witness), runs the selected backend through `risc0_groth16_sys::Registry`, and judges with the
oracle. Build with `--features cuda` on a CUDA host to make the canonical backend available.

## Results

_Filled in per run; each row is MEASURED on the host named._

### Synthetic cases, reference arm, CRCS host (no GPU) — MEASURED 2026-09-12

Host: an unprivileged CRCS container, 24 CPU cores, 30 GB RAM shared with the host, no CUDA
userland, no macOS. Artifacts: rzup `risc0-groth16` v0.1.0 (`stark_verify_final.zkey` 3.45 GiB,
num_vars 5,635,930, num_public 5, domain 2^23; SHA-256 of the archive matches the signed
distribution manifest). Versions: risc0-zkvm 3.0.4, risc0-groth16 3.0.3, risc0-groth16-sys 0.1.0
(the base tag). Stage inputs: the in-tree `loop` guest proven to a succinct receipt on this host (0
iterations: 11.9 s; 100,000 iterations: 32.2 s). **These are synthetic stage inputs, not corpus
cases**: no canonical output exists for them, so assertions 2 and 3 are `not run` by construction,
and the `canonical answers` arm is `not run` because the canonical CUDA backend is not compiled into
this host's harness.

| case                | rewrite   | canonical verifies (3)                                                     | rewrite verifies (1) | same public inputs (2)                               | bit-flip rejected (4)                                              | cross-claim rejected                                                                                                                                | canonical answers                                                                      | malformed input                                                                   | derive s | prove s |
| ------------------- | --------- | -------------------------------------------------------------------------- | -------------------- | ---------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------: | ------: |
| synthetic-loop-0    | reference | not run (the case carries no canonical output (synthetic or not captured)) | yes                  | not run (no canonical output to compare claims with) | killed (one bit flipped in each of pi_a, pi_b, pi_c: all rejected) | killed (rejected against the other case has the same claim digest; used the own digest with bit 0 flipped: verification indicates proof is invalid) | not run (control `canonical` is not compiled into this harness (available: reference)) | killed (rejected before the boundary (bincode): io error: unexpected end of file) |     26.3 |   113.9 |
| synthetic-loop-100k | reference | not run (the case carries no canonical output (synthetic or not captured)) | yes                  | not run (no canonical output to compare claims with) | killed (one bit flipped in each of pi_a, pi_b, pi_c: all rejected) | killed (rejected against no second case; used the own digest with bit 0 flipped: verification indicates proof is invalid)                           | not run (control `canonical` is not compiled into this harness (available: reference)) | killed (rejected before the boundary (bincode): io error: unexpected end of file) |     26.5 |   113.3 |

Timings are this host's wall-clock: `derive s` = identity_p254 on the CPU (≈23.9 s) + circom witness
generation (≈2.5 s); `prove s` = the CPU reference behind the boundary (≈113 s: zkey parse with
on-curve checks, scatter of 29.1M coefficients, three coset transforms on 2^23, five Pippenger MSMs
on 22 threads, assembly, JSON). They are reported for information, not as performance claims.

**What the two cases establish.** On the real circuit, the reference arm's proof verifies under the
unmodified upstream verifier (assertion 1) for two distinct stage inputs; corrupting any of π_A,
π_B, π_C is rejected (assertion 4); verifying against a foreign statement is rejected (cross-claim
arm — derived from the case's own claim digest, because both loop-guest runs are the same statement:
the claim does not encode work); a truncated stage input is rejected before the boundary with its
error class (malformed-input arm). **What they do not establish:** agreement with a production
canonical output (needs the corpus, E1) and the `canonical answers` control (needs the CUDA backend,
E2). Both cells say so.

Oracle: risc0-groth16 3.0.3 (workspace at the milestone base tag v3.0.4) `Verifier` via risc0-zkvm
3.0.4 `Groth16Receipt::verify_integrity_with_context`.

### Synthetic cases, CUDA-arm kernel bodies on the host launcher (`oxide-cpu`), control = `reference` — MEASURED 2026-09-12

Same host, artifacts and stage inputs as above. The rewrite is `risc0-groth16-oxide`'s kernel bodies
— the code cuda-oxide compiles to PTX — run by the CPU launcher through the boundary
(`BackendKind::OxideCpu`, a testing kind); the control is the reference. Peak memory 7.7 GB
anonymous (11.4 GB with page cache) per run.

| case                | rewrite   | canonical verifies (3)                                                     | rewrite verifies (1) | same public inputs (2)                               | bit-flip rejected (4)                                              | cross-claim rejected                                                                                                      | canonical answers                                                                                                            | malformed input                                                                   | derive s | prove s |
| ------------------- | --------- | -------------------------------------------------------------------------- | -------------------- | ---------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------: | ------: |
| synthetic-loop-0    | oxide-cpu | not run (the case carries no canonical output (synthetic or not captured)) | yes                  | not run (no canonical output to compare claims with) | killed (one bit flipped in each of pi_a, pi_b, pi_c: all rejected) | killed (rejected against no second case; used the own digest with bit 0 flipped: verification indicates proof is invalid) | killed (control `reference` answered and verified; registry reported `reference`; a registry without `oxide-cpu` refuses it) | killed (rejected before the boundary (bincode): io error: unexpected end of file) |     26.4 |    29.5 |
| synthetic-loop-100k | oxide-cpu | not run (the case carries no canonical output (synthetic or not captured)) | yes                  | not run (no canonical output to compare claims with) | killed (one bit flipped in each of pi_a, pi_b, pi_c: all rejected) | killed (rejected against no second case; used the own digest with bit 0 flipped: verification indicates proof is invalid) | killed (control `reference` answered and verified; registry reported `reference`; a registry without `oxide-cpu` refuses it) | killed (rejected before the boundary (bincode): io error: unexpected end of file) |     26.3 |    29.4 |

**What these two runs add.** The `canonical answers` arm now runs: with the rewrite disabled, the
control (`reference`) answers, the oracle accepts it, the registry reports the control kind, and a
registry without `oxide-cpu` refuses to select it — the harness cannot report a rewrite result that
the control produced. The CUDA arm's kernels are therefore proven on the production circuit before
any GPU executes them; on a CUDA host the only change is the launcher. Prove time is this host's
wall-clock for information only.

### Rehearsal of the canonical-output path (assertions 2 and 3) — MEASURED 2026-09-12

No corpus case exists yet, so the canonical code paths had never executed.
`groth16-s01-harness rehearse <case> <artifacts> reference` stores the reference's stage output as a
**labeled** stand-in (`rehearsal.reference.bincode`); the report marks the cell
`[rehearsal:reference]` so it can never be read as a production output. Rewrite `oxide-cpu`, control
`reference`:

| case             | rewrite   | canonical verifies (3)    | rewrite verifies (1) | same public inputs (2) | bit-flip rejected (4)                                              | cross-claim rejected                                                                                                      | canonical answers                                                                                                            | malformed input                                                                   | derive s | prove s |
| ---------------- | --------- | ------------------------- | -------------------- | ---------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------: | ------: |
| synthetic-loop-0 | oxide-cpu | yes [rehearsal:reference] | yes                  | yes                    | killed (one bit flipped in each of pi_a, pi_b, pi_c: all rejected) | killed (rejected against no second case; used the own digest with bit 0 flipped: verification indicates proof is invalid) | killed (control `reference` answered and verified; registry reported `reference`; a registry without `oxide-cpu` refuses it) | killed (rejected before the boundary (bincode): io error: unexpected end of file) |     26.4 |    29.6 |

Mutation of the control arm — one byte of the stand-in flipped — must turn assertion 3 into `no` and
fail the case:

| case                  | rewrite   | canonical verifies (3)                                                                         | rewrite verifies (1) | same public inputs (2) | bit-flip rejected (4)                                              | cross-claim rejected                                                                                                      | canonical answers                                                                                                            | malformed input                                                                   | derive s | prove s |
| --------------------- | --------- | ---------------------------------------------------------------------------------------------- | -------------------- | ---------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------: | ------: |
| mut-corrupt-canonical | oxide-cpu | **no** (canonical output rejected by the oracle: invalid receipt format) [rehearsal:reference] | yes                  | yes                    | killed (one bit flipped in each of pi_a, pi_b, pi_c: all rejected) | killed (rejected against no second case; used the own digest with bit 0 flipped: verification indicates proof is invalid) | killed (control `reference` answered and verified; registry reported `reference`; a registry without `oxide-cpu` refuses it) | killed (rejected before the boundary (bincode): io error: unexpected end of file) |     26.3 |    29.4 |

The harness exited non-zero (`case mut-corrupt-canonical did not pass`). When the corpus arrives,
the same two columns fill from `canonical.groth16.bincode` with provenance `corpus`, through code
that has already been shown to accept a valid output and reject a corrupted one.

### Frozen corpus (`corpus-groth16-s01-v1`)

_pending — needs the corpus (E1 on #1)._

## What this harness does not cover

Seal encoding for the chain, selector routing through the verifier router, and SetVerifier Merkle
inclusion — those are s01/6's on-chain oracle. Timing here is the harness host's wall-clock,
reported for information, not as a performance claim.

### The production circuit through the CUDA arm on hardware (`cuda-oxide`, RTX 5080), control = `reference` — MEASURED 2026-09-12

The session container was recycled onto a box with an NVIDIA GeForce RTX 5080 (`sm_120`, driver
580.95.05). The arm ran from the `ptx-c13` release's PTX, unchanged since it was built and validated
without a GPU (C13); the driver's user-space library was fetched rootless at the module's version.

| case                | rewrite    | rewrite verifies (1)                                                     | bit-flip | cross-claim | canonical answers | malformed input | derive s | prove s |
| ------------------- | ---------- | ------------------------------------------------------------------------ | -------- | ----------- | ----------------- | --------------- | -------: | ------: |
| synthetic-loop-100k | cuda-oxide | **yes**                                                                  | killed   | killed      | killed            | killed          |     26.1 |   125.7 |
| synthetic-loop-0    | cuda-oxide | not run: `DriverError(2, "out of memory")` uploading a 256 MB polynomial |          |             |                   |                 |     25.7 |       — |

Assertions 2 and 3 remain _not run_ (no canonical output: E1). Full rows in
[`reports/synthetic-loop-100k.cuda-oxide.md`](./reports/synthetic-loop-100k.cuda-oxide.md).

**What the numbers mean.** The arm's first production-scale proof on a device verifies under the
upstream verifier with every mutation arm killed. 125.7 s is an unoptimised arm — serial bucket
loops per thread, the host's counting sort of 22 × 5.6M digits per MSM, 493 MB read back per MSM —
against 113 s for the reference on the CPU (C5) and 29.5 s for the CPU launcher (C6); correctness
came first, and the profile of where the time goes is the next hardware question.

**The out-of-memory is the environment's, not the arm's.** The device is shared: three
`cuMemGetInfo` samples two seconds apart read 11.4, 3.2 and 11.4 GiB free of 15.5, so another tenant
peaks at ≈ 12 GiB. The arm's working set (≈ 4.7 GB resident, ≈ 3 GB per-proof scratch) plus that
peak exceeds the card; the first run met the peak, the second did not. GPU runs from this session
stopped there, pending a quiet window or a dedicated device (asked of root-CP), because on the
tenant's peak either side fails — and the tenant may be production.

**Also through the boundary, on the fixture:**
`cargo test -p risc0-groth16-sys --features cuda-oxide` runs the reference backend's three
properties for `cuda-oxide` on the device (verifies; an unsatisfied witness is rejected by the
verifier; a non-field witness value is rejected before proving), and `--features metal-cpu` the same
for the Metal shaders on the CPU — one shared fixture helper, every kind.

### Device etiquette on a shared GPU

Before a production-sized run, `groth16_s01/scripts/gpu-free.py 5 2` samples the driver's free
memory five times two seconds apart and exits non-zero when the minimum is below 10 GiB (or a
`GPU_FREE_MIN_GIB` of your choosing); a swing above 1 GiB between samples means another tenant is
allocating. On this box, minutes after the passing run, three samples read 2.5–3.2 GiB free: the
other tenant held ≈ 13 GB. A run started then would have failed, and might have failed the tenant.
`RISC0_GROTH16_TIMING=1` makes the CUDA prover print per-phase wall-clock (uploads, scatter, the
three coset transforms, each MSM's host sort / bucket sums / reduction, assembly) to stderr, so a
granted window yields a profile and not only a total.

### The production circuit through the Metal shaders on the CPU (`metal-cpu`), control = `reference` — MEASURED 2026-09-12

The Metal arm has no device in this session; its shaders, compiled as C++ (C14), prove the
production circuit on the CPU through the same boundary and the same harness:

| case                | rewrite   | rewrite verifies (1) | bit-flip | cross-claim | canonical answers   | malformed input | derive s | prove s |
| ------------------- | --------- | -------------------- | -------- | ----------- | ------------------- | --------------- | -------: | ------: |
| synthetic-loop-100k | metal-cpu | **yes**              | killed   | killed      | not run (see below) | killed          |     26.2 |   424.5 |

Full row in
[`reports/synthetic-loop-100k.metal-cpu.md`](./reports/synthetic-loop-100k.metal-cpu.md). 424.5 s is
the shader source run one thread index at a time on one core; it is a correctness run, not a timing.
What a Mac adds is the Metal compiler and the device: the arithmetic, the layouts and the whole
pipeline down to a verifying production proof are established here.

The `canonical answers` arm reports _not run_ because this harness binary was built with the
canonical CUDA kernels compiled in (`--features cuda-canonical`) and the arm then insists on
`canonical` as the control rather than the `reference` given — an arm-selection rule to revisit when
the canonical control runs on the GPU. The two earlier runs (control `reference`, canonical not
compiled in) killed it.

The first attempt was killed by the host's low-memory watchdog at a 20.4 GB peak: the parsed zkey
was alive while its packed copies were made, and the resident set stayed while the control proved.
`prepare_owned` (both arms) now consumes the zkey field by field, and `RISC0_GROTH16_RESIDENT=0`
drops the resident set after a one-off proof; the rerun peaked at 12.8 GB (5-second samples beside
the run) on a 30 GB host shared with a production tenant.
