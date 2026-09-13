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

| case                | rewrite                | rewrite verifies (1) | bit-flip | cross-claim | canonical answers | malformed input | derive s | prove s |
| ------------------- | ---------------------- | -------------------- | -------- | ----------- | ----------------- | --------------- | -------: | ------: |
| synthetic-loop-100k | cuda-oxide             | **yes**              | killed   | killed      | killed            | killed          |     26.1 |   125.7 |
| synthetic-loop-0    | cuda-oxide (streaming) | **yes** (2026-09-13) | killed   | killed      | killed            | killed          |     25.9 |   149.4 |

Assertions 2 and 3 remain _not run_ (no canonical output: E1). Full rows in
[`reports/synthetic-loop-100k.cuda-oxide.md`](./reports/synthetic-loop-100k.cuda-oxide.md) and
[`reports/synthetic-loop-0.cuda-oxide.md`](./reports/synthetic-loop-0.cuda-oxide.md).

**The second case, through the streaming path (C18, C19) — MEASURED 2026-09-13.** The first attempt
at synthetic-loop-0 (2026-09-12) died uploading a 256 MB polynomial: the resident arm on the other
tenant's peak. With `RISC0_GROTH16_RESIDENT=0` the arm streams (nothing resident, ≈ 2 GB on the
device at peak) and proved the case beside the tenant's activity: verified under the upstream
verifier, every mutation arm killed. The 149.4 s are 8.6 s of zkey read, parse and grouping on the
host (paid per call on the streaming path, once per process on the resident one) and 139.7 s on the
device. The device's free memory, printed at every phase mark, moved between 1.15 and 11.2 GiB
during the run — the tenant cycles every 15–30 s, and the arm fit in what it left.

**The profile** (`RISC0_GROTH16_TIMING=1`, the streaming run): scatter 0.5 s; the three coset
transforms and the quotient 0.3 s; the five MSMs 139.0 s — h 28.8, a 16.3, b1 16.1, b2 (G2) 61.1, c
16.5 — of which the host's counting sorts are 0.2–0.9 s each and the reductions 0.05–0.2 s. The
bucket-sum launch is 99 % of the proof: one thread per (window, bucket) walking its bucket's points
serially, 22 × 4,095 threads for 5.6 M points, most buckets long and the tail buckets idle. That is
W-14 in the register, and the next change to the arm.

### The MSM's critical path, found and bounded (C21) — MEASURED 2026-09-13

The static reading of the kernel (3,951 lines of PTX for `bucket_sum_g1`, 144 wide multiplies, 118
registers, modest local traffic) predicted about 1 G additions/s and could not explain a launch at 8
M/s. One microbenchmark could: `groth16-cuda-msm-bench` runs the bucket-sum kernel unchanged over a
range layout of the caller's choosing (the kernel's contract is only "`starts[b]..starts[b+1]` is my
run of `order`"), on 2^20 uniform scalars and 2^16 distinct G1 points tiled to 2^20, all 22 windows
— 22.5 M additions per launch in every row:

| layout (same kernel, same additions)                            |    ranges | per range | M additions/s (3 launches) |
| --------------------------------------------------------------- | --------: | --------: | -------------------------: |
| one thread per (window, bucket) — the arm's layout before C21   |    90,090 |       250 |                  4.2 – 5.0 |
| the same, gathering from a 4.7 MB point set that stays in cache |    90,090 |       250 |                  4.2 – 5.3 |
| pieces of at most 1024 points                                   |    90,601 |       249 |                  410 – 870 |
| pieces of at most 256                                           |   133,709 |       169 |                860 – 1,100 |
| pieces of at most 128                                           |   221,751 |       102 |                740 – 1,060 |
| pieces of at most 64                                            |   397,839 |        57 |              1,800 – 1,810 |
| pieces of at most 32                                            |   750,103 |        30 |              1,420 – 1,780 |
| pieces of at most 8                                             | 2,859,176 |         8 |              1,050 – 2,270 |

**The tail.** "250 per range" is an average that hides the launch's critical path: with 12-bit
windows the top window of a scalar below 2^253 (the bench) holds one varying bit, so one bucket
receives half of all points — 524,288 serial mixed additions in one thread, about 5 s, which is the
whole launch's time. On the production circuit (scalars below 2^254) the top window has two bits and
three such buckets of n/4 ≈ 1.4 M points each: ≈ 14 s for a G1 MSM at ≈ 10 µs per serial addition,
2.1 M for `h` (28 s), and the G2 MSM's addition is four times the cost (61 s) — the profile above,
line by line. A skewed witness (small values, booleans) fattens the low windows' first buckets the
same way. The memory system is not involved: the cached-gather row is no faster.

**The bound.** The shared pipeline plans the reduction (`pipeline::plan_ranges`): the bucket-sum
launch runs over pieces of at most `CHUNK` = 64 points; then a new kernel, `jacobian_sum` (full
Jacobian additions over ranges of the previous level's outputs), runs one level per ⌈log₆₄⌉ of the
longest bucket — two or three levels for the production circuit — until one sum per (window, bucket)
remains, and the host reduces the windows as before. The plan is data, computed once per MSM from
the sorted layout, so the CPU launcher, the CUDA host and the Metal host execute the same levels;
the fixture proof stays byte-identical through all three. The kernel contract did not change; the
ABI grew by two names (`jacobian_sum_g1`, `jacobian_sum_g2`), `kernel-check` reports 15 checks —
15/15 on the RTX 5080 for the `sm_120` and the `sm_89` module (`ptx-c21`).

**The production circuit after the bound — MEASURED 2026-09-13, RTX 5080 shared with the other
tenant, streaming path.** synthetic-loop-100k through `cuda-oxide`: verified under the upstream
verifier, every mutation arm killed; the harness's prove column 15.9 s, of which 8.6 s is the
streaming path's per-call zkey read, parse and grouping on the host and 6.0 s the device — the five
MSMs 5.2 s (h 1.84, a 0.76, b1 0.80, b2 (G2) 0.98, c 0.78) against 139 s before; the scatter 0.47 s,
the transforms 0.29 s. Inside an MSM the bucket-sum launch is now 0.04–0.14 s and the levels above
it are inside that figure; what remains is the host: the counting sort (0.21 s per witness MSM, 0.90
s for h), the digits round trip and the `order` upload (≈ 0.5 s per MSM, the rest of each line), and
the per-window reduction (0.05 s; 0.19 s for G2). The resident path (the zkey uploaded once per
process) would drop the 8.6 s per call; it needs ≈ 8 GB of device memory and a lull, so it is
INFERRED here from the C15 resident run's non-MSM phases, not measured on this shared device.
synthetic-loop-0 the same minute: verified, every arm killed, prove 14.7 s (device 5.9 s: the five
MSMs 5.1 s — h 1.89, a 0.73, b1 0.71, b2 1.05, c 0.71). Reports:
[`reports/synthetic-loop-100k.cuda-oxide.md`](./reports/synthetic-loop-100k.cuda-oxide.md),
[`reports/synthetic-loop-0.cuda-oxide.md`](./reports/synthetic-loop-0.cuda-oxide.md).

**The host counting sort in parallel (C22) — MEASURED 2026-09-13.** Once the device chain is
bounded, the largest host item is the per-window counting sort, with the GPU idle through it. The
windows are independent, so `sort_all_windows_parallel` runs them on separate threads (24 on this
box) and stitches the results in order; `sort_all_windows` stays as the serial reference. On the
streaming run the `h` MSM's host sort fell from 0.90 s to 0.68 s and each witness MSM's from 0.21 s
to ≈ 0.09 s — ≈ 0.6 s off a ≈ 6.7 s device-side proof; the memory-bound stitch (740 MB of `order`
for `h`) is the floor. The two GPU hosts and the CPU launcher share the function, so the metal-cpu
arm and the byte-identical-proof test cover it.

**One steer shaped C22's form.** The bounded-chain orchestration of C21 was written by mutating the
proven `msm` in place; on review that was changed to keep the original single-launch `msm` as the
in-repo reference and add the bounded-chain path as `msm_planned` beside it (in the shared pipeline,
the CUDA host and the Metal host), with the prove paths calling `msm_planned` and the kernel check
running BOTH and requiring each to equal the naive expectation. The device kernel check is 17 checks
now (was 15): every kernel, `msm (g1/g2)`, `msm planned (g1/g2)`, and the fixture proof — 17/17 on
the RTX 5080 for the `sm_120` and the `sm_89` module. The kernels are unchanged, so `ptx-c21` still
loads.

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
`RISC0_GROTH16_RESIDENT=0` makes the CUDA arm stream: every phase uploads what it needs and frees it
before the next, ≈ 2 GB on the device instead of ≈ 8 GB resident, so the arm fits beside the other
tenant's peak at the price of re-uploading the zkey per proof (≈ 2 GB since C19, which runs the
transform phase in four buffers). `RISC0_GROTH16_TIMING=1` makes the CUDA prover print per-phase
wall-clock (uploads, scatter, the three coset transforms, each MSM's host sort / bucket sums /
reduction, assembly) to stderr, so a granted window yields a profile and not only a total.

A short sample window is no guarantee for a step that needs the tenant idle: on 2026-09-13 the
canonical control (≈ 7 GiB) met `cudaMallocAsync … out of memory` 29 s after three samples had read
10.3–11.4 GiB free. The waiter for such a step is `GPU_FREE_MIN_GIB=10.5 gpu-free.py 30 2` — a full
minute with every sample above the line, which on this box means the tenant is between jobs — and
the streaming arm (2.5 GiB) is what runs when the tenant is not.

### The production circuit through the Metal shaders on the CPU (`metal-cpu`), control = `reference` — MEASURED 2026-09-12

The Metal arm has no device in this session; its shaders, compiled as C++ (C14), prove the
production circuit on the CPU through the same boundary and the same harness:

| case                | rewrite                                  | rewrite verifies (1) | bit-flip | cross-claim | canonical answers   | malformed input | derive s | prove s |
| ------------------- | ---------------------------------------- | -------------------- | -------- | ----------- | ------------------- | --------------- | -------: | ------: |
| synthetic-loop-100k | metal-cpu                                | **yes**              | killed   | killed      | not run (see below) | killed          |     26.2 |   424.5 |
| synthetic-loop-100k | metal-cpu (C21 plan + C22 parallel sort) | **yes** (2026-09-13) | killed   | killed      | not run             | killed          |     26.1 |   405.6 |

Full row in
[`reports/synthetic-loop-100k.metal-cpu.md`](./reports/synthetic-loop-100k.metal-cpu.md). 424.5 s is
the shader source run one thread index at a time on one core; it is a correctness run, not a timing.
Re-run 2026-09-13 with the bounded-chain plan (C21) and the parallel host sort (C22) it verifies at
405.6 s (backend prove 391.7 s); the plan and the parallel sort help the CPU arm too, but this stays
a correctness result, not a timing one. What a Mac adds is the Metal compiler and the device: the
arithmetic, the layouts and the whole pipeline down to a verifying production proof are established
here.

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
