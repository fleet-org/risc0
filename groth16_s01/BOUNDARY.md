# GROTH16 s01/2 — the SNARK stage, mapped, and the reimplementation boundary named

**Milestone:** [GROTH16 s01](https://github.com/fleet-org/risc0/milestone/1) · **Issue:** [#3@fleet-org/risc0](https://github.com/fleet-org/risc0/issues/3) · **Base:** `integration/staging-platform` = upstream tag `v3.0.4` ([`d7ee368e`](https://github.com/fleet-org/risc0/tree/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366))
**Status:** complete for what can be read from source; wall-clock shares are UNVERIFIED (need a CUDA host, see §3).
**Epistemic marks:** MEASURED = read from the pinned source or produced by a command; INFERRED = derived from a measured fact; UNVERIFIED = not checked.

Versions this map is pinned to (MEASURED from bento's lockfile at boundless [`93e971a6`](https://github.com/boundless-xyz/boundless/blob/93e971a653258bb2cf1de920b103fcfdfd9ef994/bento/Cargo.lock#L6268)): `risc0-zkvm 3.0.4`, `risc0-zkp 3.0.3`, `risc0-groth16 3.0.3`, `risc0-groth16-sys 0.1.0`, `sppark 0.1.12`, `blst 0.3.15`, `circom-witnesscalc 0.2.1`. The `risc0-groth16-sys 0.1.0` kernels on crates.io are byte-identical to `risc0/groth16-sys/kernels/` at the base tag (diffed).

---

## 1. Call graph — from the bento task to the first kernel launch

bento is not in this repository's `main`; the deployed bento is `bento/` in `boundless-xyz/boundless` (fleet manifest `FLEET_MARKET_SOURCE_REPO`). It reaches this repository's crates through crates.io, so the graph crosses three ownership domains: **boundless** (task + blake3 circuit), **risc0-zkvm** (compression driver), **risc0-groth16 / risc0-groth16-sys** (prover + kernels; the fork's editable surface).

```
bento agent  (-t snark stream; task_def = {"Snark": {"receipt": <stark uuid>, "compress_type": Groth16 | Blake3Groth16}})
  └─ workflow::tasks::snark::stark2snark                          boundless  bento/crates/workflow/src/tasks/snark.rs
       ├─ S3 read  receipts/stark/<uuid>.bincode  → Receipt (succinct)
       ├─ CompressType::Groth16
       │    └─ agent.prover.compress(&ProverOpts::groth16(), &receipt)     ── crate boundary → risc0-zkvm (crates.io 3.0.4)
       │         └─ ProverServer::compress → succinct_to_groth16          risc0/zkvm/src/host/server/prove/mod.rs
       │              ├─ identity_p254(succinct)   (recursion program: poseidon254-hashed STARK; GPU HAL if built with cuda)
       │              ├─ seal_bytes = ident_receipt.get_seal_bytes()
       │              └─ risc0_groth16::prove::shrink_wrap(&seal_bytes)  ── crate boundary → risc0-groth16 (3.0.3)
       │                   ├─ [cfg(not(cuda))]  prove/docker.rs  → `docker run risczero/risc0-groth16-prover:v2025-04-03.1`  (canonical CPU path = Docker)
       │                   └─ [cfg(cuda)]       prove/cuda.rs
       │                        ├─ to_json(seal)                          → {"iop": [decimal strings]}
       │                        ├─ circom_witnesscalc::calc_witness(json, stark_verify_graph.bin)   (CPU, single thread)
       │                        ├─ risc0_zkp::hal::cuda::singleton().lock()
       │                        └─ risc0_groth16_sys::prove(&ProverParams, &SetupParams)   ── crate boundary → risc0-groth16-sys (0.1.0)
       │                             └─ extern "C" risc0_groth16_cuda_prove(SetupParams*, ProveParams*)   kernels/cuda/ffi.cu
       │                                  ├─ SRS(zkey)            mmap + HtoD of A, B1, B2, C, H point sections     ← FIRST GPU WORK (copies)
       │                                  ├─ groth16_prover(...)  load preprocessed_coeffs.bin, fuzzed_msm_results.bin; init kernels
       │                                  └─ prove(public_path, witness)  → kernel sequence of §2                     ← FIRST KERNEL LAUNCH: generate_partial_twiddles (init), witness_into_poly (prove)
       └─ CompressType::Blake3Groth16
            └─ blake3_groth16::compress_blake3_groth16(&receipt)           boundless  blake3_groth16/src/lib.rs
                 ├─ default_prover().compress(&ProverOpts::succinct())
                 ├─ prove::succinct_to_blake3_groth16 → identity_p254 → identity_seal_json (iop + journal/pre/post/id bits + control_root)
                 ├─ [cfg(not(cuda))] prove/docker.rs → `docker run boundless-blake3-g16:latest`
                 └─ [cfg(cuda)]      prove/cuda.rs → circom_witnesscalc (verify_for_guest_graph.bin) → risc0_groth16_sys::prove(...)  ← SAME CALL, different artifacts
```

Permalinks (MEASURED): task [`snark.rs#L43`](https://github.com/boundless-xyz/boundless/blob/93e971a653258bb2cf1de920b103fcfdfd9ef994/bento/crates/workflow/src/tasks/snark.rs#L43) (Groth16) and [`#L63`](https://github.com/boundless-xyz/boundless/blob/93e971a653258bb2cf1de920b103fcfdfd9ef994/bento/crates/workflow/src/tasks/snark.rs#L63) (Blake3Groth16); zkvm [`server/prove/mod.rs`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/zkvm/src/host/server/prove/mod.rs) (`succinct_to_groth16`); groth16 [`prove/mod.rs#L17`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/groth16/src/prove/mod.rs#L17), [`prove/cuda.rs#L54`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/groth16/src/prove/cuda.rs#L54), [`prove/docker.rs#L57`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/groth16/src/prove/docker.rs#L57); sys [`lib.rs#L70`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/groth16-sys/src/lib.rs#L70), [`ffi.cu#L53`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/groth16-sys/kernels/cuda/ffi.cu#L53); blake3 [`prove/cuda.rs#L63`](https://github.com/boundless-xyz/boundless/blob/93e971a653258bb2cf1de920b103fcfdfd9ef994/blake3_groth16/src/prove/cuda.rs#L63).

Two facts the plan did not carry, both MEASURED:

- **The "canonical CPU path" is Docker.** Without the `cuda` feature the prover shells out to a container image. A CPU-only bento cannot run this stage without Docker; the s01/5 control arm therefore needs a CUDA host or Docker, or the captured canonical outputs of the corpus.
- **Two circuits, one engine.** `Groth16` (risc0's `stark_verify` circuit, rzup component `risc0-groth16` v0.1.0) and `Blake3Groth16` (boundless's `verify_for_guest` circuit, `BLAKE3_GROTH16_ARTIFACTS_URL`, 2.4 GB tar.xz) differ only in the artifact directory handed to the same `risc0_groth16_sys::prove`. INFERRED: production exercises the blake3 variant (the fleet template env pins its artifacts URL; the selector picks the compress type per order).

## 2. Kernel inventory

All device code lives in [`risc0/groth16-sys/kernels/cuda/`](https://github.com/fleet-org/risc0/tree/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/groth16-sys/kernels/cuda) (1,297 lines incl. comments: `ffi.cu` 80, `groth16_prover.cuh` 396, `groth16_srs.cuh` 361, `groth16_coeffs.cuh` 239, `chacha.cuh` 91, `util.cuh` 130) and sits on **sppark 0.1.12** templates (6,708 lines used: `ff/mont_t.cuh` 1,120, `ff/alt_bn128*.hpp` 612, `ec/{jacobian,xyzz,affine}_t.hpp` 1,247, `msm/pippenger.cuh` + `.hpp` + `sort.cuh` + `batch_addition.cuh` 1,683, `ntt/*` 1,649, `util/gpu_t.cuh` 397). Built by `risc0-build-kernel` with nvcc, `-D__ADX__`, `blst` linked statically. `gpu` = `gpu_t` from sppark (default device, streams, `sm_count()`).

Per proof (`groth16_prover::prove`), in launch order. `n` = `num_vars` (= size of the zkey's A section), `N` = `domain_size` (= size of the H section, a power of two), `sm` = SM count of the device.

| # | kernel / call | computes | launch geometry (grid × block) | source |
|---|---|---|---|---|
| i1 | `generate_partial_twiddles` (init, once per `groth16_prover`) | table of coset-shift powers (`WINDOW_NUM × WINDOW_SIZE`) for the LDE; shift = ω<sub>2N</sub> | `WINDOW_SIZE/32 × 32` | `groth16_prover.cuh` `init_memory` |
| i2 | `chacha_generate_random_scalars<8>` (init) | deterministic "fuzz" scalars (ChaCha8, key = 0, nonce = 0) for n elements | `sm × 1024` | `util.cuh` |
| 1 | `witness_into_poly` (A) | for each unique constraint c: A<sub>c</sub> = Σ coeff.value · w[coeff.s] over the A-side coefficient list | `sm × 512` (`__launch_bounds__(512)`) | `groth16_prover.cuh` |
| 2 | `witness_into_poly` (B) | same for the B-side list (second stream, event-synchronised) | `sm × 512` | |
| 3 | `coeff_wise_mul` | C = A ∘ B on the evaluation domain H | `sm × 1024` | `util.cuh` |
| 4 | `NTT_sequence` × 3 (A, B, C) | sppark `NTT::Base_dev_ptr` inverse (NR) → `LDE_distribute_powers` (multiply by shift powers = move to coset g·H) → `NTT::Base_dev_ptr` forward (RN) | sppark mixed-radix CT/GS kernels; `LDE_distribute_powers`: `N/warp × warp` | `groth16_prover.cuh` `LDE_launcher`, sppark `ntt/` |
| 5 | `coeff_wise_mul_and_sub` | A ∘ B − C on the coset = h(x)·Z<sub>H</sub>(x) evaluations (the quotient's coset form; the zkey's H section absorbs Z<sub>H</sub>) | `sm × 1024` | `util.cuh` |
| 6 | `msm_g1.invoke(h, H points, N, coset evals)` | π<sub>h</sub> = Σ evals<sub>i</sub> · H<sub>i</sub> | sppark Pippenger: `wbits = min(⌊log2(1.5·N)⌋ − 8, 18)`, digit sort + bucket accumulation + reduction | sppark `msm/pippenger.cuh` |
| 7 | `coeff_wise_add` | w ← w + fuzz (blinds the MSM inputs; the precomputed −MSM(fuzz) is added back on the host) | `sm × 1024` | `util.cuh` |
| 8 | `msm_g1.invoke(a, A points, n, w, mont=false)` | Σ w<sub>i</sub>·A<sub>i</sub> | Pippenger | |
| 9 | `msm_g1.invoke(b_g1, B1 points, n, w, mont=false)` | Σ w<sub>i</sub>·B1<sub>i</sub> | Pippenger | |
| 10 | `msm_g2.invoke(b_g2, B2 points (G2), n, w, mont=false)` | Σ w<sub>i</sub>·B2<sub>i</sub> in G2 (Fp2 coordinates) | Pippenger over `xyzz_t<fp2_t>` | |
| 11 | `msm_g1.invoke(c, C points, |C|, w[n−|C|..], mont=false)` | Σ<sub>i ≥ num_public+1</sub> w<sub>i</sub>·C<sub>i</sub> (private-witness slice) | Pippenger | |
| h | host (CPU, blst-backed) | add −MSM(fuzz) precomputes; sample r, s (`std::random_device`, 31 bytes each); π<sub>A</sub> = α + Σ + r·δ, π<sub>B</sub> = β + Σ + s·δ (G1 and G2), π<sub>C</sub> = Σ<sub>C</sub> + π<sub>h</sub> + s·π<sub>A</sub> + r·π<sub>B1</sub> − rs·δ; write `proof.json`; `public.json` is written by a parallel host thread from w[1..=num_public] | — | `groth16_prover.cuh` `prove` |

Not in the per-proof path but in the per-call cost: `SRS::SRS` — `open` + `mmap` of the zkey and `HtoD` of all five point sections **on every call** (`risc0_groth16_cuda_prove` constructs `SRS` and `groth16_prover` per proof; nothing is cached across proofs), plus `preprocessed_coeffs` load (mmap + 2 `HtoD`) and `read_fuzzed_results_from_file`.

**Share of wall-clock — UNVERIFIED.** No CUDA host is reachable from this session. INFERRED ordering from operation counts: the five MSMs (≈3n + N point-scalar pairs in G1 plus n in G2, the G2 MSM costing ≈3× per point) dominate; the three NTT sequences (6 NTTs of size N + 3 LDE passes) are second; the per-call SRS upload is proportional to the zkey size and may be comparable to the MSMs for a multi-GB zkey; witness generation (`circom-witnesscalc`, single-threaded CPU) precedes all of it and is inside the stage's wall-clock. s01/6 measures these on the production host; the corpus (s01/3) records the canonical total per case.

## 3. The reimplementation boundary — named

**The boundary is `risc0_groth16_sys::prove(&ProverParams, &SetupParams) -> anyhow::Result<()>`** ([`risc0/groth16-sys/src/lib.rs#L70`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/groth16-sys/src/lib.rs#L70)), whose canonical implementation is the C ABI `risc0_groth16_cuda_prove` in `ffi.cu`.

Why here and not higher or lower:

- **Both circuits pass through it** (§1), so one substitution serves `Groth16` and `Blake3Groth16`, and boundless's crate needs no change for a CUDA arm.
- **Everything above it is circuit- and hardware-agnostic Rust** (identity_p254, seal-to-JSON, witness generation, receipt assembly), so it stays as-is — exactly the plan's "anything upstream of the boundary" rule.
- **Everything below it is the GPU prover proper** (§2): zkey ingestion, coefficient scatter, NTT/LDE, MSM, proof assembly. That is the whole surface a cuda-oxide or Metal arm replaces.
- **It is already an exact signature both sides implement** — the ffi-boundary concept from `fleet-org/zkvm-gpu-prover` `GOAL-EXEC.md` §4, which the plan says to reuse verbatim: mismatch = BLOCKED + escalate.
- Lower (per kernel) would tie the arms to sppark's data layouts; higher (`shrink_wrap`) would force each arm to re-implement witness generation and JSON I/O that are not GPU work.

### 3.1 Shape after s01/4 — three implementations, selected at runtime, available by build

```rust
// risc0-groth16-sys (fork). Signature unchanged for callers.
pub fn prove(prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()> {
    backend::select()?.prove(prover, setup)
}

pub trait Groth16Backend {
    fn name(&self) -> &'static str;                       // "canonical" | "cuda-oxide" | "metal"
    fn prove(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()>;
}

// availability = build target/features; selection = configuration (env RISC0_GROTH16_BACKEND, default "canonical")
//   canonical   #[cfg(feature = "cuda")]                                       → extern "C" risc0_groth16_cuda_prove (unchanged)
//   cuda-oxide  #[cfg(feature = "cuda-oxide")]                                 → Rust kernels (s01/4)
//   metal       #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))] → MSL kernels (s01/4b)
//   reference   #[cfg(feature = "reference")]                                  → risc0-groth16-core on the CPU (tests + harness control arm; never the default)
//   oxide-cpu   #[cfg(feature = "oxide-cpu")]                                  → the CUDA arm's kernel bodies on the host launcher (tests; never the default)
```

**Landed (C3, C4):** `risc0/groth16-sys/src/backend.rs` implements exactly this; `risc0/groth16-core` is the shared `no_std` crate (field · fp2 · ec · ntt · msm · zkey/wtns · prover) the arms build on, and `backend/reference.rs` runs its pipeline behind the boundary (DEF-G16-006). The reference is proven on the in-tree `multiplier2` fixture: its `proof.json` verifies under the unmodified `risc0-groth16` verifier, and an unsatisfied witness yields a proof that verifier rejects.

**Landed (C8, host side of `cuda-oxide`):** `risc0/groth16-cuda` is the CUDA arm's host — it loads the cuda-oxide device module through `cuda-core` 0.3.1 (the driver-API runtime cuda-oxide itself uses; `RISC0_GROTH16_CUDA_MODULE` names a `.ptx`, a `.cubin`, or a `cargo oxide` build product) and runs the pipeline with device buffers; `backend/cuda_oxide.rs` registers it as `BackendKind::CudaOxide` under the `cuda-oxide` feature. The contract between the two halves is one shared module, `risc0_groth16_oxide::abi` (records = core's `#[repr(C)]` types with their sizes pinned by test; every slice a `(ptr, len)` pair; 256 threads per block; the kernel-name list the host resolves at start-up), and the coset transform's launch order is data, `risc0_groth16_oxide::schedule`, executed by the CPU pipeline, the Metal prover and the CUDA prover alike. The device half — `#[kernel]` wrappers around the oxide bodies — is a template in `groth16_s01/cuda-kernels` that only `cargo oxide` on a CUDA host can compile (E2); `groth16-cuda-kernel-check` is the first thing to run there. MEASURED without a GPU: the host crate and the backend type-check with the CUDA 13 headers alone (`groth16_s01/scripts/cuda-headers.sh`), in CI and in this session.

**Landed (C9, `metal` behind the boundary):** `backend/metal.rs` registers `risc0_groth16_metal::device::prove` as `BackendKind::Metal` under the `metal` feature, compiled only for macOS on Apple Silicon (the feature exists on every build, so selecting `metal` elsewhere is an explicit *unavailable*, never a fallback); the harness gained `--features metal` and `--features cuda-oxide` so both arms are selectable as the confirmed kind of a run. MEASURED: type-checks for `aarch64-apple-darwin` with `--features metal`; unrun (E3).

Constraints this satisfies: the canonical path stays selectable on every build that has it (definition of done #2); selecting an unavailable backend is an error, never a silent fallback (three-state: *unavailable* ≠ *failed* ≠ *succeeded*); the harness (s01/5) runs canonical and rewrite on the same input in one process by flipping the selector.

**Deltas above the boundary that the Metal arm needs** (CUDA arms need none): `risc0-groth16`'s [`prove/mod.rs`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/groth16/src/prove/mod.rs#L17) branches only on `cuda`; with `cuda` off it is Docker. A Metal build needs the `cuda.rs` code path (witness generation + `risc0_groth16_sys::prove`) available under a `metal` feature, minus the CUDA-only `risc0_zkp::hal::cuda::singleton().lock()`. The same two-line change applies to boundless's `blake3_groth16/src/prove.rs` for the blake3 circuit on macOS — a fleet/boundless-side integration item, noted here, not built here.

## 4. Data contract at the boundary

### 4.1 Inputs

`SetupParams` (three paths, all produced once per circuit by the trusted setup + `risc0_groth16_cuda_setup`):

| field | file | format (MEASURED from `groth16_srs.cuh` / `groth16_coeffs.cuh` / `groth16_prover.cuh`) |
|---|---|---|
| `srs_path` | `stark_verify_final.zkey` (risc0) / `verify_for_guest_final.zkey` (blake3) | snarkjs **zkey**: magic `"zkey"`, u32 version, u32 `num_sections` (must be 10); each section = u32 id + u64 size. §1 header: u32 protocol (must be 1 = Groth16). §2 Groth16 header: `q` (u32 len + bytes), `r` (u32 len + bytes), u32 `num_vars`, u32 `num_public`, u32 `domain_size`, then vk α<sub>1</sub> (64 B), β<sub>1</sub> (64), β<sub>2</sub> (128), γ<sub>2</sub> (128), δ<sub>1</sub> (64), δ<sub>2</sub> (128). §3 IC (skipped). §4 coefficients: `{u32 m, u32 c, u32 s, 32-byte value}` packed, 4-byte pad before the array. §5 A, §6 B1, §8 C, §9 H: G1 affine, 64 B each; §7 B2: G2 affine, 128 B each. Points are copied to the device **raw** (`HtoD` of the byte range, no conversion). **MEASURED** on the in-tree `groth16_proof/circom-compat/test/data/multiplier2_final.zkey`: coordinates are little-endian limbs in **Montgomery form** (raw α₁ is off-curve, ×R⁻¹ is on-curve); §4 coefficient `value`s are stored as **v·R² mod r** (the constant 1 reads back as R², −1 as −R²), so `witness_into_poly`'s Montgomery multiply `w · vR² · R⁻¹` yields `(w·v)·R` — the pipeline is in Montgomery form from the scatter onward, which is why the h-MSM takes `mont = true` and the witness MSMs `mont = false`. |
| `pcoeffs_path` | `preprocessed_coeffs.bin` | `4 × size_t` header = (`count_a`, `count_b`, `unique_a`, `unique_b`) followed by `coeff_t[count_a + count_b]` (A-side list then B-side list, each sorted by constraint `c`, `coeff_t = {u32 m, c, s; fr_t value}`) followed by `u32[unique_a + unique_b]` start-index lists (each list ends with a sentinel = its coefficient count). Derived from zkey §4 by the setup path (`SRS_READ_COEFFS`). |
| `fres_path` | `fuzzed_msm_results.bin` | one raw `msm_results` struct = `{point_t a, b_g1, c, h; point_fp2_t b_g2}` (sppark Jacobian layouts) holding **−MSM(fuzz)** for A, B1, B2, C, computed at setup with the same deterministic ChaCha8 stream (`h` unused). Size check enforced. |

`ProverParams`:

| field | meaning |
|---|---|
| `witness: *const u8` | `num_vars` consecutive **32-byte little-endian field elements in canonical (non-Montgomery) form**, exactly the `wtns` payload (`wtns_file::FieldElement<32>`); `w[0] = 1`, `w[1..=num_public]` = the public inputs, rest private. MEASURED twice: every witness-derived MSM passes `mont = false`, and the fixture's `multiplier2.wtns` reads back as the plain integers `[1, 33, 3, 11]`. |
| `public_path` | output: `public.json` = JSON array of `num_public` decimal strings, `w[1..=num_public]` |
| `proof_path` | output: `proof.json` (below) |

### 4.2 Outputs

`proof.json` (snarkjs shape, decimal strings, affine, canonical form):
```json
{ "pi_a": ["x", "y", "1"], "pi_b": [["x0", "x1"], ["y0", "y1"], ["1", "0"]], "pi_c": ["x", "y", "1"], "protocol": "groth16" }
```
Rust then maps it ([`types.rs`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/groth16/src/types.rs)) to `Seal { a: [x, y], b: [[x1, x0], [y1, y0]], c: [x, y] }` — 32-byte **big-endian** words, G2 coordinates **swapped** — and `Seal::to_vec()` = 256 bytes (`a` 64 · `b` 128 · `c` 64). That byte string is `Groth16Receipt.seal`.

### 4.3 What the stage receives and emits (the replayable forms for s01/3)

| | serialized form | where bento keeps it |
|---|---|---|
| stage input | `risc0_zkvm::Receipt` (succinct), bincode | object store `receipts/stark/<stark-uuid>.bincode` |
| boundary input | derived deterministically from the stage input on any machine with the recursion prover: `identity_p254` → seal bytes (`K_SEAL_WORDS` u32) → `to_json` → witness via `circom-witnesscalc` (UNVERIFIED: identity_p254 output byte-stability across hosts; the harness treats it as a derived, not stored, artifact) | not stored |
| boundary output | `proof.json` + `public.json` | not stored (temp dir unless `RISC0_WORK_DIR` / `BLAKE3_GROTH16_WORK_DIR` is set) |
| stage output | `Receipt` with `InnerReceipt::Groth16(Groth16Receipt { seal: 256 B, claim, verifier_parameters })` (or `Blake3Groth16Receipt`), bincode | `receipts/groth16/<snark-job-uuid>.bincode` · `receipts/blake3_groth16/<snark-job-uuid>.bincode` |

### 4.4 Public inputs the oracle checks (risc0 circuit)

`Groth16Receipt::verify_integrity_with_context` ([`receipt/groth16.rs#L76`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/zkvm/src/receipt/groth16.rs#L76)) calls `risc0_groth16::Verifier::new(seal, control_root, claim_digest, bn254_control_id, vk)` ([`verifier.rs#L93`](https://github.com/fleet-org/risc0/blob/d7ee368e59ce6d9f94ecaeb6fc845e39ec5e9366/risc0/groth16/src/verifier.rs#L93)), which builds the five public inputs **`[a0, a1, c0, c1, id_bn254_fr]`** = `split_digest(control_root)`, `split_digest(claim.digest())`, and `BN254_IDENTITY_CONTROL_ID` byte-reversed as an Fr — with `Groth16ReceiptVerifierParameters::default()` = `(ALLOWED_CONTROL_ROOT, BN254_IDENTITY_CONTROL_ID, risc0_groth16::verifying_key())`. This is the exact "same public inputs" assertion #2 of s01/5: the rewrite's seal must verify against the public inputs derived from the **same claim** as the canonical seal. The verifier is in `risc0-groth16`'s default feature set (no `prove`, no GPU) and is the unmodified oracle.

## 5. Not in scope (stays as-is)

Everything above the boundary: bento task plumbing and the object store; `ProverOpts` / `compress`; `identity_p254` and the recursion prover; `to_json` / `identity_seal_json`; witness generation (`circom-witnesscalc` — a CPU step whose wall-clock s01/6 should still report); receipt assembly and `verify_integrity`; seal encoding on chain. Also out: the zkey/trusted-setup artifacts themselves, the rzup component pipeline, and the `setup` (precomputation) path — an arm may keep consuming `preprocessed_coeffs.bin` / `fuzzed_msm_results.bin` exactly as produced today.

## 6. Structural patterns to carry (the plan's §3.2)

From `fleet-org/zkvm-gpu-prover` `GOAL-EXEC.md` §4 (module manifest: `crate` · `cuda` · `ffi-boundary` · `rust-owns` · `cuda-owns` · `surface`) and `fleet-org/bbstark` (launcher declaration headers decoupling callers from implementations; per-module kernel files; the pipeline assembled in Rust):

| manifest field | this milestone |
|---|---|
| `crate:` | `risc0-groth16-sys` (host: backend selection, params, artifact validation, output JSON) + one kernel crate per arm |
| `cuda:` / `metal:` | `risc0/groth16-sys/kernels/cuda/` (canonical, untouched) · `kernels/oxide/` (Rust `#[kernel]` modules: field, coeffs, ntt, msm, proof) · `kernels/metal/` (MSL) |
| `ffi-boundary:` | `Groth16Backend::prove(&ProverParams, &SetupParams)` (Rust) and `risc0_groth16_cuda_prove(SetupParams*, ProveParams*) -> const char*` (C ABI, canonical); both sides implement it; mismatch = BLOCKED + escalate |
| `rust-owns:` | zkey parsing and validation, coefficient/fuzz artifact loading, witness marshalling, stream/launch orchestration (the **pipeline**), r/s sampling, proof assembly and serialization |
| `cuda-owns:` / `metal-owns:` | field arithmetic, coefficient scatter, NTT + LDE, Pippenger MSM, ChaCha fuzz — kernel compute only |
| `surface:` | the exhaustive file list, per PR |

The launcher/implementation split maps onto cuda-oxide as: the kernel library crate is the *implementation*; the generated typed launch API of each `#[cuda_module]` is the *declaration header*; the host crate assembles the sequence of §2 in Rust. On Metal: one Rust dispatch wrapper per pipeline state is the declaration, the `.metal` sources compiled into a metallib are the implementation. Substitutability — the property the differential harness needs — follows from keeping the pipeline in Rust on both arms.

---

**Bottom line.** The boundary is `risc0_groth16_sys::prove` in this fork; both circuits already cross it; the canonical CUDA implementation stays selectable behind a `Groth16Backend` trait with runtime selection; the data contract is fully specified above with one remaining mark (kernel wall-clock shares: UNVERIFIED until a CUDA host runs the corpus). Next: s01/3 captures the stage input/output forms of §4.3; s01/4 and s01/4b implement §3.1.

## 7. Why `oxide-cpu` exists, and the devices the arms are for

**`oxide-cpu` is not a CPU prover.** It is the CUDA arm's kernel code — the per-output-index bodies in `risc0-groth16-oxide::kernels`, the same functions the `#[kernel]` wrappers call on the device — run by the host launcher (`CpuLauncher`) through the same boundary. cuda-oxide compiles *Rust*, so the bodies can be compiled twice: by rustc for the host and by cuda-oxide for PTX. The host compilation is what lets the rewrite be proven against the canonical verifier on the production circuit before any GPU exists (C6: byte-identical proof to the reference on `stark_verify`) and keeps it proven on every PR afterwards, where CI has no GPU. It is registered as a **testing kind** (`BackendKind::OxideCpu`, DEF-G16-007), never the default and never a product path; the product arms are `cuda-oxide` and `metal`.

**It does not relax the device constraints.** A body is one thread's work: no allocation, no data-dependent writes outside its own output slot, indices computed from the output index — the shape a GPU thread has. What the CPU launcher proves is therefore the code the device runs; the device-only parts are isolated where they can be checked on the device alone: the thread-index wrapper and `(ptr, len)` parameters (`groth16_s01/cuda-kernels`, the ABI), buffer transfer and launch (`risc0-groth16-cuda`), and the launch order (`risc0_groth16_oxide::schedule`, data shared by all three executors).

**Devices, in order of the plan.** The baseline differential run (s01/5 with `--control canonical`) needs the canonical CUDA kernels next to the rewrite, i.e. a CUDA host: a rented instance of the class the fleet's prover nodes are (the builder image is CUDA 13.0.2; the publish labels name `sm_120`). The next version targets owned hardware:

| device | arm | what matters | status |
|---|---|---|---|
| NVIDIA RTX 4090 (sm_89, 24 GB, R580+ driver for CUDA 13) | `cuda-oxide` | in cuda-oxide's `sm_80`–`sm_100a` range (pin §5); 24 GB is 6× the arm's peak working set below | INFERRED from the pin; unrun |
| fleet prover nodes (`sm_120`) | `cuda-oxide` | in the pinned target list (`crates/cuda-target-spec`; CORRECTED — an earlier read had `sm_80`–`sm_100a`): build with `cargo oxide build --arch sm_120`; base `sm_100` PTX also JIT-loads on `sm_120`, `sm_100a`/`sm_100f` do not (NVIDIA compatibility rules) | MEASURED by reading; unrun |
| Apple M3 Max, 40-core GPU, Metal 3 / Apple9, 128 GB unified (spec captured 2026-09-12) | `metal` | no native INT64 ALU and multi-cycle `mulhi`: the MSL kernels use 32-bit-limb CIOS for that reason; 32 KB threadgroup memory is untouched (one thread per output, no tiling); `maxBufferLength` 72 GiB and unified memory make the whole zkey resident and zero-copy (`MTLStorageModeShared`) a later, cheap step | MEASURED spec; arm unrun |

**Working-set budget per proof on the production circuit** (INFERRED from the MEASURED dimensions in §4: 5,635,930 variables, domain 2²³, 29,098,147 coefficients), for the CUDA host as written (polynomial buffers freed before the points go up):

| phase | device bytes |
|---|---|
| scatter: coefficients (40 B) + witness + tables | ≈ 2.1 GB |
| transform: 7 polynomials × 2²³ × 32 B + tables | ≈ 2.7 GB |
| MSM: G1 a, b1, c, h (72 B) + G2 b2 (136 B) + scalars + digits/order | ≈ 3.0 GB |

So the arm fits any 8 GB card, the M3 Max trivially, and keeping the zkey's points resident across calls (the canonical path re-uploads per call, §1) costs about 2.6 GB — the obvious first optimisation once correctness is measured. The MSM's bucket-sum launch has only 4095 threads per window (the M3 Max has 5,120 INT32 lanes; the RTX 4090 16,384): running all 22 windows in one launch is the second.

