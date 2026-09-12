# GROTH16 s01/1 — what cuda-oxide is, pinned before anything is built on it

**Milestone:** [GROTH16 s01](https://github.com/fleet-org/risc0/milestone/1) · **Issue:**
[#2@fleet-org/risc0](https://github.com/fleet-org/risc0/issues/2) **Pinned at:**
[NVlabs/cuda-oxide @ `6abfaa09`](https://github.com/NVlabs/cuda-oxide/commit/6abfaa091e29a6275c1943895bfbc97efa306e98)
(main, 2026-09-11) · latest release `v0.2.1` (2026-06-10) **Epistemic marks:** MEASURED = read from
the pinned repository / crates.io / GitHub API on 2026-09-12; INFERRED = derived; UNVERIFIED = not
checked (no CUDA host in this session).

## The six questions

### 1. Identity and version — MEASURED

`cuda-oxide` is **[NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide)**: "a Rust-to-CUDA
compiler that lets you write (SIMT) GPU kernels in safe(ish), idiomatic Rust … compiles standard
Rust code directly to PTX". Apache-2.0, created 2026-04-22, status **alpha** ("expect bugs,
incomplete features, and API breakage"), releases `v0.1.0` (2026-05-07), `v0.2.0` (2026-06-05),
`v0.2.1` (2026-06-10); active (pushed 2026-09-11).

- **Input language:** Rust. Kernels are `#[kernel]` functions inside a `#[cuda_module]` module,
  helpers are `#[device]`; host and device code share one source file. `core` only on the device (no
  `std`, no allocator, panics trap).
- **Output:** **PTX** for `sm_70`–`sm_121` (CORRECTED 2026-09-12 by a full read of
  `crates/cuda-target-spec/src/lib.rs` at the pinned SHA: base `sm_70`…`sm_121` including
  `sm_120`/`sm_121`, plus the `a` and `f` variants; the earlier `sm_80`–`sm_100a` was an incomplete
  read; the backend's floor is `sm_80`) (pipeline: Rust MIR → `dialect-mir` (Pliron) → LLVM dialect
  → LLVM IR → `llc` → PTX), and optionally **NVVM IR → LTOIR → cubin** via libNVVM + nvJitLink. It
  is a custom **rustc codegen backend** (`crates/rustc-codegen-cuda`, `-Zcodegen-backend`), not a
  `nvptx64` target in the rust-lang sense and not a CUDA C++ frontend.
  [README](https://github.com/NVlabs/cuda-oxide/blob/6abfaa091e29a6275c1943895bfbc97efa306e98/README.md),
  [supported-features](https://github.com/NVlabs/cuda-oxide/blob/6abfaa091e29a6275c1943895bfbc97efa306e98/cuda-oxide-book/appendix/supported-features.md).
- **Name collision:** crates.io `cuda-oxide` 0.4.0 (2021, a different author, "a rusty wrapper over
  CUDA") is unrelated; `cargo add cuda-oxide` installs the wrong thing. The NVlabs compiler is
  git-distributed.

### 2. Toolchain shape — MEASURED

| requirement       | value                                                                                                                                                                                                                                                                                                                              | evidence                                                                                                                                                                                                                   |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rust              | **nightly-2026-08-28** with `rust-src`, `rustc-dev`, `llvm-tools` (+ clippy, rustfmt, rust-analyzer)                                                                                                                                                                                                                               | [`rust-toolchain.toml`](https://github.com/NVlabs/cuda-oxide/blob/6abfaa091e29a6275c1943895bfbc97efa306e98/rust-toolchain.toml)                                                                                            |
| CUDA              | **CUDA Toolkit 13.0+** at build (incl. cuRAND headers; `nvcc` is used for LTOIR/FFI paths), **CUDA 13.x driver (R580+)** at run — `cuda-bindings` dlopens `libcuda`                                                                                                                                                                | README "Requirements"                                                                                                                                                                                                      |
| C toolchain       | `clang-21` + libclang (bindgen for the host `cuda-bindings` crate); LLVM 21+ `llc` optional (auto-discovers `llc-23/22/21`, prefers the rustup toolchain's `llvm-tools`)                                                                                                                                                           | README                                                                                                                                                                                                                     |
| OS                | Linux (Ubuntu 24.04 tested); Nix flake and devcontainer provided                                                                                                                                                                                                                                                                   | README                                                                                                                                                                                                                     |
| cargo integration | the **`cargo oxide`** subcommand (`build` / `run` / `test` / `inspect` / `pipeline` / `emit-ltoir` / `sanitize` / `debug` / `doctor`); it fetches and builds the codegen backend on first use into `~/.cargo/cuda-oxide/`. Install: `cargo +nightly-2026-08-28 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide` | README "Install"                                                                                                                                                                                                           |
| published crates  | host runtime `cuda-core` 0.3.1, `cuda-async` 0.3.1, `cuda-bindings` 0.3.1 are on crates.io (published from NVlabs/cutile-rs, 2026-09-04); `cuda-device`, `cuda-macros`, `cuda-host`, `cuda-intrinsics`, `cargo-oxide`, `libnvvm-sys` are **git-only**                                                                              | crates.io API                                                                                                                                                                                                              |
| compiler deps     | `pliron` from git at a pinned rev (`deny.toml` `allow-git`); no other git sources                                                                                                                                                                                                                                                  | [`Cargo.toml`](https://github.com/NVlabs/cuda-oxide/blob/6abfaa091e29a6275c1943895bfbc97efa306e98/Cargo.toml), [`deny.toml`](https://github.com/NVlabs/cuda-oxide/blob/6abfaa091e29a6275c1943895bfbc97efa306e98/deny.toml) |

**What this means for the fleet's build pipeline** (MEASURED against
`prover-build-pipeline/Dockerfile.builder`: `vastai/base-image:cuda-13.0.2` + Rust stable 1.88): the
CUDA side already matches; the nightly pin, `cargo-oxide`, and `clang-21` must be added to the
builder for this arm. The bigger shape question is **single-source vs two-phase**: `#[cuda_module]`
embeds the compiled artifact into the _host_ crate's rlib (`.oxart` section; library-crate loading
fixed in cuda-oxide issue #72), which means the host crate is compiled by the cuda-oxide backend
too. For `risc0-groth16-sys`, which bento builds with stable Rust, the low-blast-radius integration
is **two-phase**: build the kernel crate with `cargo oxide` (nightly) into a checked-in PTX/LTOIR
artifact — the same shape as risc0's `kernels/` + `risc0-build-kernel` today — and load it at run
time through the driver API from stable Rust (`cuda-host` runtime-loaded modules / `cuda_launch!`).
UNVERIFIED which loader path works without the backend on the host side; **this is the first thing
s01/4 proves on a CUDA host, before any kernel is ported.** The alternative (compile the whole agent
with `cargo oxide`) puts nightly in bento's build and is not recommended.

### 3. What it can compile from the existing tree — MEASURED, and it reshapes s01/4

**cuda-oxide consumes Rust, not CUDA C++.** Neither bbstark's `cuda/**/*.cu` + `launchers.cuh` nor
risc0's `risc0/groth16-sys/kernels/cuda/*.cu(h)` can be fed to it. There is a bi-directional **LTOIR
device FFI** (`#[device] extern "C"`; Rust kernels calling CUDA C++ device functions compiled by
nvcc, demonstrated with CUB/CCCL and MathDx), so a _hybrid_ — keep sppark's device code, drive it
from Rust kernels — is technically possible, but it keeps nvcc and the C++ templates in the
toolchain and delivers none of the milestone's point.

The honest sizing (MEASURED line counts at the base tag and sppark 0.1.12): risc0's own kernels are
**1,297 lines**, but they are thin wrappers over **sppark's 6,708 lines** of templates — BN254
Montgomery field (`mont_t`, `alt_bn128*`), G1/G2 point arithmetic (`jacobian_t`, `xyzz_t`,
`affine_t`), Pippenger MSM with digit sort and batch addition, mixed-radix NTT kernels and the LDE
coset shift, plus `blst` on the host for the final proof assembly. **A cuda-oxide arm is therefore a
from-scratch Rust implementation of a BN254 MSM + NTT prover stack, not a port.** The bbstark
patterns still apply structurally (kernel modules, declaration vs implementation, pipeline in Rust —
see `BOUNDARY.md` §6), the source language does not.

### 4. Linking and launch from Rust — MEASURED

- `#[cuda_module]` generates a typed launch API: `module.kernel(&stream, LaunchConfig, args…)` (raw
  configs are `unsafe`; `#[launch_contract]` gives a checked `PreparedLaunch`), plus `_async`
  variants returning a lazy `DeviceOperation` (`cuda-async`). Closures and generics monomorphize at
  the use site, including across crates (`cross_crate_kernel`).
- Runtime-loaded modules use `cuda_launch!` / `cuda_launch_async!` (unsafe); LTOIR artifacts load
  through `cuda-host::ltoir`.
- Host runtime = **driver API** (`CudaContext`, `DeviceBuffer<T>`, streams, pinned transfers, HMM
  host-pointer access), shared with cutile-rs; **no `cudart` needed** — relevant because the fleet
  links `-lcudart_static` (D-030) for the existing agent, and the two must coexist. UNVERIFIED:
  driver-API context vs risc0's `cudart` primary context in one process; the existing
  `risc0_zkp::hal::cuda::singleton()` lock must keep serialising GPU use either way.
- Compatibility with the launcher pattern of (3): yes at the structural level — the kernel crate is
  the implementation, the generated launch methods are the declaration surface, and the sequence of
  launches (the pipeline) is plain host Rust.

### 5. Determinism and numerics — MEASURED where marked

This workload is **integer-only** (256-bit modular arithmetic; no floating point anywhere in the
prover), so the classic nvcc-vs-LLVM divergences (`--use_fast_math`, FMA contraction, denormal
flushing) do not apply. What matters instead:

| property                                                        | status                                                                                                                                                   | evidence                                                                                                                                                                                                                     |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 64-bit integer arithmetic                                       | Full                                                                                                                                                     | supported-features "Arithmetic and Casting"                                                                                                                                                                                  |
| `u128` / `i128` arithmetic, shifts, ABI passing, `wrapping_mul` | supported (example `primitive_stress`) — so 64×64→128 products are expressible; the lowering turns ext/mul/shift idioms into `mul.lo` / `mul.hi` / `mad` | [`mir-lower/src/convert/ops/call.rs#L1142`](https://github.com/NVlabs/cuda-oxide/blob/6abfaa091e29a6275c1943895bfbc97efa306e98/crates/mir-lower/src/convert/ops/call.rs#L1142), examples `primitive_stress`, `checked_arith` |
| Rust `asm!`                                                     | **not implemented** (Planned)                                                                                                                            | supported-features "Not Yet Implemented"                                                                                                                                                                                     |
| `ptx_asm!` (cuda-oxide's inline PTX)                            | Partial: `in`/`out`/`inout`, ≤16 outputs, `h r l q f d n C` constraints, `clobber("memory")`                                                             | supported-features "Inline PTX"                                                                                                                                                                                              |
| panics                                                          | trap on device (no unwinding); `gpu_assert!`                                                                                                             | supported-features                                                                                                                                                                                                           |
| compiler correctness                                            | alpha; a differential codegen fuzzer (rustlantis adapter) ships in-tree                                                                                  | `crates/fuzzer`, book "fuzzing-and-differential-testing"                                                                                                                                                                     |

Consequences: (a) sppark's Montgomery multiplication is written as `mad.lo.cc` / `madc.hi.cc` carry
chains in inline PTX; a faithful Rust port either reproduces them with `ptx_asm!` or relies on
`u128` lowering — correctness is unaffected either way, throughput is (s01/6's concern, not
s01/4's). (b) The realistic numeric risk is a **miscompile** in an alpha backend, not a numeric
mode; the s01/5 harness (canonical verifier + mutation arms) is the guard, and running each corpus
case repeatedly, as the plan asks, is the way to surface intermittent divergence. (c) Groth16 proof
randomness (r, s) is sampled on the host in both implementations; proofs are never byte-comparable,
as the plan already states.

### 6. Licensing and provenance — MEASURED

Apache-2.0 (NVlabs). The repository ships `deny.toml` (allowlist: MIT, Apache-2.0, Apache-2.0 WITH
LLVM-exception, BSD-2/3-Clause; `allow-git` only `pliron`), `dependency-licenses.csv`, and
`THIRD_PARTY_NOTICES`. Compatible with `risc0` (Apache-2.0) and with bento (BSL) consuming it.
Provenance points a signing pipeline must handle: `cargo-oxide` **fetches and builds the backend
from git on first run** and `pliron` is a git dependency — pin cuda-oxide by full SHA
(`cargo install --git … --rev 6abfaa091e29a6275c1943895bfbc97efa306e98 cargo-oxide`), cache or
vendor `~/.cargo/cuda-oxide/`, and record the backend SHA next to the kernel artifact in
`build-meta.json`. CUDA Toolkit 13 and the R580+ driver are NVIDIA-licensed as today.

## Verdict

**cuda-oxide is a viable compiler for this stage, under five constraints — and the answer to
question 3 reshapes s01/4 as the issue anticipated: it is a rewrite, not a port.**

1. **Rewrite, not port.** The whole device stack (≈8k lines of C++/CUDA templates today) must be
   authored in Rust: BN254 field, EC, Pippenger MSM, NTT/LDE, coefficient scatter, ChaCha fuzz.
   Correctness against the canonical verifier is the bar; nothing from bbstark or sppark is reusable
   as source.
2. **Toolchain additions in the builder:** nightly-2026-08-28 + `cargo-oxide` + `clang-21`; CUDA 13
   toolkit is already there. Every prover host that runs this arm needs an R580+ driver (UNVERIFIED
   for the production prover).
3. **Pin by SHA and expect breakage** (alpha).
4. **Prove the integration shape first** (two-phase artifact vs single-source) on a CUDA host before
   porting a single kernel; this session cannot (no CUDA userland).
5. **Coexistence with risc0's cudart HAL** in one agent process is UNVERIFIED and is the second
   thing to prove on that host.

If (4) or (5) fails on the host, the fallback that preserves the milestone's substitutability goal
is the LTOIR hybrid (Rust kernels + sppark device functions), which should then be reported as a
finding rather than adopted silently.

## Addendum (C8) — what a GPU-less session could verify, MEASURED 2026-09-12

- The published host runtime `cuda-core` 0.3.1 (with `cuda-bindings` 0.3.1) **type-checks without a
  GPU or toolkit** given only the CUDA 13.0 headers: `cuda-driver-dev`, `cuda-cudart-dev`,
  `cuda-crt`, `cuda-cccl` and `libcurand-dev` (its `wrapper.h` includes `curand.h`), 154 MB
  extracted rootless from NVIDIA's Ubuntu 24.04 repository by `groth16_s01/scripts/cuda-headers.sh`.
  `cuda-bindings` needs `CUDA_HOME` at build time and dlopens `libcuda` at run time — so the fleet's
  builder image needs nothing new for the host side.
- What that buys: `risc0-groth16-cuda` (host side of s01/4) and the `cuda-oxide` backend are
  compiled in CI on `ubuntu-latest`; only the device module (`groth16_s01/cuda-kernels`,
  `cargo oxide`) and every launch remain for the CUDA host (E2).
- What it does not settle: questions 4 and 5 of the verdict above (artifact shape on a real build;
  cudart/driver-API coexistence in one agent process) — unchanged, UNVERIFIED.

## Addendum (C10) — the device template verified against the pinned tree, MEASURED 2026-09-12

A read-only review of the pinned repository (every claim with file and line in the C10 PR) settled
the template's two UNVERIFIED points and corrected one statement above:

- **Targets:** `sm_120` and `sm_121` ARE in the pinned target list
  (`crates/cuda-target-spec/src/lib.rs`), selectable with `cargo oxide build --arch sm_120` (or
  `CUDA_OXIDE_TARGET`, or `.cargo/cuda-oxide.toml`). NVIDIA's compatibility rules (nvcc guide
  §4.2.9, PTX ISA 9.4 §11.1.2, Blackwell compatibility guide): base `sm_100` PTX JIT-loads on
  `sm_120`; `sm_100a` and `sm_100f` do not. So: `--arch sm_120` for the fleet's nodes,
  `--arch sm_89` for an RTX 4090.
- **Kernel parameters:** raw pointers (one `.u64` param each), `u32` (`.u32`), and by-value
  `#[repr(C)]` structs (one byval param) are accepted; slices lower to two `.u64` params. The ABI's
  `(ptr, len: u32)` pairs and the host's per-parameter slots therefore match 1:1. The 32-byte
  by-value `Fr` is accepted by the model but exercised by no example — runtime UNVERIFIED.
- **Thread index:** `thread::index_1d().get()` inside a `#[kernel]` body (macro-rewritten there; a
  free helper is a diagnostic); no `intrinsics` module exists.
- **Build product:** `cargo oxide build` (no `--release` flag) writes `<crate_name>.ptx` beside the
  crate (or into `CUDA_OXIDE_PTX_DIR`) and embeds an `.oxart` section in the rlib/bin; a separate
  host loads the `.ptx` — `ModuleSource::Ptx`, i.e.
  `RISC0_GROTH16_CUDA_MODULE=groth16_s01/cuda-kernels/risc0_groth16_cuda_kernels.ptx`.
  `artifact_bundles_from_binary_path` parses object bytes, not rlib archives.
- **Toolchain:** the nightly is mandatory for any crate depending on `cuda-device`/`cuda-macros`
  (`#![feature(f16)]`, proc-macro features), with or without the backend; a device-only crate
  `cargo check`s on that nightly with no CUDA toolkit; `cuda-macros` must be
  `default-features = false` (its `host` feature emits `cuda_host` loaders that need `cuda.h`).

## Addendum (C13) — the device module compiled and validated without a GPU, MEASURED 2026-09-12

- **Toolchain, rootless, on the session box:** nightly-2026-08-28 (rust-src, rustc-dev, llvm-tools),
  `cargo-oxide` installed from the pinned commit (`cargo +nightly install --git … --root ~`, host
  target pinned — the home cargo config's musl default breaks the install otherwise),
  `cargo oxide setup` (the backend, ≈ 2 minutes), the CUDA 13.0 headers and the `cuda-nvcc` package
  (ptxas) from NVIDIA's repository by `groth16_s01/scripts/cuda-headers.sh`. `cargo oxide doctor`
  reports libNVVM, nvJitLink and libdevice missing, which matters only for libdevice math; these
  kernels are integer-only.
- **What compiled:** `groth16_s01/cuda-kernels` type-checks, is clippy-clean, and
  `cargo oxide build --arch sm_89` / `--arch sm_120` each emit `risc0_groth16_cuda_kernels.ptx` (688
  KB, `.version 7.8`), settling verdict item 4: the two-phase shape (device crate built alone, host
  loads the `.ptx`) works.
- **What the PTX shows (`groth16_s01/scripts/ptx-check.sh`):** ten `.visible .entry` kernels under
  the ABI's names; parameters as the ABI says — `.u64 .ptr` per pointer, `.u32` per length or
  scalar, one `.align 8 .b8 [32]` byval param for the by-value `Fr` (question 3 of the template
  review, settled); `ptxas` assembles the `sm_89` module for `sm_89` and `sm_120` and the `sm_120`
  module for `sm_120`.
- **One device limitation, fixed:** a derived `PartialEq` on `[u64; 4]` lowers to the `raw_eq`
  intrinsic, "not yet supported on the device"; `risc0-groth16-core`'s field types now compare
  limb-wise (I-G16-023). Everything else the kernels reach — 64×64→128 products, the `bool` in
  `Affine`, the generic `bucket_sum<F>`, cross-crate bodies — lowered without change.
- **Still UNVERIFIED:** the launches (driver load, per-kernel agreement, coexistence, timing) — the
  CUDA host's first hour, from the `ptx-c13` release's PTX.

## Addendum (C15) — the first hardware run, MEASURED 2026-09-12

- **Device:** NVIDIA GeForce RTX 5080 (`sm_120`, the class of the fleet's prover nodes), driver
  580.95.05 (open kernel module), device nodes world-readable in the session container; no driver
  user-space library in the container — `libnvidia-compute-580` at the module's exact version from
  NVIDIA's CUDA repository, extracted rootless, put on `LD_LIBRARY_PATH`; `cuInit` 0, driver
  API 13000.
- **Result:** `groth16-cuda-kernel-check` against the `ptx-c13` release — every kernel, both MSMs
  and the fixture proof (byte-identical to the core prover) agree, for the `sm_120` module and for
  the `sm_89` module through the driver's JIT. Nothing changed between the GPU-less build (C13) and
  the launch: verdict items 4 (artifact shape) and the template review's open points are settled by
  hardware; item 5 (coexistence with the cudart HAL in one process) remains for the
  canonical-beside- the-arm build.
