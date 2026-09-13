# The CUDA arm's device module (GROTH16 s01/4)

The device half of the CUDA arm. It compiles in two stages on the pinned cuda-oxide toolchain
([`CUDA_OXIDE_PIN.md`](../CUDA_OXIDE_PIN.md) §2 and its addenda): `cargo check`/`clippy` on
nightly-2026-08-28 need neither a toolkit nor a GPU (MEASURED on the GPU-less session box and in
CI's `device-module` job), and `cargo oxide build` emits `risc0_groth16_cuda_kernels.ptx` through
cuda-oxide's codegen backend and the toolchain's `llc`, which `ptxas` (from the `cuda-nvcc` package,
rootless) assembles for the target GPU — still no GPU needed. Only the launches need one. This
directory is its own workspace, excluded from the repo's.

```sh
# STAGE 1 — anywhere with the toolchain, no GPU (the session box, CI): compile to PTX and validate
cargo install --git https://github.com/NVlabs/cuda-oxide \
  --rev 6abfaa091e29a6275c1943895bfbc97efa306e98 cargo-oxide        # the CLI (nightly; host target!)
eval "$(groth16_s01/scripts/cuda-headers.sh "$HOME/cuda")"; export PATH="$CUDA_BIN:$PATH"  # cuda.h, ptxas
cd groth16_s01/cuda-kernels
cargo oxide doctor                            # libNVVM/libdevice "missing" is fine: no libdevice math here
cargo oxide setup                             # builds the codegen backend once (minutes)
cargo oxide build --arch sm_89                # RTX 4090; `--arch sm_120` for the fleet's prover nodes
cd ../.. && groth16_s01/scripts/ptx-check.sh  # kernels + parameter layouts vs the ABI; ptxas for sm_89, sm_120

# STAGE 2 — on the CUDA host: launch (on a shared device, check it is quiet first; see HARNESS.md)
groth16_s01/scripts/gpu-free.py 5 2 || echo "not quiet: a production-sized run can fail either tenant"
export RISC0_GROTH16_TIMING=1                # per-phase wall-clock on stderr
export RISC0_GROTH16_CUDA_MODULE=$PWD/groth16_s01/cuda-kernels/risc0_groth16_cuda_kernels.ptx
cargo run -p risc0-groth16-cuda --bin groth16-cuda-kernel-check   # every kernel, the MSM, a fixture proof
RISC0_GROTH16_BACKEND=cuda-oxide cargo test -p risc0-groth16-sys --features cuda-oxide
cargo run -p groth16-s01-harness --features cuda,cuda-oxide -- run <case> cuda-oxide --control canonical
```

Target choice: `sm_120` and `sm_121` are in the pinned target list; base `sm_100` PTX also JIT-loads
on `sm_120`, while `sm_100a`/`sm_100f` do not (NVIDIA's compatibility rules). Selection is `--arch`,
then `CUDA_OXIDE_TARGET`, then `.cargo/cuda-oxide.toml`.

What stage 1 proved on the GPU-less session box (MEASURED, C13): the crate type-checks and is
clippy-clean on the pinned nightly; `cargo oxide build` emits the PTX (`.version 7.8`, 688 KB) for
`sm_89` and for `sm_120`; every one of the ten kernels is a `.visible .entry` under its ABI name
with the ABI's parameter list — a raw pointer as `.u64 .ptr`, a `u32` as `.u32`, the by-value `Fr`
of `pointwise_scale` as one `.align 8 .b8 [32]` param, exactly what the host stages per parameter —
and `ptxas` 13.0 assembles the `sm_89` module for `sm_89` and for `sm_120` (451 and 544 KB of SASS)
and the `sm_120` module for `sm_120`. One device-side limitation surfaced and was fixed in
`risc0-groth16-core`: a derived `PartialEq` on `[u64; 4]` lowers to the `raw_eq` intrinsic, which
the backend rejects; the field types compare limb-wise now. The PTX files are attached to the fork's
`ptx-c13` release (SHA-256 in the release notes), so the CUDA host can start at stage 2 without the
toolchain.

What to expect from `kernel-check`: it names the first kernel or composite step (`msm (g1)`,
`msm (g2)`, `proof (fixture)`) whose output differs from `risc0_groth16_oxide::kernels`, on the
shared cases of `risc0_groth16_oxide::check` — the same cases the Metal arm's check runs.

First hardware run (MEASURED, C15, on an RTX 5080 with the 580.95.05 driver — the fleet's `sm_120`
class): `groth16-cuda-kernel-check` against the `ptx-c13` release passed 13/13 for the `sm_120`
module and for the `sm_89` module through the driver's JIT, unchanged from what was built without a
GPU. The container had only the device nodes; `libnvidia-compute-580` at the module's exact version,
fetched rootless from NVIDIA's repository onto `LD_LIBRARY_PATH`, was all `cuda-core` needed.

Since C21 the module has twelve kernels (`jacobian_sum_g1`/`_g2` — one level of the bounded-chain
reduction above the bucket sums, `pipeline::plan_ranges`) and `kernel-check` reports 15 checks; the
`ptx-c21` release carries the PTX (1.18 MB per target: the full Jacobian addition inlined twice).
`groth16-cuda-msm-bench` times the bucket-sum kernel alone under a chosen range layout (`--chunk C`,
`--tiny`, `--check`) — the experiment that found the MSM's critical path (HARNESS.md).

Still UNVERIFIED after that: cudart/driver-API context coexistence in one bento process (needs the
canonical path built with nvcc beside the arm) and timing against the canonical kernels (the same
build); both are the next hardware step.
