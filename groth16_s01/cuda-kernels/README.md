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

# STAGE 2 — on the CUDA host: launch
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

What to expect from `kernel-check`What to expect from `kernel-check`: it names the first kernel or
composite step (`msm (g1)`, `msm (g2)`, `proof (fixture)`) whose output differs from
`risc0_groth16_oxide::kernels`, on the shared cases of `risc0_groth16_oxide::check` — the same cases
the Metal arm's check runs.

Still UNVERIFIED until a CUDA host runs it — the launches themselves:

1. that the driver (R580+) loads this `.version 7.8` PTX and `cuLaunchKernel` accepts each entry's
   parameters as staged (the layouts match by inspection; the launch is the test);
2. that every kernel agrees with the Rust bodies on the shared cases (`groth16-cuda-kernel-check`)
   and the fixture proof matches the core prover — the arithmetic lowered by an alpha backend;
3. from the pin document, unchanged: cudart/driver-API context coexistence in one agent process;
4. wall-clock against the canonical kernels (s01/5 with `--control canonical`).
