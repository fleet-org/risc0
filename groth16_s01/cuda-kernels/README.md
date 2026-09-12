# The CUDA arm's device module (GROTH16 s01/4)

Built **only on a CUDA host** with the pinned cuda-oxide toolchain
([`CUDA_OXIDE_PIN.md`](../CUDA_OXIDE_PIN.md) §2 and the C10 addendum: nightly-2026-08-28, CUDA 13
toolkit, R580+ driver, `cargo oxide`). This directory is its own workspace, excluded from the
repo's; nothing here is compiled by CI. Every statement below about cuda-oxide was verified by
reading the pinned tree (file and line in the C10 PR); every statement about running it is
UNVERIFIED until a CUDA host has run it.

```sh
# on the CUDA host, once:
cargo install --git https://github.com/NVlabs/cuda-oxide --rev 6abfaa091e29a6275c1943895bfbc97efa306e98 cargo-oxide
cd groth16_s01/cuda-kernels
cargo oxide doctor
cargo check                                   # the pinned nightly type-checks this crate without a toolkit
cargo oxide build --arch sm_89                # RTX 4090; `--arch sm_120` for the fleet's prover nodes
# product: risc0_groth16_cuda_kernels.ptx beside this file (or in $CUDA_OXIDE_PTX_DIR); the host loads it
export RISC0_GROTH16_CUDA_MODULE=$PWD/risc0_groth16_cuda_kernels.ptx
cd ../..
cargo run -p risc0-groth16-cuda --bin groth16-cuda-kernel-check   # every kernel, the MSM, a fixture proof
RISC0_GROTH16_BACKEND=cuda-oxide cargo test -p risc0-groth16-sys --features cuda-oxide
cargo run -p groth16-s01-harness --features cuda,cuda-oxide -- run <case> cuda-oxide --control canonical
```

Target choice: `sm_120` and `sm_121` are in the pinned target list; base `sm_100` PTX also JIT-loads
on `sm_120`, while `sm_100a`/`sm_100f` do not (NVIDIA's compatibility rules). Selection is `--arch`,
then `CUDA_OXIDE_TARGET`, then `.cargo/cuda-oxide.toml`.

What to expect from `kernel-check`: it names the first kernel or composite step (`msm (g1)`,
`msm (g2)`, `proof (fixture)`) whose output differs from `risc0_groth16_oxide::kernels`, on the
shared cases of `risc0_groth16_oxide::check` — the same cases the Metal arm's check runs.

Still UNVERIFIED until run (from the review; each is a small fix, and the ABI does not move for
them):

1. that a standalone **library** crate's `cargo oxide build` writes the `.ptx` (the pinned
   standalone examples are binaries; the backend's code path is crate-type-agnostic);
2. that the 32-byte by-value `Fr` parameter of `pointwise_scale` is laid out as one byval `.param`
   (examples cover `u32`-sized and packed structs only) — fallback: pass `n_inv` as a
   `(ptr, len = 1)` slice on both halves;
3. that every reachable `risc0-groth16-core` function lowers (64×64→128 products, the `bool` in
   `Affine`, the generic `bucket_sum<F>`), and that cross-crate bodies are collected without
   `#[device]` (they are, by `-Zalways-encode-mir`, per the pinned source);
4. that the emitted `.version`/`.target` loads on the R580 driver, and that the PTX kernel names
   equal `abi::KERNELS` (the macro strips its reserved prefix — source-read only);
5. from the pin document, unchanged: cudart/driver-API context coexistence in one agent process.
