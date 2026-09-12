# The CUDA arm's device module (GROTH16 s01/4)

Built **only on a CUDA host** with the pinned cuda-oxide toolchain
([`CUDA_OXIDE_PIN.md`](../CUDA_OXIDE_PIN.md) §2: nightly-2026-08-28, CUDA 13 toolkit, R580+ driver,
`cargo oxide`). This directory is excluded from the workspace, so nothing here is compiled by CI.

```sh
# on the CUDA host, once:
cargo install --git https://github.com/NVlabs/cuda-oxide --rev 6abfaa091e29a6275c1943895bfbc97efa306e98 cargo-oxide
cd groth16_s01/cuda-kernels && cargo oxide doctor && cargo oxide build --release
# the product carries the PTX (an .oxart artifact bundle); point the host at it:
export RISC0_GROTH16_CUDA_MODULE=<path printed by cargo oxide build>   # or a .ptx / .cubin
cd ../.. && groth16_s01/scripts/cuda-headers.sh "$HOME/cuda-headers" > /dev/null   # only if no toolkit
cargo run -p risc0-groth16-cuda --bin groth16-cuda-kernel-check          # every kernel vs the Rust bodies
RISC0_GROTH16_BACKEND=cuda-oxide cargo test -p risc0-groth16-sys --features cuda-oxide
cargo run -p groth16-s01-harness --features cuda -- run <case> cuda-oxide --control canonical
```

What to expect: `kernel-check` names the first kernel whose output differs from
`risc0_groth16_oxide::kernels` on a 64-element input. The two UNVERIFIED points in `src/lib.rs`
(the thread-index intrinsic; raw-pointer kernel parameters) fail at `cargo oxide build`, not at
run time, and are one-line fixes; the ABI (`risc0_groth16_oxide::abi`) does not move for them.
