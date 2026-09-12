#!/usr/bin/env bash
# Validate the CUDA device module's PTX without a GPU: the kernels and their parameter layouts
# against the ABI (ptx-abi-check.py), then assembly by `ptxas` for each target GPU (the fleet's
# sm_120 nodes and the RTX 4090's sm_89 by default) — a codegen or ISA error surfaces here, not on
# the first launch.
#
#   ptx-check.sh [module.ptx] [sm_XX…]        (defaults: groth16_s01/cuda-kernels/risc0_groth16_cuda_kernels.ptx; sm_89 sm_120)
#
# `ptxas` comes with the cuda-nvcc package (groth16_s01/scripts/cuda-headers.sh fetches the headers;
# the same rootless deb extraction gives the compiler: see CUDA_OXIDE_PIN.md, C13 addendum).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ptx=${1:-groth16_s01/cuda-kernels/risc0_groth16_cuda_kernels.ptx}
shift || true
archs=("$@")
[ ${#archs[@]} -gt 0 ] || archs=(sm_89 sm_120)
[ -f "$ptx" ] || {
  echo "ptx-check: $ptx not found (run 'cargo oxide build --arch <sm>' in groth16_s01/cuda-kernels)" >&2
  exit 1
}
python3 groth16_s01/scripts/ptx-abi-check.py "$ptx"
command -v ptxas > /dev/null || {
  echo "ptx-check: ptxas not on PATH (cuda-nvcc package); ABI check passed, assembly skipped" >&2
  exit 1
}
for arch in "${archs[@]}"; do
  out=$(mktemp)
  if ptxas -arch="$arch" -O3 -o "$out" "$ptx" 2> "$out.err"; then
    echo "ptxas: ok for $arch ($(stat -c %s "$out") bytes of SASS)"
  else
    echo "ptxas: FAILED for $arch" >&2
    head -20 "$out.err" >&2
    rm -f "$out" "$out.err"
    exit 1
  fi
  rm -f "$out" "$out.err"
done
echo "ptx-check: ok (${archs[*]})"
