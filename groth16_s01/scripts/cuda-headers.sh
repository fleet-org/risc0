#!/usr/bin/env bash
# CUDA 13 headers and the PTX assembler without a toolkit or root: the driver-API, runtime, CRT, CCCL,
# cuRAND dev and nvcc
# packages from NVIDIA's Ubuntu 24.04 repository, extracted with dpkg-deb into <dest>. Enough for
# `cuda-bindings` (bindgen over cuda.h + curand.h), hence for type-checking risc0-groth16-cuda
# on a machine with no GPU — CI and the CRCS both use it. Prints the two variables to export.
#
#   eval "$(groth16_s01/scripts/cuda-headers.sh "$HOME/cuda-headers")"
#   groth16_s01/scripts/cuda-headers.sh "$RUNNER_TEMP/cuda" >> "$GITHUB_ENV"
#
# CUDA_HEADERS_VERSION selects the package series (default 13-0, the fleet's builder image).
set -euo pipefail
dest=${1:?usage: cuda-headers.sh <dest-dir>}
series=${CUDA_HEADERS_VERSION:-13-0}
repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
mkdir -p "$dest/debs" "$dest/tree"
[ -f "$dest/Packages.gz" ] || curl -sSfL -o "$dest/Packages.gz" "$repo/Packages.gz"
pick() { # newest Filename of a package
  zcat "$dest/Packages.gz" | awk -v p="$1" '
    $0 == "Package: " p { f = 1 }
    f && /^Version:/  { v = $2 }
    f && /^Filename:/ { fn = $2 }
    f && /^$/         { print v, fn; f = 0 }' | sort -V | tail -1 | cut -d' ' -f2
}
# cuda-nvcc carries ptxas (and nvcc), which cargo-oxide's doctor and ptx-check.sh use; ≈ 30 MB more
pkgs=(cuda-driver-dev cuda-cudart-dev cuda-crt cuda-cccl libcurand-dev cuda-nvcc)
for name in "${pkgs[@]}"; do
  p=$name-$series
  f=$(pick "$p")
  [ -n "$f" ] || {
    echo "cuda-headers: no package $p in $repo" >&2
    exit 1
  }
  deb="$dest/debs/$(basename "$f")"
  [ -f "$deb" ] || curl -sSfL -o "$deb" "$repo/${f#./}"
  dpkg-deb -x "$deb" "$dest/tree"
done
root=$(find "$dest/tree" -maxdepth 4 -type d -name "cuda-${series/-/.}*" | head -1)
[ -f "$root/include/cuda.h" ] || [ -f "$root/targets/x86_64-linux/include/cuda.h" ] || {
  echo "cuda-headers: cuda.h not found under $root" >&2
  exit 1
}
echo "CUDA_HOME=$root"
echo "CUDA_TOOLKIT_PATH=$root"
# ptxas lives beside nvcc; callers add it to PATH (a GITHUB_ENV file cannot extend PATH)
echo "CUDA_BIN=$root/bin"
