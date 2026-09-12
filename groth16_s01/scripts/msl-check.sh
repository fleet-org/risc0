#!/usr/bin/env bash
# Syntax pass over the Metal arm's shaders on a machine without Metal: concatenate the sources the
# way `risc0_groth16_metal::MSL_SOURCE` does and compile them as C++14 against a stub <metal_stdlib>
# (risc0/groth16-metal/msl-host/, the same stub the crate's host backend compiles the shaders
# with; force-included: consts.metal precedes the shader's own #include, as MSL allows), with
# MSL's built-in type names poisoned so that using one as an identifier is an error.
# Catches: C++ syntax errors, undeclared names, and reserved-type-name identifiers (I-G16-019).
# Does not catch: anything semantic (address spaces, attribute meaning, arithmetic).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
CXX=${CXX:-g++}
stub=risc0/groth16-metal/msl-host
parse() {
  "$CXX" -std=c++14 -x c++ -fsyntax-only -isystem "$stub" -include "$stub/metal_stdlib" "$@"
}
if [ "${1:-}" = --self-test ]; then
  # the check must have teeth: a poisoned name used as an identifier is an error
  if printf 'uint half = 1;\n' | parse - 2> /dev/null; then
    echo "msl-check self-test: FAILED — 'uint half = 1;' was accepted" >&2
    exit 1
  fi
  echo "msl-check self-test: ok (a reserved MSL type name used as an identifier is rejected)"
  exit 0
fi
src=$(mktemp --suffix=.cpp)
trap 'rm -f "$src"' EXIT
{
  cat risc0/groth16-metal/src/consts.metal
  echo
  cat risc0/groth16-metal/src/kernels.metal
} > "$src"
if parse -Wno-attributes -Wno-unused "$src" 2> "$src.err"; then
  echo "msl-check: ok (consts.metal + kernels.metal parse as C++14 against the stub;" \
    "no poisoned name used)"
else
  echo "msl-check: FAILED" >&2
  sed "s#$src#kernels.metal(+consts)#g" "$src.err" | head -20 >&2
  rm -f "$src.err"
  exit 1
fi
rm -f "$src.err"
