#!/usr/bin/env python3
"""Check an emitted PTX module against the CUDA arm's ABI (risc0_groth16_oxide::abi) without a GPU.

For every kernel the ABI names, the PTX must export a `.visible .entry` whose `.param` list has the
ABI's arity and kinds: a raw pointer or a `u64` is one `.u64` param, a `u32` one `.u32` param, a
by-value `Fr` one 32-byte byval param (`.align 8 .b8 name[32]`). The host stages exactly these per
parameter (risc0-groth16-cuda `CudaProver::run`), so a mismatch here is a launch failure or a silent
misread there.

    ptx-abi-check.py <module.ptx>
"""
import re
import sys

# kind letters: p = pointer (.u64), u = u32 (.u32), F = Fr by value (32-byte byval param)
ABI = {
    "scatter_group": "pupupupu",  # out, n, coeffs, len, starts, len, witness, len
    "pointwise_mul": "pupupu",  # out, n, a, len, b, len
    "pointwise_mul_sub": "pupupupu",  # out, n, a, len, b, len, c, len
    "pointwise_scale": "pupupuF",  # out, n, a, len, k, len, n_inv
    "bit_reverse": "pupuu",  # out, n, a, len, lg_n
    "ntt_stage": "pupuupuu",  # out, n, a, len, len, twiddles, len, stride
    "digits": "pupuuu",  # out, n, scalars, len, window, w
    "digits_all": "pupuu",  # out, n_total, scalars, len, w
    "bucket_sum_g1": "pupupupu",  # out, n, points, len, order, len, starts, len
    "bucket_sum_g2": "pupupupu",
}


def parse(ptx: str):
    """{entry name: [(kind, declaration)]} from `.visible .entry name( .param ..., ... )`."""
    out = {}
    # the parameter list holds no ")" — so `[^)]*` cannot run past the entry into its body
    for m in re.finditer(r"\.visible\s+\.entry\s+(\w+)\s*\(([^)]*)\)", ptx, re.S):
        name, params = m.group(1), m.group(2)
        kinds = []
        for decl in [d.strip() for d in params.split(",") if d.strip()]:
            if re.search(r"\.b8\s+\w+\[32\]", decl):
                kinds.append("F")
            elif ".u64" in decl:
                kinds.append("p")
            elif ".u32" in decl:
                kinds.append("u")
            else:
                kinds.append("?")
            kinds[-1] = (kinds[-1], decl)
        out[name] = kinds
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    ptx = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    entries = parse(ptx)
    target = re.search(r"^\.target\s+(\S+)", ptx, re.M)
    version = re.search(r"^\.version\s+(\S+)", ptx, re.M)
    print(f"ptx: .version {version.group(1) if version else '?'} .target {target.group(1) if target else '?'}; "
          f"{len(entries)} entries")
    bad = 0
    for name, want in ABI.items():
        got = entries.get(name)
        if got is None:
            # cuda-oxide may mangle; accept a unique entry whose name ends with the kernel name
            cands = [e for e in entries if e == name or e.endswith("_" + name) or e.endswith(name)]
            if len(cands) == 1:
                got = entries[cands[0]]
                name_note = f" (as `{cands[0]}`)"
            else:
                print(f"MISSING  {name}: no .entry (candidates: {cands})")
                bad += 1
                continue
        else:
            name_note = ""
        kinds = "".join(k for k, _ in got)
        if kinds == want:
            print(f"ok       {name}{name_note}: {kinds}")
        else:
            print(f"MISMATCH {name}{name_note}: got {kinds}, want {want}")
            for k, decl in got:
                print(f"           {k}: {decl}")
            bad += 1
    if bad:
        print(f"ptx-abi-check: FAILED ({bad} of {len(ABI)} kernels)", file=sys.stderr)
        return 1
    print(f"ptx-abi-check: ok (all {len(ABI)} kernels match the ABI)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
