# GROTH16 s01/5 — the differential correctness harness

**Issue:** [#7@fleet-org/risc0](https://github.com/fleet-org/risc0/issues/7) · **Crate:** `groth16_s01/harness` (`groth16-s01-harness`) · **Oracle:** the canonical upstream verifier, unmodified — `risc0_zkvm::Groth16Receipt::verify_integrity_with_context` with the default `VerifierContext` (the call bento's SNARK task makes on its own output, `[BENTO-SNARK-005]`), which wraps `risc0_groth16::Verifier`.

## The correctness argument, as the harness checks it

Groth16 proofs are randomized, so nothing is compared byte for byte. Per corpus case and per arm the harness establishes, three-state (`yes` / `no` with evidence / `not run` with reason — never collapsing "could not look" into "nothing there"):

| # | assertion | how |
|---|---|---|
| 1 | the rewrite's seal verifies | `oracle::verify(seal, claim)` where `claim` is the identity_p254 receipt's claim derived from the stage input |
| 2 | it verifies against the same public inputs as the canonical output | the canonical output's claim digest equals the derived claim's digest (the five public inputs are a function of the claim digest, the control root and the BN254 identity control id) |
| 3 | the canonical output still verifies in the same run (control) | `oracle::verify` on the corpus's `canonical.*.bincode` seal and claim |
| 4 | a corrupted seal is rejected (mutation) | one bit flipped in each of π_A, π_B, π_C; all three variants must be rejected |

Mutation arms beyond assertion 4 (`mutations.rs`):

| arm | what must happen | how the harness knows it is not fooling itself |
|---|---|---|
| cross-claim | the rewrite's seal verified against another case's claim is rejected | skipped with a reason when the other case has the same claim digest |
| canonical answers | the rewrite is disabled, the control implementation answers, the suite still passes | the registry's *confirmed* kind is recorded (`RunResult::kind`), and a registry without the rewrite must refuse to select it — a substitution cannot be silent |
| malformed input | the truncated stage input is rejected | rejection happens before the boundary (deserialization), so both implementations fail identically by construction; the error class is recorded |

## Running it

```
# a synthetic stage input (the in-tree loop guest proven on this host; not a corpus case)
groth16-s01-harness synth cases/synthetic-loop-0 0

# one case, rewrite = <kind>, control = canonical (default) or --control reference
groth16-s01-harness run cases/<case> <artifacts-dir> <kind> [--control <kind>] [--other cases/<other>] [--report reports/<case>.md]
```

`<artifacts-dir>` holds the rzup `risc0-groth16` v0.1.0 set (`stark_verify_final.zkey`, `stark_verify_graph.bin`, `preprocessed_coeffs.bin`, `fuzzed_msm_results.bin`). The harness derives the boundary input exactly as the canonical path does (`identity_p254` → seal → `to_json` → circom witness), runs the selected backend through `risc0_groth16_sys::Registry`, and judges with the oracle. Build with `--features cuda` on a CUDA host to make the canonical backend available.

## Results

_Filled in per run; each row is MEASURED on the host named._

### Synthetic case, reference arm, CRCS host (no GPU)

_pending_

### Frozen corpus (`corpus-groth16-s01-v1`)

_pending — needs the corpus (E1 on #1)._

## What this harness does not cover

Seal encoding for the chain, selector routing through the verifier router, and SetVerifier Merkle inclusion — those are s01/6's on-chain oracle. Timing here is the harness host's wall-clock, reported for information, not as a performance claim.
