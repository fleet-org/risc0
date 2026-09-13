# GROTH16 s01 — the device memory budget

This is the written memory budget the `prover-expert` skill's substrate recommends (DEF-PRV-007,
harvested from fleet-org/bbstark): a proof's device footprint as a closed-form function of the
circuit's dimensions, so **resident vs streaming is a computed decision against the device's free
memory, not a hand-set flag**. The formulas live in code — `risc0_groth16_oxide::budget` — with unit
tests that pin them to the measured footprint, so the numbers here cannot drift from the
implementation.

## Element sizes (from the ABI, `risc0_groth16_oxide::abi`)

| element             | bytes | element            | bytes |
| ------------------- | ----: | ------------------ | ----: |
| `Fr` (scalar)       |    32 | `Jacobian<Fp>` G1  |    96 |
| `Affine<Fp>` G1     |    72 | `Jacobian<Fp2>` G2 |   192 |
| `Affine<Fp2>` G2    |   136 | `GroupedCoeff`     |    40 |
| `u32` (digit/index) |     4 |                    |       |

## The three residency tiers (the bbstark taxonomy)

- **Persistent** — uploaded once, lives the whole proof: the resident zkey (NTT tables, grouped
  coefficients, the five point sets). Present only on the resident path.
- **Ephemeral** — one phase's working buffers, freed before the next (the four transform buffers; an
  MSM's digits/order/sums).
- **Scratch** — kernel-local, inside a phase.

## Closed-form budget (`n` = `num_vars`, `N` = `domain`, `w` = 12 → 22 windows, 4095 buckets)

| buffer                          | tier       | bytes                                       |
| ------------------------------- | ---------- | ------------------------------------------- |
| NTT tables (fwd+inv+shift)      | persistent | `2·N·32`                                    |
| grouped coefficients + starts   | persistent | `coeffs·40 + 2·N·4`                         |
| point sets a,b1,c,h (G1)+b2(G2) | persistent | `(3n − pub − 1 + N)·72 + n·136`             |
| transform (four buffers)        | ephemeral  | `4·N·32`                                    |
| one MSM's scratch (G2)          | ephemeral  | `N·32 + 2·22·N·4 + 22·4095·4 + 22·4095·192` |
| largest point set (streamed)    | ephemeral  | `max(N·72, n·136)`                          |
| witness                         | ephemeral  | `n·32`                                      |

- **Resident set** = tables + coefficients + point sets.
- **Resident peak** = resident set + max(transform, MSM scratch) + witness.
- **Streaming peak** = max(transform + tables, largest point set + MSM scratch) — nothing persists.

## The production circuit (MEASURED dims: `n` = 5,635,930, `N` = 2²³, `pub` = 5, `coeffs` = 29,098,147)

| quantity             | budget (GiB) | corroboration                                        |
| -------------------- | -----------: | ---------------------------------------------------- |
| NTT tables           |         0.50 | BOUNDARY §7 ≈ 0.8 (incl. `n_inv`, alignment)         |
| grouped coefficients |         1.15 | 29,098,147 × 40 B = 1.16 GB                          |
| point sets           |         2.41 | BOUNDARY §7 ≈ 2.6                                    |
| **resident set**     |     **4.06** | `ResidentZkey::device_bytes` ≈ 4.6 (starts, padding) |
| transform scratch    |         1.00 | four-buffer, C19                                     |
| one MSM scratch (G2) |         1.64 | digits + order dominate                              |
| **resident peak**    |     **5.87** | fits an 8 GB card                                    |
| **streaming peak**   |     **2.36** | MEASURED ≈ 2 GB (C18/C19, lowest free 1.15 GiB)      |

## VRAM-aware path selection (`Budget::choose(free, 0.9)`)

`RISC0_GROTH16_RESIDENT` is tristate: `0`/`false`/`off` forces streaming, any other value forces
resident, and **unset is AUTO** — the backend reads the device's free memory (`cuMemGetInfo`), keeps
10 % as headroom, and picks resident if its peak fits, else streaming if its peak fits, else fails
fast (the proof does not fit even streaming). The chosen decision is logged on stderr.

**On a shared device the reading is not enough (C24).** The free line swings as a co-tenant
allocates, so AUTO takes the MINIMUM over several readings (`min_free_device_bytes`), and — because
even that can be stale between the probe and the upload — a resident attempt that hits an
out-of-memory FALLS BACK to streaming for that call (the bbstark P5.2 adaptive pattern): the cache
is evicted to free the partial resident set, the zkey is re-read, and the proof completes streaming.
So AUTO produces a verified proof whatever the co-tenant does: it runs resident when the device
really holds it, and degrades to streaming when it does not, without an operator flag.

| device                       | free (GiB) | usable (0.9) | chosen       |
| ---------------------------- | ---------: | -----------: | ------------ |
| RTX 5080 16 GB, idle         |         16 |         14.4 | resident     |
| RTX 5080, co-tenant at peak  |          3 |          2.7 | streaming    |
| RTX 4090 24 GB               |         24 |         21.6 | resident     |
| fleet `sm_120` node, 80 GB   |         80 |         72.0 | resident     |
| Apple M3 Max, 128 GB unified |        128 |        115.2 | resident     |
| an 8 GB card                 |          8 |          7.2 | resident     |
| ≤ ~2.6 GiB free              |          2 |          1.8 | does not fit |

So on any unshared card ≥ 8 GB the arm runs resident (fast, upload-once); on the session's shared
RTX 5080, whose co-tenant peaks near 12 GiB, AUTO picks streaming exactly when the free window is
too small for the resident set — the decision the operator used to make with the env flag, now
computed.

## Not yet borrowed from bbstark (the remaining gaps, ranked)

These are the MEDIUM-value items from the bbstark harvest (`prover-expert` DEF-PRV-007), left as
follow-ups so this change stays additive and reviewable:

1. A **bump-arena allocator with checkpoint/restore** (one `cudaMalloc`, `checkpoint()` after the
   resident upload, `restore()` after each proof) to remove per-proof `cudaMalloc`/`cudaFree` churn.
2. **MSM scratch reused across the five MSMs** (one buffer sized to the largest) plus an **adaptive
   OOM → k-way chunk fallback** WITHIN an MSM (bbstark's P5.2 at the kernel level; C24 applies the
   same adaptive idea at the coarser resident→streaming grain).
3. **Entrypoint-drives-all-allocations** as a grep-auditable invariant.
