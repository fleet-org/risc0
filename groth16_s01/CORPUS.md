# GROTH16 s01/3 — the frozen SNARK-stage corpus: what to capture, how, and the manifest

**Milestone:** [GROTH16 s01](https://github.com/fleet-org/risc0/milestone/1) · **Issue:** [#4@fleet-org/risc0](https://github.com/fleet-org/risc0/issues/4) · **Depends on:** the boundary in [`BOUNDARY.md`](./BOUNDARY.md) §4.3
**Who runs the capture:** whoever holds access to the self-hosted production prover (a privileged CRCS or the operator). The owner session of this milestone does **not** have that access (P0) and wrote this so the capture is a copy operation, not a research task.
**Epistemic marks:** MEASURED = read from the pinned bento source (boundless [`93e971a6`](https://github.com/boundless-xyz/boundless/tree/93e971a653258bb2cf1de920b103fcfdfd9ef994)); UNVERIFIED = not checked against the live deployment.

## 1. What exists already, without any code change (MEASURED)

bento persists both ends of the stage in its object store (`S3_BUCKET` at `S3_URL`, MinIO in the self-hosted deployment):

| artifact | key | produced by |
|---|---|---|
| **stage input** — the succinct `Receipt`, bincode | `receipts/stark/<stark-job-uuid>.bincode` | the set-builder / aggregation job that precedes compression |
| **stage output** (risc0 circuit) — `Receipt` with `InnerReceipt::Groth16`, bincode | `receipts/groth16/<snark-job-uuid>.bincode` | `stark2snark`, `CompressType::Groth16` |
| **stage output** (blake3 circuit) — `Blake3Groth16Receipt`, bincode | `receipts/blake3_groth16/<snark-job-uuid>.bincode` | `stark2snark`, `CompressType::Blake3Groth16` |

Constants: [`workflow-common/src/s3.rs`](https://github.com/boundless-xyz/boundless/blob/93e971a653258bb2cf1de920b103fcfdfd9ef994/bento/crates/workflow-common/src/s3.rs) (`RECEIPT_BUCKET_DIR = "receipts"`, `STARK_BUCKET_DIR = "stark"`, `GROTH16_BUCKET_DIR = "groth16"`, `BLAKE3_GROTH16_BUCKET_DIR = "blake3_groth16"`). The same objects are reachable through the bento REST API (`GET /receipts/stark/receipt/<job>`, `GET /receipts/groth16/receipt/<job>`, `GET /receipts/shrink_bitvm2/receipt/<job>` for blake3) — but the API host is the production `bento :8081` surface this milestone's sessions must not touch; the object store copy below is the intended path.

The link between the two, and the canonical wall-clock, is in the task database (`DATABASE_URL`, Postgres): a snark task's `task_def` is the externally-tagged JSON `{"Snark": {"receipt": "<stark-job-uuid>", "compress_type": "Groth16" | "Blake3Groth16"}}`, and `started_at` → `updated_at` brackets the run ([`1_taskdb.sql`](https://github.com/boundless-xyz/boundless/blob/93e971a653258bb2cf1de920b103fcfdfd9ef994/bento/crates/taskdb/migrations/1_taskdb.sql), [`workflow-common/src/lib.rs`](https://github.com/boundless-xyz/boundless/blob/93e971a653258bb2cf1de920b103fcfdfd9ef994/bento/crates/workflow-common/src/lib.rs) `SnarkReq`). The wall-clock includes witness generation and the object-store round trips — record it as the **stage** baseline, which is what s01/6 compares against.

## 2. The capture recipe (verbatim; run where `DATABASE_URL`, `S3_*` are set — the agent's own env)

```bash
set -euo pipefail
OUT=${OUT:-./corpus-groth16-s01}; mkdir -p "$OUT/cases"
# 1. Enumerate completed snark tasks with their input receipt, circuit, and wall-clock.
#    (state literal: check `\dT+ task_state`; 'done' is the terminal success state in bento's schema — UNVERIFIED against the live enum)
psql "$DATABASE_URL" -At -F $'\t' -c "
  SELECT t.job_id, t.task_def->'Snark'->>'receipt', t.task_def->'Snark'->>'compress_type',
         to_char(t.started_at AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'),
         EXTRACT(EPOCH FROM (t.updated_at - t.started_at)), t.retries
  FROM tasks t WHERE t.task_def ? 'Snark' AND t.state = 'done'
  ORDER BY t.updated_at DESC LIMIT 40;" > "$OUT/snark-tasks.tsv"
# 2. For each, copy the stage input and the canonical stage output. Nothing else is needed.
while IFS=$'\t' read -r SNARK STARK CT STARTED WALL RETRIES; do
  case "$CT" in Groth16) DIR=groth16;; Blake3Groth16) DIR=blake3_groth16;; *) echo "skip $SNARK ($CT)"; continue;; esac
  C="$OUT/cases/$SNARK"; mkdir -p "$C"
  aws --endpoint-url "$S3_URL" s3 cp "s3://$S3_BUCKET/receipts/stark/$STARK.bincode"   "$C/input.stark.bincode"
  aws --endpoint-url "$S3_URL" s3 cp "s3://$S3_BUCKET/receipts/$DIR/$SNARK.bincode"    "$C/canonical.$DIR.bincode"
  # 3. Segment count of the input job (prove tasks in the stark job) — the same task table.
  SEGS=$(psql "$DATABASE_URL" -At -c "SELECT count(*) FROM tasks WHERE job_id='$STARK' AND task_def ? 'Prove';")
  printf '{"snark_job":"%s","stark_job":"%s","compress_type":"%s","started_at_utc":"%s","canonical_wall_clock_s":%s,"retries":%s,"segment_count":%s}\n' \
    "$SNARK" "$STARK" "$CT" "$STARTED" "$WALL" "$RETRIES" "$SEGS" > "$C/task.json"
done < "$OUT/snark-tasks.tsv"
# 4. Hash everything; the manifest carries the digests, the release carries the bytes.
( cd "$OUT" && find cases -type f | sort | xargs sha256sum > SHA256SUMS )
```

What the recipe does **not** capture, and where it comes from:

- **batch size, claims folded, includes_assessor** — broker-side facts (the broker's batch record for the batch whose aggregation receipt is `stark_job`). Record them from the broker database or its logs when selecting cases; the exact broker query is UNVERIFIED here (broker schema not read). If they cannot be read for a case, write `null` and say so — never guess.
- **`segment_po2`** — the value in force on the capturing agent (bento default 20; fleet-legacy config default 21). Read it from the agent's configuration or start-up log, record which.
- **risc0 versions** — fixed by the deployed bento build: `risc0-zkvm 3.0.4`, `risc0-groth16 3.0.3`, `risc0-groth16-sys 0.1.0` (bento lockfile at the pinned boundless rev; confirm against the deployed binary's `build-meta.json`).

## 3. Coverage axes (mandatory, not suggestions)

Pick cases so the table below has at least one row per cell that the deployment can produce; if a cell cannot be produced, say so in the manifest rather than leaving it blank:

| axis | values |
|---|---|
| circuit | `Groth16` · `Blake3Groth16` (both are live paths; the boundary is shared) |
| batch size | one order · a full batch at the configured target |
| claims folded | 1 (the boundary case, root may be the assessor commitment digest) · many |
| assessor | finalize batch (assessor claim included) · non-finalize |
| malformed | **two** synthetic cases, generated by the harness from a real case, not captured: (a) the stage input truncated to half its bytes — rejected **before** the boundary (deserialization), so both implementations must fail identically; (b) a boundary-level malformation — a witness of the wrong length, or a zkey whose section count is not 10 — which exercises each implementation's *own* validation and must be rejected by both with the same error class |

## 4. Manifest — one row per case

`manifest.json`:

```json
{
  "corpus": "groth16-s01", "version": 1, "frozen_at_utc": "2026-..", 
  "risc0": {"zkvm": "3.0.4", "groth16": "3.0.3", "groth16_sys": "0.1.0"},
  "bento_source": {"repo": "boundless-xyz/boundless", "rev": "<full sha of the deployed build>"},
  "provenance": {"deployment": "self-hosted production prover (role only — no host names, no addresses)", "captured_by": "<label of the capturing session>"},
  "cases": [
    {
      "case_id": "<snark-job-uuid>", "compress_type": "Groth16",
      "input":  {"file": "cases/<id>/input.stark.bincode",      "sha256": "…", "bytes": 0},
      "canonical_output": {"file": "cases/<id>/canonical.groth16.bincode", "sha256": "…", "bytes": 0},
      "expected_output_digest": "sha256 of the canonical seal bytes (256 B) — the value s01/5 prints beside its own",
      "claim_digest": "hex, from the input receipt (the harness recomputes and must match)",
      "batch_size": null, "claims_folded": null, "includes_assessor": null,
      "segment_count": 0, "segment_po2": 20,
      "canonical_wall_clock_s": 0.0, "retries": 0,
      "notes": "anything that makes the case unusual; null-valued fields say why they are null"
    }
  ]
}
```

Three-state rule for every field: a value, `null` **with a reason in `notes`** (could not look), or the field absent is not allowed. "Could not read the broker record" and "the batch had no assessor" are different facts.

## 5. Freezing and where it lives

- **Location (proposed, DEF-G16-003):** a GitHub Release on this fork, tag `corpus-groth16-s01-v1`, assets = `manifest.json`, `SHA256SUMS`, and one `cases-<id>.tar` per case (receipts are hundreds of KB; well under asset limits). Immutable by tag, reachable from CI with `gh release download`, no infrastructure identifiers required. Alternative: the fleet's R2 under a `corpus/groth16-s01/v1/` prefix — equally acceptable; choose one and record it in the manifest.
- **Frozen means:** a new case set is a new version (`v2`), never an edit of `v1`. The harness pins the tag.
- **Redaction gate before upload:** the receipts and manifest contain no infrastructure identifiers by construction; the capturing session must not add any (no host names, no RFC1918 addresses, no bucket URLs). Run the publish-bound scan on `manifest.json` and `task.json` files before the release is created.

## 6. What the harness (s01/5) does with a case

1. Reads `input.stark.bincode`, derives the boundary input on the harness host (identity_p254 → seal → witness), runs the **selected** backend and the **canonical** backend on it, and verifies both outputs with the unmodified `risc0-groth16` verifier against the public inputs derived from the input receipt's claim (`BOUNDARY.md` §4.4).
2. Verifies `canonical_output` from the corpus in the same run (control arm, assertion 3), and records that its claim digest equals the one derived from the input.
3. Runs the mutation arms of s01/5 and prints the per-case row.

---

**Bottom line.** The corpus is a copy of objects bento already stores, plus a task-table query for timing — no instrumentation, no code change, no access to anything but the agent's own object store and database. The ask on [#4@fleet-org/risc0](https://github.com/fleet-org/risc0/issues/4) is for root-CP to route this recipe to a session that holds that access; the owner session cannot run it (P0).
