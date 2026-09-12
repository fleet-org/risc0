#!/usr/bin/env bash
# groth16_s01/scripts/precommit.sh — the milestone's commit gate. One script, four surfaces:
#
#   --staged          what the pre-commit hook runs: the index
#   --msg <file>      what the commit-msg hook runs: the message
#   --range <a>..<b>  everything a PR carries: the files changed in the range and every message in it
#   --install         write both hooks into .git/hooks (they call this script)
#
# Checks, in order: privacy (no local paths, private hosts, RFC1918 addresses, e-mail addresses,
# credentials, DO-NOT-MERGE markers), GitHub links in markdown (full-SHA permalinks that resolve),
# whitespace, rustfmt, license headers (the upstream checker), clippy on the crates touched, and the
# commit message shape. The same checks run in CI (.github/workflows/groth16-s01.yml, job `hygiene`).
#
# Env: PRECOMMIT_OFFLINE=1 skips link resolution (the shape checks still run);
#      PRECOMMIT_NO_LINT=1 skips clippy (use for docs-only commits when the target dir is cold).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SELF=groth16_s01/scripts/precommit.sh
fail=0
say() { printf '%s\n' "$*" >&2; }
bad() { fail=1; say "FAIL: $*"; }

# ---- what to look at -------------------------------------------------------------------------
mode=${1:-}; shift || true
files=(); msgs=(); merges=(); added=""
case "$mode" in
  --install)
    for h in pre-commit commit-msg; do
      printf '#!/bin/sh\nexec %s %s "$@"\n' "$SELF" "$([ $h = pre-commit ] && echo --staged || echo --msg)" > ".git/hooks/$h"
      chmod +x ".git/hooks/$h"
    done
    say "installed .git/hooks/pre-commit and .git/hooks/commit-msg -> $SELF"; exit 0 ;;
  --staged)
    mapfile -t files < <(git diff --cached --name-only --diff-filter=ACMR)
    added=$(git diff --cached -U0 --no-color --diff-filter=ACMR -- . ":!$SELF" | grep -E '^\+' | grep -vE '^\+\+\+ ' || true) ;;
  --msg)
    msgs=("$(cat "$1")") ;;
  --range)
    range=$1
    mapfile -t files < <(git diff --name-only --diff-filter=ACMR "$range")
    added=$(git diff -U0 --no-color --diff-filter=ACMR "$range" -- . ":!$SELF" | grep -E '^\+' | grep -vE '^\+\+\+ ' || true)
    # merge commits carry GitHub's subject, not a conventional one: privacy-scanned, shape-exempt
    while IFS= read -r sha; do
      if [ "$(git rev-list --parents -n1 "$sha" | wc -w)" -gt 2 ]; then merges+=("$(git log -1 --format=%B "$sha")")
      else msgs+=("$(git log -1 --format=%B "$sha")"); fi
    done < <(git rev-list --reverse "$range") ;;
  *) say "usage: $SELF --staged | --msg <file> | --range <a>..<b> | --install"; exit 2 ;;
esac
exists() { [ -f "$1" ] && ! git check-ignore -q "$1"; }
present=(); for f in "${files[@]}"; do exists "$f" && present+=("$f"); done

# ---- 1. privacy: publish-bound text carries no local metadata, hosts, addresses or secrets --------
# (github-skill P-009 / VC-009; the milestone's rule: describe private infrastructure by role)
PRIV='(/chome|/work/|/Users/[A-Za-z]|/home/[a-z][a-z0-9_-]*/|jojo|bastion|\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b|\b192\.168\.[0-9]{1,3}\.[0-9]{1,3}\b|\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}\b|DO[_ ]?NOT[_ ]?MERGE)'
SECRETS='(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|sk-ant-[A-Za-z0-9_-]{10,}|xox[abp]-[A-Za-z0-9-]{10,}|eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})'
MAIL='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
MAIL_OK='(noreply@anthropic\.com|users\.noreply\.github\.com|@risczero\.com|@example\.(com|org))'
scan() { # $1 = label, stdin = text
  local hit
  hit=$(grep -nEi "$PRIV" || true);                      [ -n "$hit" ] && bad "$1: private path/host/address or merge marker:"$'\n'"$hit"
  hit=$(grep -nE "$SECRETS" || true);                    [ -n "$hit" ] && bad "$1: credential-shaped string:"$'\n'"$(printf '%s' "$hit" | cut -c1-60)"
  hit=$(grep -noE "$MAIL" | grep -vE "$MAIL_OK" || true); [ -n "$hit" ] && bad "$1: e-mail address:"$'\n'"$hit"
  return 0
}
[ -n "$added" ] && printf '%s\n' "$added" | scan "added lines"
for i in "${!msgs[@]}"; do printf '%s\n' "${msgs[$i]}" | scan "commit message $((i+1))"; done
for i in "${!merges[@]}"; do printf '%s\n' "${merges[$i]}" | scan "merge commit message $((i+1))"; done

# ---- 2. links in markdown: full-SHA permalinks, well-formed, and resolving ------------------------
# (github-skill VC-011 / TRAP-012 / I-GHW-005; milestone rule I-G16-008)
md=(); for f in "${present[@]}"; do [[ $f == *.md ]] && md+=("$f"); done
if [ ${#md[@]} -gt 0 ]; then
  hit=$(grep -nE 'blob//|/pull/\)|\[#@|/commit/\)|/issues/\)' "${md[@]}" || true)
  [ -n "$hit" ] && bad "malformed link (empty SHA or number):"$'\n'"$hit"
  while IFS= read -r ref; do
    [[ $ref =~ ^[0-9a-f]{40}$ ]] || bad "blob link on a non-SHA ref '$ref' (use a full commit SHA)"
  done < <(grep -ohE 'github\.com/[^/ )]+/[^/ )]+/blob/[^/ )]+' "${md[@]}" | sed -E 's#.*/blob/##' | sort -u)
  if [ "${PRECOMMIT_OFFLINE:-0}" != 1 ] && command -v gh >/dev/null; then
    cache=${XDG_CACHE_HOME:-$HOME/.cache}/groth16-s01-links; mkdir -p "$cache"
    resolve() { # $1 = api path; cached on success
      local key; key=$(printf '%s' "$1" | sha1sum | cut -c1-40)
      [ -f "$cache/$key" ] && return 0
      if gh api "$1" --silent >/dev/null 2>&1; then : > "$cache/$key"; else bad "link does not resolve: $1"; fi
    }
    while IFS= read -r u; do
      u=${u%%#*}; u=${u%%\?*}
      if [[ $u =~ github\.com/([^/]+)/([^/]+)/blob/([0-9a-f]{40})/(.+)$ ]]; then
        resolve "repos/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}/contents/${BASH_REMATCH[4]}?ref=${BASH_REMATCH[3]}"
      elif [[ $u =~ github\.com/([^/]+)/([^/]+)/commit/([0-9a-f]{7,40})$ ]]; then
        resolve "repos/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}/commits/${BASH_REMATCH[3]}"
      elif [[ $u =~ github\.com/([^/]+)/([^/]+)/(issues|pull)/([0-9]+)$ ]]; then
        resolve "repos/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}/issues/${BASH_REMATCH[4]}"
      elif [[ $u =~ github\.com/([^/]+)/([^/]+)/milestone/([0-9]+)$ ]]; then
        resolve "repos/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}/milestones/${BASH_REMATCH[3]}"
      fi
    done < <(grep -ohE 'https://github\.com/[^ )>]+' "${md[@]}" | sort -u)
  fi
fi

# ---- 3. whitespace ------------------------------------------------------------------------------
for f in "${present[@]}"; do
  case "$f" in *.bin|*.bincode|*.png|*.jpg|*.wtns|*.zkey|*.json) continue ;; esac
  grep -qI . "$f" || continue                                   # binary
  [ -n "$(tail -c1 "$f")" ] && bad "$f: no newline at end of file"
  grep -q $'\r' "$f" && bad "$f: CRLF line endings"
  case "$f" in *.rs|*.toml|*.yml|*.yaml|*.sh|*.py|*.metal)
    hit=$(grep -nE '[[:space:]]+$' "$f" || true); [ -n "$hit" ] && bad "$f: trailing whitespace:"$'\n'"$(printf '%s' "$hit" | head -3)" ;;
  esac
done

# ---- 3b. Metal shaders: a C++ syntax pass with MSL's type names poisoned (I-G16-019) ------------
for f in "${present[@]}"; do
  case "$f" in risc0/groth16-metal/src/*.metal|groth16_s01/scripts/msl-check.sh|groth16_s01/scripts/msl-stub/*)
    groth16_s01/scripts/msl-check.sh --self-test >/dev/null 2>&1 || bad "msl-check self-test failed"
    groth16_s01/scripts/msl-check.sh >/dev/null 2>&1 || bad "Metal shaders: run groth16_s01/scripts/msl-check.sh"
    break ;;
  esac
done

# ---- 4. rustfmt, cargo-sort, license headers, clippy — on the crates touched ---------------------
crate_of() { case "$1" in
  risc0/groth16-core/*)  echo risc0-groth16-core ;;  risc0/groth16-oxide/*) echo risc0-groth16-oxide ;;
  risc0/groth16-metal/*) echo risc0-groth16-metal ;; risc0/groth16-sys/*)   echo risc0-groth16-sys ;;
  risc0/groth16-cuda/*)  echo risc0-groth16-cuda ;;
  groth16_s01/harness/*) echo groth16-s01-harness ;; esac; }
declare -A crates=(); rs=0
for f in "${present[@]}"; do c=$(crate_of "$f"); [ -n "$c" ] && crates[$c]=1; [[ $f == *.rs ]] && rs=1; done
if [ ${#crates[@]} -gt 0 ]; then
  pk=(); for c in "${!crates[@]}"; do pk+=(-p "$c"); done
  [ $rs = 1 ] && { cargo fmt --check "${pk[@]}" >/dev/null 2>&1 || bad "rustfmt: run 'cargo fmt ${pk[*]}'"; }
  if command -v cargo-sort >/dev/null; then cargo sort --workspace --grouped --check >/dev/null 2>&1 || bad "Cargo.toml order: run 'cargo sort --workspace --grouped'"; fi
fi
for f in "${present[@]}"; do [[ $f == *.rs || $f == *.h || $f == *.cpp ]] && { python3 license-check.py >/dev/null 2>&1 || bad "license headers: run 'python3 license-check.py' (and --fix for years)"; break; }; done
if [ $rs = 1 ] && [ "${PRECOMMIT_NO_LINT:-0}" != 1 ]; then
  for c in "${!crates[@]}"; do
    # the CUDA arm's host crate builds cuda-bindings, which needs the CUDA headers (CUDA_HOME;
    # groth16_s01/scripts/cuda-headers.sh fetches them rootless) — without them, say so and skip it
    sysfeats=reference,oxide-cpu
    if [ -n "${CUDA_HOME:-}" ]; then sysfeats=$sysfeats,cuda-oxide
    elif [ "$c" = risc0-groth16-cuda ]; then say "note: CUDA_HOME unset — clippy on $c skipped (CI runs it)"; continue; fi
    # the harness pulls the CPU recursion prover: a cold clippy is a 30-minute compile, so it is opt-in
    # (PRECOMMIT_HARNESS_CLIPPY=1, set in CI); the owning session checks it in the harness target dir
    if [ "$c" = groth16-s01-harness ] && [ "${PRECOMMIT_HARNESS_CLIPPY:-0}" != 1 ]; then say "note: clippy on $c skipped (PRECOMMIT_HARNESS_CLIPPY=1 to run; CI runs it)"; continue; fi
    case $c in risc0-groth16-sys) feats=(--features "$sysfeats") ;; *) feats=() ;; esac
    if ! out=$(cargo clippy -p "$c" "${feats[@]}" --all-targets --locked -- -D warnings 2>&1); then
      bad "clippy ($c):"$'\n'"$(printf '%s\n' "$out" | grep -E '^(error|warning)' | head -5)"
    fi
  done
fi

# ---- 5. commit message shape -----------------------------------------------------------------
# (github-skill P-005 conventional commits; the trailer set is DEF-G16-004 — warned, not required,
#  so a human successor's commit passes)
for i in "${!msgs[@]}"; do
  subj=$(printf '%s\n' "${msgs[$i]}" | head -1)
  [[ $subj =~ ^(feat|fix|docs|ci|refactor|chore|test|perf|build)(\([a-z0-9_./-]+\))?!?:\ .+ ]] || bad "commit message $((i+1)): subject is not 'type(scope): description' — '$subj'"
  [ ${#subj} -le 100 ] || bad "commit message $((i+1)): subject longer than 100 characters"
  printf '%s\n' "${msgs[$i]}" | grep -q '^Co-Authored-By: ' || say "note: commit message $((i+1)) has no Co-Authored-By trailer"
done

[ $fail = 0 ] && { say "precommit: ok ($mode, ${#present[@]} files, ${#msgs[@]} messages)"; exit 0; }
say "precommit: FAILED — fix the items above (PRECOMMIT_OFFLINE=1 / PRECOMMIT_NO_LINT=1 narrow the gate; never skip privacy)"; exit 1
