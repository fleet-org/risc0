#!/usr/bin/env bash
# groth16_s01/scripts/fmt.sh — one formatter per file type for the milestone's files, so that a
# diff is reviewable: prose wrapped at 100 columns, shell / TOML / Metal / YAML in their formatters'
# canonical layout, Rust by rustfmt. `--check` fails on a file that would change; `--fix` rewrites.
#
#   fmt.sh --check [files…]     (no files: every milestone file)
#   fmt.sh --fix   [files…]
#
# Tools (rootless-installable by fmt/install.sh; CI installs them the same way): rustfmt, prettier
# (Markdown, YAML, JSON), shfmt + shellcheck (shell), taplo (TOML), clang-format (Metal).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mode=${1:-}
shift || true
case "$mode" in --check | --fix) ;; *)
  echo "usage: fmt.sh --check|--fix [files…]" >&2
  exit 2
  ;;
esac
cfg=groth16_s01/scripts/fmt
if [ $# -eq 0 ]; then
  mapfile -t files < <(git ls-files 'groth16_s01/*' 'handoffs/*' 'risc0/groth16-core/*' \
    'risc0/groth16-oxide/*' 'risc0/groth16-cuda/*' 'risc0/groth16-metal/*' 'risc0/groth16-sys/*' \
    '.github/workflows/groth16-s01*.yml')
else
  files=("$@")
fi
md=() yml=() sh=() toml=() metal=() json=()
rs=0
for f in "${files[@]}"; do
  [ -f "$f" ] || continue
  case "$f" in
    *.md) md+=("$f") ;;
    *.yml | *.yaml) yml+=("$f") ;;
    *.json) json+=("$f") ;;
    *.sh) sh+=("$f") ;;
    *.toml) toml+=("$f") ;;
    *.metal) metal+=("$f") ;;
    *.rs) rs=1 ;;
  esac
done
fail=0
need() {
  command -v "$1" > /dev/null && return 0
  echo "fmt: $1 is not installed — run groth16_s01/scripts/fmt/install.sh" >&2
  fail=1
  return 1
}
crates=(-p risc0-groth16-core -p risc0-groth16-oxide -p risc0-groth16-cuda -p risc0-groth16-metal
  -p risc0-groth16-sys -p groth16-s01-harness)
if [ "$rs" = 1 ]; then
  if [ "$mode" = --check ]; then
    cargo fmt --check "${crates[@]}" || fail=1
  else
    cargo fmt "${crates[@]}"
  fi
fi
pretty=("${md[@]}" "${yml[@]}" "${json[@]}")
if [ ${#pretty[@]} -gt 0 ] && need prettier; then
  if [ "$mode" = --check ]; then
    prettier --config "$cfg/prettierrc.json" --log-level warn --check "${pretty[@]}" || fail=1
  else
    prettier --config "$cfg/prettierrc.json" --log-level warn --write "${pretty[@]}"
  fi
fi
if [ ${#sh[@]} -gt 0 ]; then
  if need shfmt; then
    if [ "$mode" = --check ]; then
      shfmt -d -i 2 -ci -bn -sr "${sh[@]}" || fail=1
    else
      shfmt -w -i 2 -ci -bn -sr "${sh[@]}"
    fi
  fi
  if need shellcheck; then shellcheck -S warning "${sh[@]}" || fail=1; fi
fi
if [ ${#toml[@]} -gt 0 ] && need taplo; then
  if [ "$mode" = --check ]; then
    taplo fmt --config "$cfg/taplo.toml" --check "${toml[@]}" || fail=1
  else
    taplo fmt --config "$cfg/taplo.toml" "${toml[@]}"
  fi
fi
if [ ${#metal[@]} -gt 0 ] && need clang-format; then
  if [ "$mode" = --check ]; then
    clang-format "--style=file:$cfg/clang-format.yaml" --dry-run -Werror "${metal[@]}" || fail=1
  else
    clang-format "--style=file:$cfg/clang-format.yaml" -i "${metal[@]}"
  fi
fi
if [ "$fail" = 0 ]; then
  echo "fmt: ok ($mode, ${#files[@]} files)"
else
  echo "fmt: FAILED — run groth16_s01/scripts/fmt.sh --fix" >&2
  exit 1
fi
