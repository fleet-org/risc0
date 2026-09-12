#!/usr/bin/env bash
# Install the formatters fmt.sh needs, rootless, into <bin-dir> (default: $HOME/bin). Pinned
# versions; the same script serves the CRCS and CI (ubuntu-latest), so both format identically.
#   prettier 3.9.6 (npm, into <bin-dir>/../npm), shfmt v3.14.1, shellcheck v0.11.0, taplo 0.10.0
#   (static release binaries) · clang-format 18.1.3 (Ubuntu 24.04 debs extracted next to <bin-dir>)
set -euo pipefail
bin=${1:-$HOME/bin}
mkdir -p "$bin"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
have() { [ -x "$bin/$1" ] && "$bin/$1" --version 2> /dev/null | grep -q "$2"; }

if ! have shfmt 3.14.1; then
  curl -sSfL -o "$bin/shfmt" \
    https://github.com/mvdan/sh/releases/download/v3.14.1/shfmt_v3.14.1_linux_amd64
  chmod +x "$bin/shfmt"
fi
if ! have shellcheck 0.11.0; then
  sc=https://github.com/koalaman/shellcheck/releases/download/v0.11.0
  curl -sSfL -o "$tmp/sc.tar.xz" "$sc/shellcheck-v0.11.0.linux.x86_64.tar.xz"
  python3 -c "import tarfile,sys; tarfile.open(sys.argv[1]).extractall(sys.argv[2])" \
    "$tmp/sc.tar.xz" "$tmp"
  install -m 0755 "$tmp/shellcheck-v0.11.0/shellcheck" "$bin/shellcheck"
fi
if ! have taplo 0.10.0; then
  curl -sSfL -o "$tmp/taplo.gz" \
    https://github.com/tamasfe/taplo/releases/download/0.10.0/taplo-linux-x86_64.gz
  gunzip -f "$tmp/taplo.gz"
  install -m 0755 "$tmp/taplo" "$bin/taplo"
fi
if ! have prettier 3.9.6; then
  npm install --prefix "$bin/../npm" --no-audit --no-fund prettier@3.9.6 > /dev/null
  ln -sf "$(cd "$bin/../npm" && pwd)/node_modules/.bin/prettier" "$bin/prettier"
fi
if ! have clang-format 18.1.3; then
  tc=$(cd "$bin/.." && pwd)/tc
  mkdir -p "$tc"
  pool=http://archive.ubuntu.com/ubuntu/pool
  debs=(main/l/llvm-toolchain-18/libclang-cpp18_18.1.3-1_amd64.deb
    main/l/llvm-toolchain-18/libllvm18_18.1.3-1_amd64.deb
    universe/l/llvm-toolchain-18/clang-format-18_18.1.3-1ubuntu1_amd64.deb)
  for deb in "${debs[@]}"; do
    curl -sSfL -o "$tmp/$(basename "$deb")" "$pool/$deb"
    dpkg-deb -x "$tmp/$(basename "$deb")" "$tc"
  done
  cat > "$bin/clang-format" << WRAP
#!/bin/sh
LD_LIBRARY_PATH="$tc/usr/lib/llvm-18/lib:$tc/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH
exec "$tc/usr/lib/llvm-18/bin/clang-format" "\$@"
WRAP
  chmod +x "$bin/clang-format"
fi
for t in shfmt shellcheck taplo prettier clang-format; do
  printf '%-13s %s\n' "$t" "$("$bin/$t" --version 2> /dev/null | head -1)"
done
