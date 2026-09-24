#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

ARCH="${ARCH:-sm_89}"
BUILD_DIR="${BUILD_DIR:-build}"
TARGET="${TARGET:-engine}"
SOURCES=(src/kernels.cu src/main.cpp)
HEADERS=(src/kernels.cuh)

if [[ "${1:-}" == "clean" ]]; then
  rm -rf "$BUILD_DIR" "$TARGET"
  echo "cleaned"
  exit 0
fi

if ! command -v nvcc >/dev/null; then
  echo "nvcc not found in PATH" >&2
  exit 1
fi

FLAGS=(-std=c++17 -arch="$ARCH" -Xcompiler -Wall)
if [[ "${DEBUG:-0}" == "1" ]]; then
  MODE=debug
  FLAGS+=(-O0 -g -G)
else
  MODE=release
  FLAGS+=(-O3 -lineinfo)
fi

stale() {
  local obj="$1"
  shift
  [[ -f "$obj" ]] || return 0
  local dep
  for dep in "$@"; do
    if [[ "$dep" -nt "$obj" ]]; then return 0; fi
  done
  return 1
}

mkdir -p "$BUILD_DIR"
echo "$MODE $ARCH"

objects=()
pids=()
for src in "${SOURCES[@]}"; do
  obj="$BUILD_DIR/$(basename "${src%.*}").$MODE.o"
  objects+=("$obj")
  if stale "$obj" "$src" "${HEADERS[@]}"; then
    echo "  compile $src"
    nvcc "${FLAGS[@]}" -c "$src" -o "$obj" &
    pids+=("$!")
  fi
done

failed=0
for pid in ${pids[@]+"${pids[@]}"}; do
  wait "$pid" || failed=1
done
if [[ "$failed" != "0" ]]; then
  echo "compilation failed, $TARGET not updated" >&2
  exit 1
fi

if stale "$TARGET" "${objects[@]}"; then
  echo "  link    $TARGET"
  nvcc -arch="$ARCH" "${objects[@]}" -lcublas -o "$TARGET"
else
  echo "  $TARGET up to date"
fi
