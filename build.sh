#!/usr/bin/env bash
set -e

# Dynamically locate CUDA directory
if [ -z "$CUDA_PATH" ]; then
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    elif [ -d "/opt/cuda" ]; then
        CUDA_PATH="/opt/cuda"
    else
        NVCC_PATH=$(command -v nvcc || echo "/usr/bin/nvcc")
        CUDA_PATH="$(dirname "$(dirname "$NVCC_PATH")")"
    fi
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

ARCH="${ARCH:-sm_89}"
BUILD_DIR="${BUILD_DIR:-build}"
TARGET="${TARGET:-engine}"
SOURCES=(src/kernels.cu src/config.cpp src/main.cpp)
HEADERS=(src/kernels.cuh src/config.h)

if [[ "${1:-}" == "clean" ]]; then
  rm -rf "$BUILD_DIR" "$TARGET"
  echo "cleaned"
  exit 0
fi

if ! command -v nvcc >/dev/null; then
  echo "nvcc not found in PATH" >&2
  exit 1
fi

echo "Using CUDA path: $CUDA_PATH"

mkdir -p build

TOKENIZERS_DIR="external/tokenizers-cpp"

echo "Compiling CUDA Kernels..."
nvcc -O3 -c src/kernels.cu -o build/kernels.o

echo "Compiling Main Binary..."
g++ -O3 src/main.cpp build/kernels.o \
    -o build/engine \
    -Isrc \
    -I${TOKENIZERS_DIR}/include \
    -I${CUDA_PATH}/include \
    -L${TOKENIZERS_DIR}/build -ltokenizers_cpp -ltokenizers_c \
    -L${CUDA_PATH}/lib64 -lcudart -lcublas

echo "Build complete: ./build/engine"