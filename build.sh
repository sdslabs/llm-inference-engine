#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Dynamically locate CUDA directory
if [ -z "${CUDA_PATH:-}" ]; then
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    elif [ -d "/opt/cuda" ]; then
        CUDA_PATH="/opt/cuda"
    else
        NVCC_PATH=$(command -v nvcc || echo "/usr/bin/nvcc")
        CUDA_PATH="$(dirname "$(dirname "$NVCC_PATH")")"
    fi
fi

if [[ "${1:-}" == "clean" ]]; then
  rm -rf build
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

ARCH="${ARCH:-sm_89}"

echo "Compiling CUDA Kernels for $ARCH..."
nvcc -O3 -arch="$ARCH" -c src/kernels.cu -o build/kernels.o

echo "Compiling Main Binary..."
g++ -O3 -std=c++17 \
    src/main.cpp src/config.cpp src/runtime.cpp src/model.cpp src/score.cpp src/telemetry.cpp \
    build/kernels.o \
    -o build/engine \
    -Isrc \
    -I${TOKENIZERS_DIR}/include \
    -I${CUDA_PATH}/include \
    -L${TOKENIZERS_DIR}/build -ltokenizers_cpp -ltokenizers_c \
    -L${CUDA_PATH}/lib64 -lcudart -lcublas

echo "Build complete: ./build/engine"
