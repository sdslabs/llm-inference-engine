#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Load .env. Shell exports win, so `ARCH=sm_86 ./build.sh` still overrides.
if [ -f .env ]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -z "$line" ] && continue
    key="${line%%=*}"
    val="${line#*=}"
    key="${key//[[:space:]]/}"
    val="${val#"${val%%[![:space:]]*}"}"
    [ -z "$key" ] && continue
    [ -n "${!key:-}" ] && continue
    export "$key=$val"
  done < .env
fi

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

# Limits that size fixed arrays and shared memory. Runtime values live in the
# model's config.json instead; these have to be baked in.
LIMITS=(
  -DENGINE_MAX_SEQUENCES="${MAX_SEQUENCES:-4}"
  -DENGINE_MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
  -DENGINE_MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
  -DENGINE_MAX_NUM_THREAD="${MAX_NUM_THREAD:-1024}"
  -DENGINE_DEFAULT_TOP_K="${DEFAULT_TOP_K:-40}"
  -DENGINE_DEFAULT_TEMPERATURE="${DEFAULT_TEMPERATURE:-0.8}f"
  -DENGINE_DEFAULT_MAX_NEW_TOKENS="${DEFAULT_MAX_NEW_TOKENS:-20}"
)

echo "Compiling CUDA Kernels for $ARCH..."
echo "  slots=${MAX_SEQUENCES:-4} prompt=${MAX_PROMPT_LEN:-512} seq=${MAX_SEQ_LEN:-2048}"
nvcc -O3 -arch="$ARCH" "${LIMITS[@]}" -c src/kernels.cu -o build/kernels.o

echo "Compiling Main Binary..."
g++ -O3 -std=c++17 "${LIMITS[@]}" \
    src/main.cpp src/config.cpp src/runtime.cpp src/model.cpp src/score.cpp src/telemetry.cpp \
    build/kernels.o \
    -o build/engine \
    -Isrc \
    -I${TOKENIZERS_DIR}/include \
    -I${CUDA_PATH}/include \
    -L${TOKENIZERS_DIR}/build -ltokenizers_cpp -ltokenizers_c \
    -L${CUDA_PATH}/lib64 -lcudart -lcublas

echo "Build complete: ./build/engine"
