#!/usr/bin/env bash
set -e

mkdir -p build

TOKENIZERS_DIR="external/tokenizers-cpp"

echo "Compiling Cuda Kernels..."
nvcc -O3 -c src/kernels.cu -o build/kernels.o

echo "Compiling Main Binary..."
g++ -O3 src/main.cpp build/kernels.o \
    -o build/engine \
    -Isrc \
    -I${TOKENIZERS_DIR}/include \
    -I/opt/cuda/include \
    -L${TOKENIZERS_DIR}/build -ltokenizers_cpp -ltokenizers_c \
    -L/opt/cuda/lib64 -lcudart -lcublas

echo "Build successful! Binary location : ./build/engine"