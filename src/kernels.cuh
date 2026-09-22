#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

using bf16 = __nv_bfloat16;

constexpr int EMBEDDING_LENGTH = 2048;
constexpr int E_DIM = EMBEDDING_LENGTH;
constexpr int NUM_Q_HEADS = 32;

// prefill
void embeddingGather(int* gpu_input_tokens, __nv_bfloat16* gpu_input_embeds, __nv_bfloat16* embed_tokens, int num_input_tokens);
void rmsNorm(bf16* input, bf16* output, bf16* norm_weights, int num_tokens);
void silu(bf16* a, bf16* b, int num_tokens);
void residualAdd(bf16* input, bf16* residual, int num_tokens);
void softmax(bf16* input, int num_tokens);


