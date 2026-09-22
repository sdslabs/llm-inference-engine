#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

// prefill
void embeddingGather(int* gpu_input_tokens, __nv_bfloat16* gpu_input_embeds, __nv_bfloat16* embed_tokens, int num_input_tokens);

