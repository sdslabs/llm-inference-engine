#include "kernels.cuh"
#include <iostream>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int EMBEDDING_LENGTH = 2048;

__global__ void embeddingGatherKernel(int* gpu_input_tokens, bf16* gpu_input_embeds, bf16* embed_tokens, int num_input_tokens) {
  int workIdx = blockIdx.x*2048 + threadIdx.x;
  if(workIdx < num_input_tokens*2048) {
    gpu_input_embeds[workIdx] = embed_tokens[gpu_input_tokens[blockIdx.x]*2048 + threadIdx.x];
    gpu_input_embeds[workIdx + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x]*2048 + threadIdx.x + 1024];
  }
}

void embeddingGather(int* gpu_input_tokens, bf16* gpu_input_embeds, bf16* embed_tokens, int num_input_tokens) {
  embeddingGatherKernel<<<num_input_tokens, 1024>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens, num_input_tokens);

  cudaError error = cudaGetLastError();
  if(error!=cudaError::cudaSuccess) {
    std::cerr << "CUDA last error: " << cudaGetLastError() << std::endl;
  }
}




