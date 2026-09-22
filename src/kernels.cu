#include "kernels.cuh"
#include <iostream>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int EMBEDDING_LENGTH = 2048;
constexpr int E_DIM = EMBEDDING_LENGTH; // should be Even

__global__ void embeddingGatherKernel(int* gpu_input_tokens, bf16* gpu_input_embeds, bf16* embed_tokens, int num_input_tokens) {
  int workIdx = blockIdx.x*E_DIM + threadIdx.x;
  if(workIdx < num_input_tokens*E_DIM) {
    gpu_input_embeds[workIdx] = embed_tokens[gpu_input_tokens[blockIdx.x]*E_DIM + threadIdx.x];
    gpu_input_embeds[workIdx + E_DIM/2] = embed_tokens[gpu_input_tokens[blockIdx.x]*E_DIM + threadIdx.x + 1024];
  }
}

void embeddingGather(int* gpu_input_tokens, bf16* gpu_input_embeds, bf16* embed_tokens, int num_input_tokens) {
  embeddingGatherKernel<<<num_input_tokens, E_DIM/2>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens, num_input_tokens);

  cudaError error = cudaGetLastError();
  if(error!=cudaError::cudaSuccess) {
    std::cerr << "CUDA last error: " << cudaGetLastError() << std::endl;
  }
}

__global__ void rmsNormKernel(bf16* input, bf16* output, bf16* norm_weights, int num_tokens) {
  __shared__ float rms_vector[E_DIM/2]; // array shared among all threads of a block 

  int workIdx = threadIdx.x + blockIdx.x*E_DIM;
  if(workIdx < num_tokens*E_DIM) {
    rms_vector[threadIdx.x] = (float)input[workIdx]*(float)input[workIdx] + (float)input[workIdx+E_DIM/2]*(float)input[workIdx+E_DIM/2];
    __syncthreads(); // threads block untill all have reached this point

    for(int i=1; i < 1024; i*=2) {
      if(threadIdx.x % (i*2) == 0) {
        rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x+i];
      }
      __syncthreads();
    }

    if(threadIdx.x==0) rms_vector[0] = sqrt(rms_vector[0] / (E_DIM + 1.0e-5));
    __syncthreads();

    output[workIdx] = (bf16)( ((float)input[workIdx]/rms_vector[0]) * (float)norm_weights[threadIdx.x]);
    output[workIdx+E_DIM/2] = (bf16)(((float)input[workIdx+E_DIM/2]/rms_vector[0]) * (float)norm_weights[threadIdx.x+E_DIM/2]);
  }

}

void rmsNorm(bf16* input, bf16* output, bf16* norm_weights, int num_tokens) {
  rmsNormKernel<<<num_tokens, E_DIM/2>>>(input, output, norm_weights, num_tokens);
  cudaError error = cudaGetLastError();
  if(error != cudaError::cudaSuccess) {
    std::cerr << "CUDA last error : " << error << std::endl;
  }
}




