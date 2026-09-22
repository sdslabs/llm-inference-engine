#include "kernels.cuh"
#include <iostream>
#include <vector>

__global__ void embeddingGatherKernel(int* gpu_input_tokens, bf16* gpu_input_embeds, bf16* embed_tokens, int num_input_tokens) {
  int workIdx = blockIdx.x*E_DIM + threadIdx.x;
  if(workIdx < num_input_tokens*E_DIM) {
    gpu_input_embeds[workIdx] = embed_tokens[gpu_input_tokens[blockIdx.x]*E_DIM + threadIdx.x];
    gpu_input_embeds[workIdx + E_DIM/2] = embed_tokens[gpu_input_tokens[blockIdx.x]*E_DIM + threadIdx.x + E_DIM/2];
  }
}

void embeddingGather(int* gpu_input_tokens, bf16* gpu_input_embeds, bf16* embed_tokens, int num_input_tokens) {
  embeddingGatherKernel<<<num_input_tokens, E_DIM/2>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens, num_input_tokens);

  cudaError error = cudaGetLastError();
  if(error!=cudaError::cudaSuccess) {
    std::cerr << "CUDA last error: " << cudaGetErrorString(error) << std::endl;
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

    if(threadIdx.x==0) rms_vector[0] = sqrtf( (rms_vector[0]/E_DIM) + 1.0e-5);
    __syncthreads();

    output[workIdx] = (bf16)( ((float)input[workIdx]/rms_vector[0]) * (float)norm_weights[threadIdx.x]);
    output[workIdx+E_DIM/2] = (bf16)(((float)input[workIdx+E_DIM/2]/rms_vector[0]) * (float)norm_weights[threadIdx.x+E_DIM/2]);
  }

}

void rmsNorm(bf16* input, bf16* output, bf16* norm_weights, int num_tokens) {
  rmsNormKernel<<<num_tokens, E_DIM/2>>>(input, output, norm_weights, num_tokens);
  cudaError error = cudaGetLastError();
  if(error != cudaError::cudaSuccess) {
    std::cerr << "CUDA last error : " << cudaGetErrorString(error) << std::endl;
  }
}

// hard-coded for Llama values for now
__global__ void siluKernel(bf16* a, bf16* b)
{
    int workIdx = threadIdx.x + blockIdx.x * 8192;
    for (int i = 0; i < 8192; i += 1024)
    {
        a[workIdx + i] = (bf16)((float)a[workIdx + i] * (1 / (1 + expf(-(float)a[workIdx + i]))) * (float)b[workIdx + i]);
    }
}

void silu(bf16* a, bf16* b, int num_tokens)
{
    siluKernel<<<num_tokens, 1024>>>(a, b);
}

__global__ void residualKernel(bf16* input, bf16* residual)
{
    int workIdx = threadIdx.x + blockIdx.x * 2048;
    input[workIdx] = input[workIdx] + residual[workIdx];
    input[workIdx + 1024] = input[workIdx + 1024] + residual[workIdx + 1024];
}

void residualAdd(bf16* input, bf16* residual, int num_tokens)
{
    residualKernel<<<num_tokens, 1024>>>(input, residual);
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetErrorString(error) << std::endl;
    }
}

__global__ void softmaxKernel(bf16* input, int num_tokens)
{
    __shared__ float m[1024];
    __shared__ float d[1024];

    int workIdx = blockIdx.x * num_tokens + threadIdx.x;
    float token = (float)input[workIdx];

    m[threadIdx.x] = token;
    d[threadIdx.x] = 1.0f;
    __syncthreads();

    for (int i = 1; i < num_tokens; i = i * 2) {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens) {
            float m_a = m[threadIdx.x];
            float d_a = d[threadIdx.x];
            float m_b = m[threadIdx.x + i];
            float d_b = d[threadIdx.x + i];

            float m_new = fmaxf(m_a, m_b);
            float d_new = d_a * expf(m_a - m_new) + d_b * expf(m_b - m_new);

            m[threadIdx.x] = m_new;
            d[threadIdx.x] = d_new;
        }
        __syncthreads();
    }

    input[workIdx] = (bf16)(expf(token - m[0]) / d[0]);
}

void softmax(bf16* input, int num_tokens)
{
    if (num_tokens > 1024) {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess) {
        std::cout << "CUDA last error: " << cudaGetErrorString(error) << std::endl;
    }
}






