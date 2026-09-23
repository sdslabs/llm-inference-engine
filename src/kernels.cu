#include "kernels.cuh"
#include <algorithm>
#include <iostream>
#include <vector>


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

    if(threadIdx.x==0) rms_vector[0] = sqrtf( (rms_vector[0] / E_DIM) + 1.0e-5);
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

__global__ void softmaxKernel(bf16* input, int num_tokens) {
  // parallel reduction
  __shared__ float m[1024];
  __shared__ float d[1024];

  int workIdx = blockIdx.x*num_tokens + threadIdx.x;
  float token = (float)input[workIdx];

  m[threadIdx.x] = token;
  d[threadIdx.x] = 1.0f;
  __syncthreads();
  
  // binary tree types
  for(int i=1; i<num_tokens; i*=2) {
    if(threadIdx.x % (i*2) == 0 && threadIdx.x + i < num_tokens) {
      float m_a = m[threadIdx.x];
      float d_a = d[threadIdx.x];
      float m_b = m[threadIdx.x+i];
      float d_b = d[threadIdx.x+i];

      float m_new = fmaxf(m_a, m_b);
      float d_new = d_a*expf(m_a-m_new) + d_b*expf(m_b - m_new);

      m[threadIdx.x] = m_new;
      d[threadIdx.x] = d_new;
    }
    __syncthreads();
  }

  input[workIdx] = (bf16)(expf(token - m[0]) / d[0]);
}

void softmax(bf16* input, int num_tokens) {
  if(num_tokens > MAX_NUM_THREAD) {
    std::cout << "Can't launch more than " << MAX_NUM_THREAD << " threads on current GPU";
    return;
  }

  // each attention head has a (num_tokens, num_tokens) matrix (Q * Kt)
  // each row of each matrix gets its own block containing num_tokens threads
  softmaxKernel<<<num_tokens*NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
  cudaError error = cudaGetLastError();
  if (error != cudaError::cudaSuccess)
  {
    std::cerr << "CUDA last error: " << cudaGetErrorString(error) << std::endl;
  }
}

__global__ void residualKernel(bf16* input, bf16* skip_conn) {
  int workIdx = threadIdx.x + blockIdx.x*E_DIM;
  input[workIdx] += skip_conn[workIdx];
  input[workIdx+E_DIM/2] += skip_conn[workIdx+E_DIM/2];
}

void residualAdd(bf16* input, bf16* skip_conn, int num_tokens) {
  residualKernel<<<num_tokens, E_DIM/2>>>(input, skip_conn);
  cudaError error = cudaGetLastError();
  if(error != cudaError::cudaSuccess) std::cout << "CUDA last error : " << cudaGetErrorString(error) << std::endl;
}

// hard-coded values for Llama 3.2 1B
__global__ void siluKernel(bf16* a, bf16* b) {
  int workIdx = blockIdx.x*INTERMEDIATE_DIM + threadIdx.x;
  for(int i=0; i<INTERMEDIATE_DIM; i += MAX_NUM_THREAD) {
    a[workIdx+i] = (bf16)((float)a[workIdx + i]*(1/(1 + expf(-(float)a[workIdx + i]))) * (float)b[workIdx + i]);
  }
}

void silu(bf16* a, bf16* b, int num_tokens) {
  siluKernel<<<num_tokens, 1024>>>(a, b);
  cudaError error = cudaGetLastError();
  if(error != cudaError::cudaSuccess) std::cout << "CUDA last error : " << cudaGetErrorString(error) << std::endl;

}

__global__ void causalMaskKernel(bf16* input, int num_tokens) {
  if( threadIdx.x + blockIdx.x*blockDim.x >= num_tokens*num_tokens*NUM_Q_HEADS) return;

  int column = threadIdx.x;
  int row = blockIdx.x%num_tokens;
  int workIdx = threadIdx.x + blockIdx.x*num_tokens;
  if(column > row) input[workIdx] = -INF;
}

void causalMask(bf16* input, int num_tokens) {
  if(num_tokens > MAX_NUM_THREAD) {
    std::cout << "Can't launch more than 1024 threads on this GPU, Causal mask kernel not launched";
    return;
  }

  causalMaskKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
  cudaError error = cudaGetLastError();
  if(error != cudaError::cudaSuccess) std::cout << "CUDA last error : " << cudaGetErrorString(error) << std::endl;

}

void init_rope_frequencies(
  int head_dim, int max_seq_len, float rope_theta, // usually 10000
  float factor, float low_freq_factor,
  float high_freq_factor, int original_max_len) {

  int half_dim = head_dim/2;
  std::vector<float> inv_freq(half_dim);
  for(int i=0; i<half_dim; i++) {
    inv_freq[i] = 1.0f / std::pow(rope_theta, (2.0f*i)/head_dim);
  }

  // RoPE frequency scaling for extended context.
  float low_freq_wavelen = (float)original_max_len / low_freq_factor;
  float high_freq_wavelen = (float)original_max_len / high_freq_factor;

  std::vector<float> inv_freq_llama = inv_freq;

  for(int i=0; i<half_dim; i++) {
    float wavelen = 2.0f*M_PI / inv_freq[i];
    if(wavelen > low_freq_wavelen) inv_freq_llama[i] = inv_freq[i] / factor;
    else if(wavelen >= high_freq_wavelen) {
      float smooth = ((float)original_max_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
      inv_freq_llama[i] = (1.0f - smooth) * (inv_freq[i] / factor) + smooth * inv_freq[i];
    }
  }

  cudaMalloc(&d_inv_freq, half_dim*sizeof(float));
  cudaMemcpy(d_inv_freq, inv_freq_llama.data(), half_dim*sizeof(float), cudaMemcpyHostToDevice);

  std::vector<float> cos_table(max_seq_len * head_dim);
    std::vector<float> sin_table(max_seq_len * head_dim);

  for (int pos = 0; pos < max_seq_len; pos++) {
    for (int i = 0; i < half_dim; i++) {
      float angle = pos * inv_freq_llama[i];
      float c = std::cos(angle);
      float s = std::sin(angle);
      cos_table[pos * head_dim + 2 * i] = c;
      cos_table[pos * head_dim + 2 * i + 1] = c;
      sin_table[pos * head_dim + 2 * i] = s;
      sin_table[pos * head_dim + 2 * i + 1] = s;
    }
  }

  cudaMalloc(&d_cos_table, max_seq_len * head_dim * sizeof(float));
  cudaMalloc(&d_sin_table, max_seq_len * head_dim * sizeof(float));
  cudaMemcpy(d_cos_table, cos_table.data(), max_seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sin_table, sin_table.data(), max_seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);

}

void free_rope_frequencies(void)
{
  if (d_inv_freq) {
     cudaFree(d_inv_freq);
     d_inv_freq = nullptr;
  }
  if (d_cos_table) {
     cudaFree(d_cos_table);
     d_cos_table = nullptr;
  }
  if (d_sin_table) {
     cudaFree(d_sin_table);
     d_sin_table = nullptr;
  }
}

__global__ void ropeKernel(bf16* input, int num_tokens, int proj_dim,
                                  int head_dim, const float* cos_table, const float* sin_table)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int half_proj = proj_dim / 2;
    int half_dim = head_dim / 2;

    if (tid >= half_proj) return;

    int head_idx = tid / half_dim;
    int pair_idx = tid % half_dim;

    int base = token_idx * proj_dim + head_idx * head_dim;
    int idx1 = base + pair_idx;
    int idx2 = base + pair_idx + half_dim;

    float x1 = (float)input[idx1];
    float x2 = (float)input[idx2];

    int table_idx = token_idx * head_dim + pair_idx * 2;
    float c = cos_table[table_idx];
    float s = sin_table[table_idx];

    input[idx1] = (bf16)(x1 * c - x2 * s);
    input[idx2] = (bf16)(x1 * s + x2 * c);
}

void rope(bf16* input, int num_tokens, int proj_dim) {
  int num_threads = proj_dim / 2;
  if(num_threads > 1024) {
    std::cout << "Can't launch more than 1024 threads on this GPU, RoPE kernel not launched";
    return;
  }

  ropeKernel<<<num_tokens, num_threads>>>(input, num_tokens, proj_dim, HEAD_DIM, d_cos_table, d_sin_table);
  cudaError error = cudaGetLastError();
  if(error != cudaError::cudaSuccess) std::cout << "CUDA last error : " << cudaGetErrorString(error) << std::endl;

}




// Decode Kernels below
__global__ void linearProjectionGEMVKernel(bf16* input, bf16* weight, bf16* output, int input_features, int output_features) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < output_features) {
    float sum = 0.0f;
    for(int col = 0; col < input_features; col++) {
      sum += (float)input[col] * (float)weight[row * input_features + col];
    }
    output[row] = (bf16)sum;
  }
}

void linearProjectionGEMV(bf16* input, bf16* weight, bf16* output, int input_features, int output_features) {
  int threads = std::min(output_features, MAX_NUM_THREAD);
  int blocks = (output_features + threads - 1) / threads;
  linearProjectionGEMVKernel<<<blocks, threads>>>(input, weight, output, input_features, output_features);
  cudaError error = cudaGetLastError();
  if(error != cudaError::cudaSuccess) {
    std::cerr << "CUDA last error in linearProjectionGEMV: " << cudaGetErrorString(error) << std::endl;
  }
}

// TODO: KVCache update kernel for decoding.

// TODO: decode Attention block kernel

__global__ void decodeSoftmaxKernel(bf16* attention_scores, int seq_len) {
  __shared__ float m[1024];
  __shared__ float d[1024];

  int workIdx = blockIdx.x * MAX_SEQ_LEN + threadIdx.x;
  float token = -INF;
  if(threadIdx.x < seq_len) token = (float)attention_scores[workIdx];

  m[threadIdx.x] = token;
  d[threadIdx.x] = 1.0f;
  __syncthreads();

  // binary tree types
  for(int i=1; i<seq_len; i*=2) {
    if(threadIdx.x % (i*2) == 0 && threadIdx.x + i <= seq_len) {
      float m_a = m[threadIdx.x];
      float d_a = d[threadIdx.x];
      float m_b = m[threadIdx.x+i];
      float d_b = d[threadIdx.x+i];

      float m_new = fmaxf(m_a, m_b);
      float d_new = d_a*expf(m_a-m_new) + d_b*expf(m_b - m_new);

      m[threadIdx.x] = m_new;
      d[threadIdx.x] = d_new;
    }
    __syncthreads();
  }

  if(threadIdx.x < seq_len) {
    attention_scores[workIdx] = (bf16)(expf(token - m[0]) / d[0]);
  }
}

void decodeSoftmax(bf16* attention_scores, int seq_len) {
  if (seq_len > MAX_NUM_THREAD) {
    std::cerr << "decodeSoftmax: seq_len " << seq_len << " exceeds " << MAX_NUM_THREAD << "\n";
    return;
  }
  decodeSoftmaxKernel<<<NUM_Q_HEADS, MAX_NUM_THREAD>>>(attention_scores, seq_len);
  cudaError error = cudaGetLastError();
  if(error != cudaError::cudaSuccess) {
    std::cerr << "CUDA last error in decodeSoftmax: " << cudaGetErrorString(error) << std::endl;
  }
}

// TODO: decode Attention values kernel