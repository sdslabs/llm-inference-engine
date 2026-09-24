#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <queue>
#include <filesystem>
#include <fstream>
#include "kernels.cuh"

using json = nlohmann::json;
namespace fs = std::filesystem;

int checkGPUStatus() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  if(device_count==0) {
    std::cerr << "No CUDA devices found\n";
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::cout << "Device: " << prop.name << "\n";
  std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
  std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
  std::cout << "SM count: " << prop.multiProcessorCount << "\n";
  std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

  size_t free_mem;
  size_t total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
  return 0;
}

// pointers to weight matrices on GPU
struct Weights {
  bf16* embed_tokens;
  bf16* input_layernorm[N_LAYERS];
  bf16* mlp_gate_proj[N_LAYERS];
  bf16* mlp_up_proj[N_LAYERS];
  bf16* mlp_down_proj[N_LAYERS];
  bf16* post_attn_layernorms[N_LAYERS];
  bf16* w_q[N_LAYERS];
  bf16* w_k[N_LAYERS];
  bf16* w_v[N_LAYERS];
  bf16* w_o[N_LAYERS];
  bf16* norm;
};

int loadWeights(Weights &weights, fs::path model_path) {
  if(checkGPUStatus()) return 1;

  std::ifstream safetensors_file(model_path, std::ios_base::binary);
  if(!safetensors_file.is_open()) {
    std::cerr << "Cannot open weights file\n";
    safetensors_file.close();
    return 1;
  }

  uint64_t header_size;
  safetensors_file.read(reinterpret_cast<char*>(&header_size), 8);

  std::string header;
  header.resize(header_size);
  safetensors_file.read(header.data(), header_size);
  json header_json = json::parse(header);

  std::unordered_map<std::string, uint64_t> offsets;
  uint64_t max_offset = 0;

  for(auto& [key, value] : header_json.items()) {
    if(key == "__metadata__") continue;
    uint64_t offset_end = value["data_offsets"].at(1).get<uint64_t>();
    if(offset_end > max_offset) max_offset = offset_end;

    offsets[key] = value["data_offsets"].at(0).get<uint64_t>(); // store the Start offset of each tensor
  }

  void* model_weights;
  cudaMalloc(&model_weights, max_offset);

  std::vector<char> model_weights_cpu;
  model_weights_cpu.resize(max_offset);
  safetensors_file.read(model_weights_cpu.data(), max_offset);

  cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset, cudaMemcpyHostToDevice);
  safetensors_file.close();
  
  weights.embed_tokens = (bf16*)((char*)model_weights + offsets.at("model.embed_tokens.weight"));
  weights.norm = (bf16*)((char*)model_weights + offsets.at("model.norm.weight"));

  for (int i = 0; i < N_LAYERS; ++i) {
    weights.input_layernorm[i] = (bf16*)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".input_layernorm.weight"));
    weights.mlp_down_proj[i] = (bf16*)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.down_proj.weight"));
    weights.mlp_gate_proj[i] = (bf16*)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"));
    weights.mlp_up_proj[i] = (bf16*)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.up_proj.weight"));
    weights.post_attn_layernorms[i] = (bf16*)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"));
    weights.w_k[i] = (bf16*)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight"));
    weights.w_o[i] = (bf16*)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight"));
    weights.w_q[i] = (bf16*)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight"));
    weights.w_v[i] = (bf16*)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight"));
  }

  return 0;
}

void prefill(
  // request
  std::vector<int>& prompt, int& prompt_len,

  // persistent, allocated once at startup
  Weights& weights, cublasHandle_t cublas_handle,
  bf16* kv_cache,                                       // N_LAYERS x 2 x MAX_PROMPT_LEN x KV_DIM
  bf16* rope_cos, bf16* rope_sin,                       // MAX_PROMPT_LEN x HEAD_DIM/2

  // residual stream scratch
  int* gpu_input_tokens,                                // prompt_len
  bf16* residual,                                       // snapshot taken before each sublayer
  bf16* hidden_state,                                   // carried across all N_LAYERS
  bf16* rms_norms,                                      // rmsNorm output, never in place

  // attention scratch
  bf16* q_proj,                                         // prompt_len x E_DIM
  bf16* k_proj, bf16* v_proj,                           // prompt_len x KV_DIM, scattered into kv_cache
  bf16* attn_scores,                                    // NUM_Q_HEADS x prompt_len x prompt_len
  bf16* attn_out,                                       // scores @ V
  bf16* o_proj,                                         // after w_o

  // mlp scratch
  bf16* gate, bf16* up,                                 // prompt_len x INTERMEDIATE_DIM
  bf16* down,                                           // prompt_len x E_DIM

  // output
  bf16* logits,                                         // VOCAB_SIZE, last token only
  std::vector<bf16>& logits_cpu                         // host readback for sampling
) {
  
  cudaMemcpy(gpu_input_tokens, prompt.data(), prompt_len*sizeof(int), cudaMemcpyHostToDevice);
  embeddingGather(gpu_input_tokens, residual, weights.embed_tokens, prompt_len);

  cudaMemcpy(hidden_state, residual, prompt_len*E_DIM*sizeof(bf16), cudaMemcpyDeviceToDevice);

  for(int layer=0; layer<N_LAYERS; layer++) {
    rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], prompt_len);

    linear(cublas_handle, rms_norms, weights.w_q[layer], q_proj, prompt_len, E_DIM, E_DIM ); 
    linear(cublas_handle, rms_norms, weights.w_k[layer], k_proj, prompt_len, E_DIM, KV_DIM); 
    linear(cublas_handle, rms_norms, weights.w_v[layer], v_proj, prompt_len, E_DIM, KV_DIM); 

    rope(q_proj, prompt_len, E_DIM );
    rope(k_proj, prompt_len, KV_DIM);

    const float scale = 1.0f / SQRT_HEAD_DIM;
    const float zero = 0.0f, one = 1.0f;

    for(int h=0; h<NUM_Q_HEADS; h++) {
      const bf16* Qh = q_proj + h*HEAD_DIM;
      const bf16* Kh = k_proj + (h/GQA_Q_TO_K_RATIO)*HEAD_DIM;
      bf16* Sh = attn_scores + h*prompt_len*prompt_len;
      cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, prompt_len, prompt_len,
                   HEAD_DIM, &scale, Kh, CUDA_R_16BF, KV_DIM, Qh, CUDA_R_16BF, E_DIM,
                   &zero, Sh, CUDA_R_16BF, prompt_len, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    causalMask(attn_scores, prompt_len);
    softmax(attn_scores, prompt_len);

    for (int h = 0; h < NUM_Q_HEADS; ++h) {
      const bf16* Vh = v_proj + (h/GQA_Q_TO_K_RATIO)*HEAD_DIM;
      const bf16* Sh = attn_scores + h*prompt_len*prompt_len;
      bf16* Oh = attn_out + h*HEAD_DIM;

      cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
               HEAD_DIM, prompt_len, prompt_len,
               &one,  Vh, CUDA_R_16BF, KV_DIM,
                      Sh, CUDA_R_16BF, prompt_len,
               &zero, Oh, CUDA_R_16BF, E_DIM,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    linear(cublas_handle, attn_out, weights.w_o[layer], o_proj, prompt_len, E_DIM, E_DIM);
    residualAdd(o_proj, residual, prompt_len);
    rmsNorm(o_proj, rms_norms, weights.post_attn_layernorms[layer], prompt_len);

    // now rms_norms has latest state. 

  }
  
}

int main() {
  checkGPUStatus();
}
