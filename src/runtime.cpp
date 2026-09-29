#include "runtime.h"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

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

  std::cerr << "Device: " << prop.name << "\n";
  std::cerr << "Compute capability: " << prop.major << "." << prop.minor << "\n";
  std::cerr << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
  std::cerr << "SM count: " << prop.multiProcessorCount << "\n";
  std::cerr << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

  size_t free_mem;
  size_t total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  std::cerr << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
  return 0;
}

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

int Buffers::allocate(bool score_mode) {
  // persistent, one independent span per slot
  cudaMalloc(&kv_cache, (size_t)MAX_SEQUENCES * N_LAYERS * 2 * MAX_SEQ_LEN * KV_DIM * sizeof(bf16));

  // residual stream scratch
  cudaMalloc(&gpu_input_tokens, (size_t)MAX_PROMPT_LEN * sizeof(int));
  cudaMalloc(&gpu_positions,    (size_t)MAX_SEQUENCES * sizeof(int));
  cudaMalloc(&hidden_state,     (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));
  cudaMalloc(&rms_norms,        (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));

  // attention scratch
  cudaMalloc(&q_proj,      (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));
  cudaMalloc(&k_proj,      (size_t)MAX_PROMPT_LEN * KV_DIM * sizeof(bf16));
  cudaMalloc(&v_proj,      (size_t)MAX_PROMPT_LEN * KV_DIM * sizeof(bf16));
  cudaMalloc(&attn_scores, (size_t)NUM_Q_HEADS * MAX_PROMPT_LEN * MAX_PROMPT_LEN * sizeof(bf16));
  cudaMalloc(&attn_out,    (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));
  cudaMalloc(&o_proj,      (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));

  // mlp scratch
  cudaMalloc(&gate, (size_t)MAX_PROMPT_LEN * INTERMEDIATE_DIM * sizeof(bf16));
  cudaMalloc(&up,   (size_t)MAX_PROMPT_LEN * INTERMEDIATE_DIM * sizeof(bf16));
  cudaMalloc(&down, (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));

  // output. decode needs one logit row per active slot, prefill only uses row 0.
  // scoring needs every prompt position at once, which is 128x larger
  size_t logit_rows = score_mode ? (size_t)MAX_PROMPT_LEN : (size_t)MAX_SEQUENCES;
  cudaMalloc(&logits, logit_rows * VOCAB_SIZE * sizeof(bf16));

  cudaMalloc(&gpu_sampled_tokens, (size_t)MAX_SEQUENCES * sizeof(int));
  cudaMalloc(&gpu_rand,           (size_t)MAX_SEQUENCES * sizeof(float));

  if(score_mode) {
    cudaMalloc(&gpu_targets,  (size_t)MAX_PROMPT_LEN * sizeof(int));
    cudaMalloc(&gpu_logprobs, (size_t)MAX_PROMPT_LEN * sizeof(float));
  }

  if(cudaGetLastError() != cudaSuccess) {
    std::cerr << "Buffer allocation failed\n";
    return 1;
  }
  return 0;
}

void Buffers::free() {
  cudaFree(kv_cache);
  cudaFree(gpu_input_tokens); cudaFree(gpu_positions);
  cudaFree(hidden_state); cudaFree(rms_norms);
  cudaFree(q_proj); cudaFree(k_proj); cudaFree(v_proj);
  cudaFree(attn_scores); cudaFree(attn_out); cudaFree(o_proj);
  cudaFree(gate); cudaFree(up); cudaFree(down);
  cudaFree(logits); cudaFree(gpu_sampled_tokens); cudaFree(gpu_rand);
  cudaFree(gpu_targets); cudaFree(gpu_logprobs);
}
