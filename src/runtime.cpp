#include "runtime.h"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <algorithm>
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

int loadModelConfig(ModelConfig& mc, const fs::path& model_path) {
  fs::path config_path = model_path.parent_path() / "config.json";

  std::ifstream file(config_path);
  if(!file.is_open()) {
    std::cerr << "No config.json next to " << model_path.filename().string()
              << ", using the values the engine was built with\n";
    return 0;
  }

  json cfg;
  try {
    file >> cfg;
  } catch(const std::exception& e) {
    std::cerr << config_path.string() << ": " << e.what() << "\n";
    return 1;
  }

  mc.num_layers          = cfg.value("num_hidden_layers", mc.num_layers);
  mc.hidden_size         = cfg.value("hidden_size", mc.hidden_size);
  mc.intermediate_size   = cfg.value("intermediate_size", mc.intermediate_size);
  mc.num_attention_heads = cfg.value("num_attention_heads", mc.num_attention_heads);
  mc.num_key_value_heads = cfg.value("num_key_value_heads", mc.num_key_value_heads);
  mc.vocab_size          = cfg.value("vocab_size", mc.vocab_size);
  mc.rms_norm_eps        = cfg.value("rms_norm_eps", mc.rms_norm_eps);
  mc.rope_theta          = cfg.value("rope_theta", mc.rope_theta);
  mc.bos_token_id        = cfg.value("bos_token_id", mc.bos_token_id);

  int derived_head_dim = mc.num_attention_heads > 0
                       ? mc.hidden_size / mc.num_attention_heads : mc.head_dim;
  mc.head_dim = cfg.value("head_dim", derived_head_dim);

  if(cfg.contains("rope_scaling") && cfg["rope_scaling"].is_object()) {
    const json& rs = cfg["rope_scaling"];
    mc.rope_factor                = rs.value("factor", mc.rope_factor);
    mc.rope_low_freq_factor       = rs.value("low_freq_factor", mc.rope_low_freq_factor);
    mc.rope_high_freq_factor      = rs.value("high_freq_factor", mc.rope_high_freq_factor);
    mc.rope_original_max_position = rs.value("original_max_position_embeddings",
                                             mc.rope_original_max_position);
  } else {
    mc.rope_factor = 1.0f;
  }

  if(cfg.contains("eos_token_id")) {
    const json& eos = cfg["eos_token_id"];
    std::vector<int> ids;
    if(eos.is_array()) ids = eos.get<std::vector<int>>();
    else if(eos.is_number_integer()) ids.push_back(eos.get<int>());

    for(int id : ids) {
      if(std::find(mc.stop_token_ids.begin(), mc.stop_token_ids.end(), id)
         == mc.stop_token_ids.end()) {
        mc.stop_token_ids.push_back(id);
      }
    }
  }

  int mismatches = 0;
  auto require = [&](const char* name, long long from_config, long long compiled) {
    if(from_config == compiled) return;
    std::cerr << config_path.string() << ": " << name << " is " << from_config
              << ", engine built for " << compiled << "\n";
    mismatches++;
  };

  require("num_hidden_layers", mc.num_layers, N_LAYERS);
  require("hidden_size", mc.hidden_size, E_DIM);
  require("intermediate_size", mc.intermediate_size, INTERMEDIATE_DIM);
  require("num_attention_heads", mc.num_attention_heads, NUM_Q_HEADS);
  require("num_key_value_heads", mc.num_key_value_heads, NUM_K_HEADS);
  require("head_dim", mc.head_dim, HEAD_DIM);
  require("vocab_size", mc.vocab_size, VOCAB_SIZE);
  require("num_key_value_heads * head_dim",
          (long long)mc.num_key_value_heads * mc.head_dim, KV_DIM);
  if(mc.num_key_value_heads > 0) {
    require("num_attention_heads / num_key_value_heads",
            mc.num_attention_heads / mc.num_key_value_heads, GQA_Q_TO_K_RATIO);
  }

  if(!cfg.value("tie_word_embeddings", true)) {
    std::cerr << config_path.string() << ": tie_word_embeddings is false, but the engine"
              << " reuses embed_tokens as the output projection\n";
    mismatches++;
  }

  std::string dtype = cfg.value("torch_dtype", std::string("bfloat16"));
  if(dtype != "bfloat16") {
    std::cerr << config_path.string() << ": torch_dtype is " << dtype
              << ", the engine only reads bfloat16\n";
    mismatches++;
  }

  if(mismatches > 0) {
    std::cerr << mismatches << " mismatch(es) against the build. These sizes are compile time;"
              << " change them in .env or src/kernels.cuh and rebuild\n";
    return 1;
  }

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
