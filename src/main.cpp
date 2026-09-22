#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <queue>
#include <filesystem>
#include <fstream>
#include <kernel.cu>

using json = nlohmann::json;
using bf16 = __nv_bfloat16;
using fs = std::filesystem;

constexpr int B_TO_MB = 1024*1024;
constexpr int B_TO_GB = 1024*1024*1024;

// architecture dependent parameters. Read from config later
constexpr int N_LAYERS = 16;
constexpr int MAX_PROMPT_LEN = 512;

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
  bf16* post_attn_layernorm[N_LAYERS];
  bf16* w_q[N_LAYERS];
  bf16* w_k[N_LAYERS];
  bf16* w_v[N_LAYERS];
  bf16* w_o[N_LAYERS];
  bf16* final_norm;
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
  std::queue<std::vector<int>>& queue, std::vector<bool>& is_slot_free,
  int slot, int* gpu_input_tokens,
  bf16* input_embeddings, Weights& weights
) {
  
  prompt = queue.front();
  prompt_len = prompt.size();
  queue.pop();
  is_slot_free[slot] = false;

  cudaMemcpy(gpu_input_tokens, prompt.data(), prompt_len*sizeof(int), cudaMemcpyHostToDevice);
  embeddingGather(gpu_input_tokens, input_embeddings, weights.embed_tokens, prompt_len);
  
}

int main() {
  
}
