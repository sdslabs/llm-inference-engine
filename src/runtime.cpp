#include "runtime.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
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

std::pair<size_t, size_t> getRowCol(std::string name, json &header_json) {
  auto shape_raw = header_json.at(name).at("shape");
  std::vector<size_t> shape = static_cast<std::vector<size_t>>(shape_raw);
  size_t rows = shape.at(0);
  size_t cols = shape.at(1);
  return {rows, cols};
}

void quantizeMatrix(std::string name, json &header_json, std::vector<fp32> &scales, std::vector<char> &qt_model_weights_cpu, bf16* weightMatrix, std::unordered_map<std::string, std::pair<size_t, size_t>> &quantized_offsets, std::unordered_map<std::string, size_t> &scale_offsets) {
  size_t block_begin = qt_model_weights_cpu.size();
  std::pair<size_t, size_t> gate_dim = getRowCol(name, header_json);
  size_t scale_mat_begin = scales.size();
  scale_offsets[name] = scale_mat_begin;
  for (size_t row=0; row<gate_dim.first; row++){
    fp32 max_val = 0;
    for (size_t col=0; col<gate_dim.second; col++){
      fp32 val = std::abs((fp32)(weightMatrix[row*gate_dim.second+col]));
      if(val > max_val){
        max_val = val;
      }
    }
    fp32 scale;
    if(max_val == 0){
      scale = 1;
    }else{
      scale = max_val/127;
    }
    scales.push_back(static_cast<fp32>(scale));
  }

  for (size_t row=0; row<gate_dim.first; row++){
    for (size_t col=0; col<gate_dim.second; col++){
      fp32 val = static_cast<fp32>(weightMatrix[row*gate_dim.second+col]);
      int8_t qt_val = std::max(-127, std::min(127, static_cast<int>(std::round(val / scales[scale_mat_begin + row]))));
      qt_model_weights_cpu.push_back(qt_val);
    }
  }
  quantized_offsets[name] = {block_begin, qt_model_weights_cpu.size()};
}

struct header_entry {
  std::string name;
  std::string dtype;
  std::vector<size_t> shape;
  std::pair<size_t, size_t> offsets;
};

void addHeaderEntry(const std::string& name,const json& header_json, const std::unordered_map<std::string, std::pair<size_t, size_t>>& quantized_offsets, const std::unordered_map<std::string, size_t>& scale_offsets,size_t weights_bytes, std::vector<header_entry>& header_package) {
  auto shape = header_json.at(name).at("shape").get<std::vector<size_t>>();
  bool is_quantized = scale_offsets.count(name) != 0;
  if(is_quantized) {
    header_package.push_back({name, "I8", shape, quantized_offsets.at(name)});
    size_t begin = weights_bytes + scale_offsets.at(name) * sizeof(fp32);
    header_package.push_back({name + ".scale", "F32", {shape.at(0)}, {begin, begin + shape.at(0) * sizeof(fp32)}});
  }else {
    header_package.push_back({name, "BF16", shape, quantized_offsets.at(name)});
  }
}

int quantWeights(fs::path model_path) {
  QuantizedWeights quantized_weights{};
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
  std::unordered_map<std::string, std::pair<size_t, size_t>> quantized_offsets;
  std::unordered_map<std::string, size_t> scale_offsets;
  uint64_t max_offset = 0;

  for(auto& [key, value] : header_json.items()) {
    if(key == "__metadata__") continue;
    uint64_t offset_end = value["data_offsets"].at(1).get<uint64_t>();
    if(offset_end > max_offset) max_offset = offset_end;

    offsets[key] = value["data_offsets"].at(0).get<uint64_t>(); // store the Start offset of each tensor
  }


  std::vector<char> model_weights_cpu;
  std::vector<char> qt_model_weights_cpu;
  std::vector<fp32> scales;
  model_weights_cpu.resize(max_offset);
  safetensors_file.read(model_weights_cpu.data(), max_offset);


  Weights weights;
// QUANTIZE BY CALCULATING SCALES
  for (int i = 0; i < N_LAYERS; ++i) {
    weights.input_layernorm[i] = (bf16*)(&model_weights_cpu[0] + offsets.at("model.layers." + std::to_string(i) + ".input_layernorm.weight"));
    auto& ln_tensor = header_json.at("model.layers." + std::to_string(i) + ".input_layernorm.weight");
    size_t layer_start = static_cast<size_t>(ln_tensor["data_offsets"][0]);
    size_t layer_end = static_cast<size_t>(ln_tensor["data_offsets"][1]);
    weights.mlp_gate_proj[i] = (bf16*)(&model_weights_cpu[0] + offsets.at("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"));
    weights.mlp_down_proj[i] = (bf16*)(&model_weights_cpu[0]  + offsets.at("model.layers." + std::to_string(i) + ".mlp.down_proj.weight"));

    weights.mlp_up_proj[i] = (bf16*)(&model_weights_cpu[0] + offsets.at("model.layers." + std::to_string(i) + ".mlp.up_proj.weight"));
    weights.post_attn_layernorms[i] = (bf16*)(&model_weights_cpu[0] + offsets.at("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"));
    auto& post_ln_tensor = header_json.at("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight");
    size_t post_layer_start = static_cast<size_t>(post_ln_tensor["data_offsets"][0]);
    size_t post_layer_end = static_cast<size_t>(post_ln_tensor["data_offsets"][1]);

    weights.w_k[i] = (bf16*)(&model_weights_cpu[0]  + offsets.at("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight"));
    weights.w_o[i] = (bf16*)(&model_weights_cpu[0] + offsets.at("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight"));
    weights.w_q[i] = (bf16*)(&model_weights_cpu[0]  + offsets.at("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight"));
    weights.w_v[i] = (bf16*)(&model_weights_cpu[0]  + offsets.at("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight"));


    quantizeMatrix("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight", header_json, scales, qt_model_weights_cpu, weights.mlp_gate_proj[i], quantized_offsets, scale_offsets);
    quantizeMatrix("model.layers." + std::to_string(i) + ".mlp.up_proj.weight", header_json, scales, qt_model_weights_cpu, weights.mlp_up_proj[i], quantized_offsets, scale_offsets);
    quantizeMatrix("model.layers." + std::to_string(i) + ".mlp.down_proj.weight", header_json, scales, qt_model_weights_cpu, weights.mlp_down_proj[i], quantized_offsets, scale_offsets);
    quantizeMatrix("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight", header_json, scales, qt_model_weights_cpu, weights.w_q[i], quantized_offsets, scale_offsets);
    quantizeMatrix("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight", header_json, scales, qt_model_weights_cpu, weights.w_k[i], quantized_offsets, scale_offsets);
    quantizeMatrix("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight", header_json, scales, qt_model_weights_cpu, weights.w_v[i], quantized_offsets, scale_offsets);
    quantizeMatrix("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight", header_json, scales, qt_model_weights_cpu, weights.w_o[i], quantized_offsets, scale_offsets);


    size_t post_ln_bytes = (post_layer_end - post_layer_start);
    size_t ln_bytes = (layer_end - layer_start);

    // CHECK
    if(qt_model_weights_cpu.size() % alignof(bf16)) qt_model_weights_cpu.push_back(0);

    size_t offset = qt_model_weights_cpu.size();
    quantized_offsets["model.layers." + std::to_string(i) + ".input_layernorm.weight"] = {offset, offset + ln_bytes};
    qt_model_weights_cpu.resize(offset + ln_bytes);
    std::memcpy(qt_model_weights_cpu.data() + offset, model_weights_cpu.data() + layer_start, ln_bytes);

    if(qt_model_weights_cpu.size() % alignof(bf16)) qt_model_weights_cpu.push_back(0);
    offset = qt_model_weights_cpu.size();
    quantized_offsets["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"] = {offset, offset + post_ln_bytes};
    qt_model_weights_cpu.resize(offset + post_ln_bytes);
    std::memcpy(qt_model_weights_cpu.data() + offset, model_weights_cpu.data() + post_layer_start, post_ln_bytes);
  }


  auto& embed_tensor = header_json.at("model.embed_tokens.weight");
  size_t embed_start = static_cast<size_t>(embed_tensor["data_offsets"][0]);
  size_t embed_end = static_cast<size_t>(embed_tensor["data_offsets"][1]);
  size_t embed_bytes = embed_end - embed_start;
  if(qt_model_weights_cpu.size() % alignof(bf16)) qt_model_weights_cpu.push_back(0);
  size_t embed_offset = qt_model_weights_cpu.size();
  quantized_offsets["model.embed_tokens.weight"] = {embed_offset, embed_offset + embed_bytes};
  qt_model_weights_cpu.resize(embed_offset + embed_bytes);
  std::memcpy(qt_model_weights_cpu.data() + embed_offset, model_weights_cpu.data() + embed_start, embed_bytes);

  auto& norm_tensor = header_json.at("model.norm.weight");
  size_t norm_start = static_cast<size_t>(norm_tensor["data_offsets"][0]);
  size_t norm_end = static_cast<size_t>(norm_tensor["data_offsets"][1]);
  size_t norm_bytes = norm_end - norm_start;
  if(qt_model_weights_cpu.size() % alignof(bf16)) qt_model_weights_cpu.push_back(0);
  size_t norm_offset = qt_model_weights_cpu.size();
  quantized_offsets["model.norm.weight"] = {norm_offset, norm_offset + norm_bytes};
  qt_model_weights_cpu.resize(norm_offset + norm_bytes);
  std::memcpy(qt_model_weights_cpu.data() + norm_offset, model_weights_cpu.data() + norm_start, norm_bytes);


  quantized_weights.embed_tokens = (bf16*)(qt_model_weights_cpu.data() + quantized_offsets.at("model.embed_tokens.weight").first);
  quantized_weights.norm = (bf16*)(qt_model_weights_cpu.data() + quantized_offsets.at("model.norm.weight").first);
  for (int i = 0; i < N_LAYERS; ++i) {
    std::string layer = "model.layers." + std::to_string(i) + ".";
    quantized_weights.input_layernorm[i] = (bf16*)(qt_model_weights_cpu.data() + quantized_offsets.at(layer + "input_layernorm.weight").first);
    quantized_weights.post_attn_layernorms[i] = (bf16*)(qt_model_weights_cpu.data() + quantized_offsets.at(layer + "post_attention_layernorm.weight").first);
    quantized_weights.mlp_down_proj[i] = (int8_t*)(qt_model_weights_cpu.data() + quantized_offsets.at(layer + "mlp.down_proj.weight").first);
    quantized_weights.mlp_down_proj_scales[i] = scales.data() + scale_offsets.at(layer + "mlp.down_proj.weight");
    quantized_weights.mlp_gate_proj[i] = (int8_t*)(qt_model_weights_cpu.data() + quantized_offsets.at(layer + "mlp.gate_proj.weight").first);
    quantized_weights.mlp_gate_proj_scales[i] = scales.data() + scale_offsets.at(layer + "mlp.gate_proj.weight");
    quantized_weights.mlp_up_proj[i] = (int8_t*)(qt_model_weights_cpu.data() + quantized_offsets.at(layer + "mlp.up_proj.weight").first);
    quantized_weights.mlp_up_proj_scales[i] = scales.data() + scale_offsets.at(layer + "mlp.up_proj.weight");
    quantized_weights.w_k[i] = (int8_t*)(qt_model_weights_cpu.data() + quantized_offsets.at(layer + "self_attn.k_proj.weight").first);
    quantized_weights.w_k_scales[i] = scales.data() + scale_offsets.at(layer + "self_attn.k_proj.weight");
    quantized_weights.w_o[i] = (int8_t*)(qt_model_weights_cpu.data() + quantized_offsets.at(layer + "self_attn.o_proj.weight").first);
    quantized_weights.w_o_scales[i] = scales.data() + scale_offsets.at(layer + "self_attn.o_proj.weight");
    quantized_weights.w_q[i] = (int8_t*)(qt_model_weights_cpu.data() + quantized_offsets.at(layer + "self_attn.q_proj.weight").first);
    quantized_weights.w_q_scales[i] = scales.data() + scale_offsets.at(layer + "self_attn.q_proj.weight");
    quantized_weights.w_v[i] = (int8_t*)(qt_model_weights_cpu.data() + quantized_offsets.at(layer + "self_attn.v_proj.weight").first);
    quantized_weights.w_v_scales[i] = scales.data() + scale_offsets.at(layer + "self_attn.v_proj.weight");
  }
  std::vector<header_entry> header_package;


  addHeaderEntry("model.embed_tokens.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
  addHeaderEntry("model.norm.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
  for(int i=0; i<N_LAYERS; ++i){
    std::string layer = "model.layers." + std::to_string(i) + ".";
    addHeaderEntry(layer + "input_layernorm.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
    addHeaderEntry(layer + "post_attention_layernorm.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
    addHeaderEntry(layer + "mlp.down_proj.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
    addHeaderEntry(layer + "mlp.gate_proj.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
    addHeaderEntry(layer + "mlp.up_proj.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
    addHeaderEntry(layer + "self_attn.k_proj.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
    addHeaderEntry(layer + "self_attn.o_proj.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
    addHeaderEntry(layer + "self_attn.q_proj.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
    addHeaderEntry(layer + "self_attn.v_proj.weight", header_json, quantized_offsets, scale_offsets, qt_model_weights_cpu.size(), header_package);
  }
  json output_header = json::object();

  for(size_t i=0; i<header_package.size(); ++i){
    output_header[header_package[i].name] = {
      {"dtype", header_package[i].dtype},
      {"shape", header_package[i].shape},
      {"data_offsets", {header_package[i].offsets.first, header_package[i].offsets.second}}
    };
  }

  std::string output_header_text = output_header.dump();
  output_header_text.append((8 - output_header_text.size() % 8) % 8, ' ');
  uint64_t output_header_size = output_header_text.size();

  std::ofstream file("quantized.safetensor", std::ios::binary);
  if(!file.is_open()) {
    std::cerr << "Cannot open quantized.safetensor for writing\n";
    return 1;
  }

  for(int i=0; i<8; ++i) {
    file.put(static_cast<char>((output_header_size >> (8 * i)) & 0xff));
  }
  file.write(output_header_text.data(), output_header_text.size());
  file.write(qt_model_weights_cpu.data(), qt_model_weights_cpu.size());
  file.write(reinterpret_cast<const char*>(scales.data()), scales.size() * sizeof(fp32));
  file.close();
  if(!file) {
    std::cerr << "Failed to write quantized.safetensor\n";
    return 1;
  }
  return 0;
}



int loadWeights_quantized(QuantizedWeights &quantized_weights,
                          fs::path model_path) {
  if (checkGPUStatus())
    return 1;

  fs::path quantized_path = "quantized.safetensor";
  if (!fs::exists(quantized_path)) {
    if (quantWeights(model_path))
      return 1;
  }

  std::ifstream safetensors_file(quantized_path, std::ios::binary);
  if (!safetensors_file.is_open()) {
    std::cerr << "Cannot open quantized weights file\n";
    return 1;
  }

  uint64_t header_size;
  safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);

  std::string header;
  header.resize(header_size);
  safetensors_file.read(header.data(), header_size);
  json header_json = json::parse(header);

  std::unordered_map<std::string, size_t> offsets;
  size_t max_offset = 0;

  for (auto &[name, tensor] : header_json.items()) {
    if (name == "__metadata__")
      continue;

    size_t end = tensor["data_offsets"].at(1).get<size_t>();
    if (end > max_offset)
      max_offset = end;

    offsets[name] = tensor["data_offsets"].at(0).get<size_t>();
  }

  std::vector<char> model_weights_cpu(max_offset);
  safetensors_file.read(model_weights_cpu.data(), max_offset);
  safetensors_file.close();

  void *model_weights;
  cudaMalloc(&model_weights, max_offset);
  cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset,
             cudaMemcpyHostToDevice);

  char *data = static_cast<char *>(model_weights);

  quantized_weights.embed_tokens =
      (bf16 *)(data + offsets.at("model.embed_tokens.weight"));
  quantized_weights.norm = (bf16 *)(data + offsets.at("model.norm.weight"));

  for (int i = 0; i < N_LAYERS; ++i) {
    std::string layer = "model.layers." + std::to_string(i) + ".";

    quantized_weights.input_layernorm[i] =
        (bf16 *)(data + offsets.at(layer + "input_layernorm.weight"));
    quantized_weights.post_attn_layernorms[i] =
        (bf16 *)(data + offsets.at(layer + "post_attention_layernorm.weight"));

    quantized_weights.mlp_down_proj[i] =
        (int8_t *)(data + offsets.at(layer + "mlp.down_proj.weight"));
    quantized_weights.mlp_down_proj_scales[i] =
        (fp32 *)(data + offsets.at(layer + "mlp.down_proj.weight.scale"));

    quantized_weights.mlp_gate_proj[i] =
        (int8_t *)(data + offsets.at(layer + "mlp.gate_proj.weight"));
    quantized_weights.mlp_gate_proj_scales[i] =
        (fp32 *)(data + offsets.at(layer + "mlp.gate_proj.weight.scale"));

    quantized_weights.mlp_up_proj[i] =
        (int8_t *)(data + offsets.at(layer + "mlp.up_proj.weight"));
    quantized_weights.mlp_up_proj_scales[i] =
        (fp32 *)(data + offsets.at(layer + "mlp.up_proj.weight.scale"));

    quantized_weights.w_k[i] =
        (int8_t *)(data + offsets.at(layer + "self_attn.k_proj.weight"));
    quantized_weights.w_k_scales[i] =
        (fp32 *)(data + offsets.at(layer + "self_attn.k_proj.weight.scale"));

    quantized_weights.w_o[i] =
        (int8_t *)(data + offsets.at(layer + "self_attn.o_proj.weight"));
    quantized_weights.w_o_scales[i] =
        (fp32 *)(data + offsets.at(layer + "self_attn.o_proj.weight.scale"));

    quantized_weights.w_q[i] =
        (int8_t *)(data + offsets.at(layer + "self_attn.q_proj.weight"));
    quantized_weights.w_q_scales[i] =
        (fp32 *)(data + offsets.at(layer + "self_attn.q_proj.weight.scale"));

    quantized_weights.w_v[i] =
        (int8_t *)(data + offsets.at(layer + "self_attn.v_proj.weight"));
    quantized_weights.w_v_scales[i] =
        (fp32 *)(data + offsets.at(layer + "self_attn.v_proj.weight.scale"));
  }

  return 0;
}

int Buffers::allocate(bool score_mode) {
  // persistent, one independent span per slot
  size_t kv_bytes = (size_t)MAX_SEQUENCES * N_LAYERS * 2 * MAX_SEQ_LEN * KV_DIM * sizeof(bf16);
  cudaMalloc(&kv_cache, kv_bytes);
  cudaMemset(kv_cache, 0, kv_bytes);

  // residual stream scratch
  cudaMalloc(&gpu_input_tokens, (size_t)MAX_PROMPT_LEN * sizeof(int));
  cudaMalloc(&gpu_positions,    (size_t)MAX_SEQUENCES * sizeof(int));
  cudaMalloc(&gpu_slots,        (size_t)MAX_SEQUENCES * sizeof(int));
  cudaMalloc(&hidden_state,     (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));
  cudaMalloc(&rms_norms,        (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));

  // attention scratch
  cudaMalloc(&q_proj,      (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));
  cudaMalloc(&k_proj,      (size_t)MAX_PROMPT_LEN * KV_DIM * sizeof(bf16));
  cudaMalloc(&v_proj,      (size_t)MAX_PROMPT_LEN * KV_DIM * sizeof(bf16));
  cudaMalloc(&attn_scores, (size_t)MAX_SEQUENCES * NUM_Q_HEADS * MAX_SEQ_LEN * sizeof(bf16));
  cudaMalloc(&attn_out,    (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));
  cudaMalloc(&o_proj,      (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));

  size_t batch_ptrs = (size_t)MAX_SEQUENCES * NUM_K_HEADS;
  cudaMalloc(&gpu_k_ptrs, (size_t)N_LAYERS * batch_ptrs * sizeof(void*));
  cudaMalloc(&gpu_v_ptrs, (size_t)N_LAYERS * batch_ptrs * sizeof(void*));
  cudaMalloc(&gpu_q_ptrs, batch_ptrs * sizeof(void*));
  cudaMalloc(&gpu_s_ptrs, batch_ptrs * sizeof(void*));
  cudaMalloc(&gpu_o_ptrs, batch_ptrs * sizeof(void*));

  // mlp scratch
  cudaMalloc(&gate, (size_t)MAX_PROMPT_LEN * INTERMEDIATE_DIM * sizeof(bf16));
  cudaMalloc(&up,   (size_t)MAX_PROMPT_LEN * INTERMEDIATE_DIM * sizeof(bf16));
  cudaMalloc(&down, (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));

  // output. decode needs one logit row per active slot, prefill only uses row
  // 0. scoring needs every prompt position at once, which is 128x larger
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
  cudaFree(gpu_input_tokens); cudaFree(gpu_positions); cudaFree(gpu_slots);
  cudaFree(hidden_state); cudaFree(rms_norms);
  cudaFree(q_proj); cudaFree(k_proj); cudaFree(v_proj);
  cudaFree(attn_scores); cudaFree(attn_out); cudaFree(o_proj);
  cudaFree(gpu_k_ptrs); cudaFree(gpu_v_ptrs);
  cudaFree(gpu_q_ptrs); cudaFree(gpu_s_ptrs); cudaFree(gpu_o_ptrs);
  cudaFree(gate); cudaFree(up); cudaFree(down);
  cudaFree(logits); cudaFree(gpu_sampled_tokens); cudaFree(gpu_rand);
  cudaFree(gpu_targets); cudaFree(gpu_logprobs);
}
