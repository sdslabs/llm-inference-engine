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

struct SlotState {
  bool active = false;
  int seq_len = 0;
  int last_token = 0;
  std::vector<int> generated;
};

static bf16* slotCache(bf16* kv_cache, int slot) {
  return kv_cache + (size_t)slot * N_LAYERS * 2 * MAX_SEQ_LEN * KV_DIM;
}

static bf16* layerK(bf16* slot_cache, int layer) {
  return slot_cache + (size_t)2 * layer * MAX_SEQ_LEN * KV_DIM;
}
static bf16* layerV(bf16* slot_cache, int layer) {
  return layerK(slot_cache, layer) + (size_t)MAX_SEQ_LEN * KV_DIM;
}

static int argmaxRow(const std::vector<bf16>& logits_cpu, int row) {
  const bf16* p = logits_cpu.data() + (size_t)row * VOCAB_SIZE;
  int best = 0;
  float best_val = (float)p[0];
  for(int i=1; i<VOCAB_SIZE; i++) {
    float v = (float)p[i];
    if(v > best_val) { best_val = v; best = i; }
  }
  return best;
}

static bool isFinished(const SlotState& s) {
  if(s.generated.empty()) return false;
  int t = s.generated.back();
  return t == END_OF_TEXT_TOKEN_ID
      || t == EOT_ID_TOKEN_ID
      || (int)s.generated.size() >= MAX_NEW_TOKENS_GENERATED
      || s.seq_len >= MAX_SEQ_LEN;
}

// Y[T, OUT] = X[T, IN] @ W[OUT, IN]^T, all row major
static void linear(cublasHandle_t h, const bf16* X, const bf16* W, bf16* Y,
                   int T, int IN, int OUT) {
  const float alpha = 1.0f, beta = 0.0f;
  cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
               OUT, T, IN,
               &alpha, W, CUDA_R_16BF, IN,
                       X, CUDA_R_16BF, IN,
               &beta,  Y, CUDA_R_16BF, OUT,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

void prefill(
  // request
  std::vector<int>& prompt, int& prompt_len, int slot,

  // persistent, allocated once at startup
  Weights& weights, cublasHandle_t cublas_handle,
  bf16* kv_cache,                                       // N_LAYERS x 2 x MAX_SEQ_LEN x KV_DIM

  // residual stream scratch
  int* gpu_input_tokens,                                // prompt_len
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
  std::vector<bf16>& logits_cpu,                        // host readback for sampling
  std::vector<SlotState>& slots
) {
  if(prompt_len > MAX_PROMPT_LEN) {
    std::cerr << "Prompt of " << prompt_len << " tokens exceeds MAX_PROMPT_LEN\n";
    return;
  }

  bf16* slot_cache = slotCache(kv_cache, slot);

  cudaMemcpy(gpu_input_tokens, prompt.data(), prompt_len*sizeof(int), cudaMemcpyHostToDevice);
  embeddingGather(gpu_input_tokens, hidden_state, weights.embed_tokens, prompt_len);

  for(int layer=0; layer<N_LAYERS; layer++) {
    rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], prompt_len);

    linear(cublas_handle, rms_norms, weights.w_q[layer], q_proj, prompt_len, E_DIM, E_DIM ); 
    linear(cublas_handle, rms_norms, weights.w_k[layer], k_proj, prompt_len, E_DIM, KV_DIM); 
    linear(cublas_handle, rms_norms, weights.w_v[layer], v_proj, prompt_len, E_DIM, KV_DIM); 

    rope(q_proj, prompt_len, E_DIM );
    rope(k_proj, prompt_len, KV_DIM);

    cudaMemcpy(layerK(slot_cache, layer), k_proj, prompt_len*KV_DIM*sizeof(bf16), cudaMemcpyDeviceToDevice);
    cudaMemcpy(layerV(slot_cache, layer), v_proj, prompt_len*KV_DIM*sizeof(bf16), cudaMemcpyDeviceToDevice);

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
    residualAdd(o_proj, hidden_state, prompt_len);

    cudaMemcpy(hidden_state, o_proj, prompt_len*E_DIM*sizeof(bf16), cudaMemcpyDeviceToDevice);
    rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], prompt_len);

    linear(cublas_handle, rms_norms, weights.mlp_gate_proj[layer], gate, prompt_len, E_DIM, INTERMEDIATE_DIM);
    linear(cublas_handle, rms_norms, weights.mlp_up_proj[layer], up, prompt_len, E_DIM, INTERMEDIATE_DIM);
    silu(gate, up, prompt_len);
    linear(cublas_handle, gate, weights.mlp_down_proj[layer], down, prompt_len, INTERMEDIATE_DIM, E_DIM);

    residualAdd(down, hidden_state, prompt_len);

    cudaMemcpy(hidden_state, down, prompt_len*E_DIM*sizeof(bf16), cudaMemcpyDeviceToDevice);
  }

  rmsNorm(hidden_state, rms_norms, weights.norm, prompt_len);
  bf16* last = rms_norms + (prompt_len-1)*E_DIM;
  linear(cublas_handle, last, weights.embed_tokens, logits, 1, E_DIM, VOCAB_SIZE);

  cudaMemcpy(logits_cpu.data(), logits, VOCAB_SIZE*sizeof(bf16), cudaMemcpyDeviceToHost);

  int best_token = argmaxRow(logits_cpu, 0);

  slots[slot].active = true;
  slots[slot].seq_len = prompt_len;
  slots[slot].last_token = best_token;
  slots[slot].generated.clear();
  slots[slot].generated.push_back(best_token);
}

void decode(
  std::vector<SlotState>& slots,

  Weights& weights, cublasHandle_t cublas_handle,
  bf16* kv_cache,

  int* gpu_input_tokens,
  int* gpu_positions,                                   // MAX_SEQUENCES, cache position of each row
  bf16* hidden_state,                                   // num_active x E_DIM
  bf16* rms_norms,

  bf16* q_proj,                                         // num_active x E_DIM
  bf16* k_proj, bf16* v_proj,                           // num_active x KV_DIM
  bf16* attn_scores,                                    // num_active x NUM_Q_HEADS x MAX_SEQ_LEN
  bf16* attn_out,
  bf16* o_proj,

  bf16* gate, bf16* up,
  bf16* down,

  bf16* logits,                                         // num_active x VOCAB_SIZE
  std::vector<bf16>& logits_cpu
) {
  // pack the active slots into dense batch rows. active[b] is the slot that row b belongs to
  std::vector<int> active;
  std::vector<int> batch_tokens;
  std::vector<int> batch_positions;
  for(int s=0; s<MAX_SEQUENCES; s++) {
    if(!slots[s].active) continue;
    active.push_back(s);
    batch_tokens.push_back(slots[s].last_token);
    batch_positions.push_back(slots[s].seq_len);       // where this token lands in its cache
  }
  int B = active.size();
  if(B == 0) return;

  const float scale = 1.0f / SQRT_HEAD_DIM;
  const float zero = 0.0f, one = 1.0f;

  cudaMemcpy(gpu_input_tokens, batch_tokens.data(), B*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_positions, batch_positions.data(), B*sizeof(int), cudaMemcpyHostToDevice);
  embeddingGather(gpu_input_tokens, hidden_state, weights.embed_tokens, B);

  for(int layer=0; layer<N_LAYERS; layer++) {
    rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], B);

    linear(cublas_handle, rms_norms, weights.w_q[layer], q_proj, B, E_DIM, E_DIM);
    linear(cublas_handle, rms_norms, weights.w_k[layer], k_proj, B, E_DIM, KV_DIM);
    linear(cublas_handle, rms_norms, weights.w_v[layer], v_proj, B, E_DIM, KV_DIM);

    ropeDecode(q_proj, gpu_positions, B, E_DIM);
    ropeDecode(k_proj, gpu_positions, B, KV_DIM);

    for(int b=0; b<B; b++) {
      bf16* sc = slotCache(kv_cache, active[b]);
      size_t off = (size_t)batch_positions[b]*KV_DIM;
      cudaMemcpy(layerK(sc, layer) + off, k_proj + (size_t)b*KV_DIM,
      KV_DIM*sizeof(bf16), cudaMemcpyDeviceToDevice);
      cudaMemcpy(layerV(sc, layer) + off, v_proj + (size_t)b*KV_DIM,
      KV_DIM*sizeof(bf16), cudaMemcpyDeviceToDevice);
    }
    
    for(int b=0; b<B; b++) {
      int len = batch_positions[b] + 1;
      bf16* Sc = slotCache(kv_cache, active[b]);
      bf16* Sb = attn_scores + (size_t)b*NUM_Q_HEADS*MAX_SEQ_LEN;

      for(int h=0; h<NUM_Q_HEADS; h++) {
        const bf16* Qh = q_proj + (size_t)b*E_DIM + h*HEAD_DIM;
        const bf16* Kh = layerK(Sc, layer) + (h/GQA_Q_TO_K_RATIO)*HEAD_DIM;
        bf16* Sh = Sb + (size_t)h*MAX_SEQ_LEN;
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, len, 1, HEAD_DIM,
                     &scale, Kh, CUDA_R_16BF, KV_DIM, Qh, CUDA_R_16BF, E_DIM,
                     &zero,  Sh, CUDA_R_16BF, len,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      }

      decodeSoftmax(Sb, len);

      for(int h=0; h<NUM_Q_HEADS; h++) {
        const bf16* Vh = layerV(Sc, layer) + (h/GQA_Q_TO_K_RATIO)*HEAD_DIM;
        const bf16* Sh = Sb + (size_t)h*MAX_SEQ_LEN;
        bf16* Oh = attn_out + (size_t)b*E_DIM + h*HEAD_DIM;
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HEAD_DIM, 1, len,
                     &one,  Vh, CUDA_R_16BF, KV_DIM, Sh, CUDA_R_16BF, len, &zero,
                     Oh, CUDA_R_16BF, E_DIM,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      }

    }

    linear(cublas_handle, attn_out, weights.w_o[layer], o_proj, B, E_DIM, E_DIM);
    residualAdd(o_proj, hidden_state, B);
    cudaMemcpy(hidden_state, o_proj, B*E_DIM*sizeof(bf16), cudaMemcpyDeviceToDevice);

    rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], B);

    linear(cublas_handle, rms_norms, weights.mlp_gate_proj[layer], gate, B, E_DIM, INTERMEDIATE_DIM);
    linear(cublas_handle, rms_norms, weights.mlp_up_proj[layer], up, B, E_DIM, INTERMEDIATE_DIM);
    silu(gate, up, B);
    linear(cublas_handle, gate, weights.mlp_down_proj[layer], down, B, INTERMEDIATE_DIM, E_DIM);

    residualAdd(down, hidden_state, B);
    cudaMemcpy(hidden_state, down, B*E_DIM*sizeof(bf16), cudaMemcpyDeviceToDevice);
  }

  rmsNorm(hidden_state, rms_norms, weights.norm, B);
  linear(cublas_handle, rms_norms, weights.embed_tokens, logits, B, E_DIM, VOCAB_SIZE);

  cudaMemcpy(logits_cpu.data(), logits, (size_t)B*VOCAB_SIZE*sizeof(bf16), cudaMemcpyDeviceToHost);

  for(int b=0; b<B; b++) {
    int s = active[b];
    int token = argmaxRow(logits_cpu, b);
    slots[s].seq_len += 1;
    slots[s].last_token = token;
    slots[s].generated.push_back(token);
  }
}

int main(int argc, char* argv[]) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.safetensors> [token ids...]\n";
    return 1;
  }

  Weights weights;
  if(loadWeights(weights, argv[1])) return 1;

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  // TODO: read from config.json
  init_rope_frequencies(HEAD_DIM, MAX_SEQ_LEN, 500000.0f, 32.0f, 1.0f, 4.0f, 8192);

  // persistent, one independent span per slot
  bf16* kv_cache;
  cudaMalloc(&kv_cache, (size_t)MAX_SEQUENCES * N_LAYERS * 2 * MAX_SEQ_LEN * KV_DIM * sizeof(bf16));

  // residual stream scratch
  int* gpu_input_tokens;
  int* gpu_positions;
  bf16 *hidden_state, *rms_norms;
  cudaMalloc(&gpu_input_tokens, (size_t)MAX_PROMPT_LEN * sizeof(int));
  cudaMalloc(&gpu_positions,    (size_t)MAX_SEQUENCES * sizeof(int));
  cudaMalloc(&hidden_state,     (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));
  cudaMalloc(&rms_norms,        (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));

  // attention scratch
  bf16 *q_proj, *k_proj, *v_proj, *attn_scores, *attn_out, *o_proj;
  cudaMalloc(&q_proj,      (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));
  cudaMalloc(&k_proj,      (size_t)MAX_PROMPT_LEN * KV_DIM * sizeof(bf16));
  cudaMalloc(&v_proj,      (size_t)MAX_PROMPT_LEN * KV_DIM * sizeof(bf16));
  cudaMalloc(&attn_scores, (size_t)NUM_Q_HEADS * MAX_PROMPT_LEN * MAX_PROMPT_LEN * sizeof(bf16));
  cudaMalloc(&attn_out,    (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));
  cudaMalloc(&o_proj,      (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));

  // mlp scratch
  bf16 *gate, *up, *down;
  cudaMalloc(&gate, (size_t)MAX_PROMPT_LEN * INTERMEDIATE_DIM * sizeof(bf16));
  cudaMalloc(&up,   (size_t)MAX_PROMPT_LEN * INTERMEDIATE_DIM * sizeof(bf16));
  cudaMalloc(&down, (size_t)MAX_PROMPT_LEN * E_DIM * sizeof(bf16));

  // output. decode needs one logit row per active slot, prefill only uses row 0
  bf16* logits;
  cudaMalloc(&logits, (size_t)MAX_SEQUENCES * VOCAB_SIZE * sizeof(bf16));
  std::vector<bf16> logits_cpu((size_t)MAX_SEQUENCES * VOCAB_SIZE);

  if(cudaGetLastError() != cudaSuccess) {
    std::cerr << "Buffer allocation failed\n";
    return 1;
  }

  // TODO: For now every argv prompt is one request
  std::queue<std::vector<int>> pending;
  if(argc > 2) {
    std::vector<int> prompt;
    for(int i=2; i<argc; i++) prompt.push_back(std::stoi(argv[i]));
    pending.push(prompt);
  } else {
    pending.push({128000});  // <|begin_of_text|>
  }

  std::vector<SlotState> slots(MAX_SEQUENCES);

  while(true) {
    for(int s=0; s<MAX_SEQUENCES && !pending.empty(); s++) {
      if(slots[s].active) continue;

      std::vector<int> prompt = pending.front();
      pending.pop();
      int prompt_len = prompt.size();

      prefill(prompt, prompt_len, s,
              weights, cublas_handle, kv_cache,
              gpu_input_tokens, hidden_state, rms_norms,
              q_proj, k_proj, v_proj, attn_scores, attn_out, o_proj,
              gate, up, down,
              logits, logits_cpu, slots);

      if(slots[s].active)
        std::cout << "slot " << s << " prefilled " << prompt_len << " tokens\n";
    }

    int num_active = 0;
    for(const SlotState& s : slots) if(s.active) num_active++;
    if(num_active == 0) break;

    decode(slots,
               weights, cublas_handle, kv_cache,
               gpu_input_tokens, gpu_positions, hidden_state, rms_norms,
               q_proj, k_proj, v_proj, attn_scores, attn_out, o_proj,
               gate, up, down,
               logits, logits_cpu);

    for(int s=0; s<MAX_SEQUENCES; s++) {
      if(!slots[s].active || !isFinished(slots[s])) continue;

      std::cout << "slot " << s << " done, " << slots[s].generated.size() << " tokens:";
      for(int t : slots[s].generated) std::cout << " " << t;
      std::cout << std::endl;

      slots[s] = SlotState{};
    }
  }

  cudaDeviceSynchronize();

  cudaFree(kv_cache);
  cudaFree(gpu_input_tokens); cudaFree(gpu_positions);
  cudaFree(hidden_state); cudaFree(rms_norms);
  cudaFree(q_proj); cudaFree(k_proj); cudaFree(v_proj);
  cudaFree(attn_scores); cudaFree(attn_out); cudaFree(o_proj);
  cudaFree(gate); cudaFree(up); cudaFree(down);
  cudaFree(logits);
  free_rope_frequencies();
  cublasDestroy(cublas_handle);

  return 0;
}
