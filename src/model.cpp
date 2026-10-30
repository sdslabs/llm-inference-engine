#include "model.h"
#include <iostream>
#include <random>

static bf16* slotCache(bf16* kv_cache, int slot) {
  return kv_cache + (size_t)slot * N_LAYERS * 2 * MAX_SEQ_LEN * KV_DIM;
}

static bf16* layerK(bf16* slot_cache, int layer) {
  return slot_cache + (size_t)2 * layer * MAX_SEQ_LEN * KV_DIM;
}

static bf16* layerV(bf16* slot_cache, int layer) {
  return layerK(slot_cache, layer) + (size_t)MAX_SEQ_LEN * KV_DIM;
}

static std::mt19937_64 rng;
void seedSampler(unsigned long long seed) {
  rng.seed(seed);
}

static void sampleTokens(const bf16* logits, int* gpu_sampled_tokens, float* gpu_rand,
                         int num_rows, int top_k, float temperature, std::vector<int>& tokens) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> rand_cpu(num_rows);
  for(int i=0; i<num_rows; i++) rand_cpu[i] = dist(rng);

  cudaMemcpy(gpu_rand, rand_cpu.data(), num_rows*sizeof(float), cudaMemcpyHostToDevice);
  topKSample(logits, gpu_sampled_tokens, gpu_rand, num_rows, top_k, temperature);

  tokens.resize(num_rows);
  cudaMemcpy(tokens.data(), gpu_sampled_tokens, num_rows*sizeof(int), cudaMemcpyDeviceToHost);
}

static std::vector<int> stop_token_ids = {END_OF_TEXT_TOKEN_ID, EOT_ID_TOKEN_ID};

void setStopTokens(const std::vector<int>& ids) {
  if(!ids.empty()) stop_token_ids = ids;
}

bool isStopToken(int token) {
  for(int id : stop_token_ids) if(token == id) return true;
  return false;
}

const char* finishReason(const SlotState& s, int max_new_tokens) {
  if(s.stopped) return "stop";
  if(s.generated.empty()) return nullptr;
  if((int)s.generated.size() >= max_new_tokens) return "length";
  if(s.seq_len >= MAX_SEQ_LEN) return "context";
  return nullptr;
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

// embedding -> N_LAYERS -> final rmsNorm. leaves the normalized hidden states
// for every prompt position in buf.rms_norms, one E_DIM row per token.
void forwardPrefill(const std::vector<int>& prompt, int prompt_len, int slot,
                    Weights& weights, cublasHandle_t cublas_handle, Buffers& buf) {
  bf16* kv_cache = buf.kv_cache;
  int* gpu_input_tokens = buf.gpu_input_tokens;
  bf16* hidden_state = buf.hidden_state;
  bf16* rms_norms = buf.rms_norms;
  bf16* q_proj = buf.q_proj;
  bf16* k_proj = buf.k_proj;
  bf16* v_proj = buf.v_proj;
  bf16* attn_out = buf.attn_out;
  bf16* o_proj = buf.o_proj;
  bf16* gate = buf.gate;
  bf16* up = buf.up;
  bf16* down = buf.down;

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

    flashAttentionPrefillWmma(q_proj, k_proj, v_proj, attn_out, prompt_len);

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
}

void prefill(std::vector<int>& prompt, int prompt_len, int slot, const Config& cfg,
             Weights& weights, cublasHandle_t cublas_handle, Buffers& buf,
             std::vector<SlotState>& slots) {
  if(prompt_len > MAX_PROMPT_LEN) {
    std::cerr << "Prompt of " << prompt_len << " tokens exceeds MAX_PROMPT_LEN\n";
    return;
  }

  forwardPrefill(prompt, prompt_len, slot, weights, cublas_handle, buf);

  bf16* last = buf.rms_norms + (prompt_len-1)*E_DIM;
  linear(cublas_handle, last, weights.embed_tokens, buf.logits, 1, E_DIM, VOCAB_SIZE);

  std::vector<int> sampled;
  sampleTokens(buf.logits, buf.gpu_sampled_tokens, buf.gpu_rand, 1, cfg.top_k, cfg.temperature, sampled);

  slots[slot].active = true;
  slots[slot].stopped = isStopToken(sampled[0]);
  slots[slot].prompt_len = prompt_len;
  slots[slot].seq_len = prompt_len;
  slots[slot].last_token = sampled[0];
  slots[slot].generated.clear();
  if(!slots[slot].stopped) slots[slot].generated.push_back(sampled[0]);
}

void decode(std::vector<SlotState>& slots, const Config& cfg,
            Weights& weights, cublasHandle_t cublas_handle, Buffers& buf) {
  bf16* kv_cache = buf.kv_cache;
  int* gpu_input_tokens = buf.gpu_input_tokens;
  int* gpu_positions = buf.gpu_positions;
  int* gpu_slots = buf.gpu_slots;
  void** gpu_k_ptrs = buf.gpu_k_ptrs;
  void** gpu_v_ptrs = buf.gpu_v_ptrs;
  void** gpu_q_ptrs = buf.gpu_q_ptrs;
  void** gpu_s_ptrs = buf.gpu_s_ptrs;
  void** gpu_o_ptrs = buf.gpu_o_ptrs;
  bf16* hidden_state = buf.hidden_state;
  bf16* rms_norms = buf.rms_norms;
  bf16* q_proj = buf.q_proj;
  bf16* k_proj = buf.k_proj;
  bf16* v_proj = buf.v_proj;
  bf16* attn_scores = buf.attn_scores;
  bf16* attn_out = buf.attn_out;
  bf16* o_proj = buf.o_proj;
  bf16* gate = buf.gate;
  bf16* up = buf.up;
  bf16* down = buf.down;
  bf16* logits = buf.logits;

  std::vector<int> active; // active[b] is the slot that row b belongs to
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

  int max_len = 0;
  for(int pos : batch_positions) if(pos + 1 > max_len) max_len = pos + 1;

  const float scale = 1.0f / SQRT_HEAD_DIM;
  const float zero = 0.0f, one = 1.0f;

  cudaMemcpy(gpu_input_tokens, batch_tokens.data(), B*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_positions, batch_positions.data(), B*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_slots, active.data(), B*sizeof(int), cudaMemcpyHostToDevice);
  embeddingGather(gpu_input_tokens, hidden_state, weights.embed_tokens, B);

  int batch_ptrs = B*NUM_K_HEADS;
  std::vector<void*> h_q(batch_ptrs), h_s(batch_ptrs), h_o(batch_ptrs);
  std::vector<void*> h_k((size_t)N_LAYERS*batch_ptrs), h_v((size_t)N_LAYERS*batch_ptrs);

  for(int b=0; b<B; b++) {
    for(int g=0; g<NUM_K_HEADS; g++) {
      int i = b*NUM_K_HEADS + g;
      h_q[i] = q_proj      + (size_t)b*E_DIM + (size_t)g*GQA_Q_TO_K_RATIO*HEAD_DIM;
      h_s[i] = attn_scores + (size_t)b*NUM_Q_HEADS*MAX_SEQ_LEN + (size_t)g*GQA_Q_TO_K_RATIO*MAX_SEQ_LEN;
      h_o[i] = attn_out    + (size_t)b*E_DIM + (size_t)g*GQA_Q_TO_K_RATIO*HEAD_DIM;
    }
  }

  for(int layer=0; layer<N_LAYERS; layer++) {
    for(int b=0; b<B; b++) {
      bf16* sc = slotCache(kv_cache, active[b]);
      for(int g=0; g<NUM_K_HEADS; g++) {
        size_t i = (size_t)layer*batch_ptrs + b*NUM_K_HEADS + g;
        h_k[i] = layerK(sc, layer) + (size_t)g*HEAD_DIM;
        h_v[i] = layerV(sc, layer) + (size_t)g*HEAD_DIM;
      }
    }
  }

  cudaMemcpy(gpu_q_ptrs, h_q.data(), batch_ptrs*sizeof(void*), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_s_ptrs, h_s.data(), batch_ptrs*sizeof(void*), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_o_ptrs, h_o.data(), batch_ptrs*sizeof(void*), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_k_ptrs, h_k.data(), h_k.size()*sizeof(void*), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_v_ptrs, h_v.data(), h_v.size()*sizeof(void*), cudaMemcpyHostToDevice);

  for(int layer=0; layer<N_LAYERS; layer++) {
    rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], B);

    linear(cublas_handle, rms_norms, weights.w_q[layer], q_proj, B, E_DIM, E_DIM);
    linear(cublas_handle, rms_norms, weights.w_k[layer], k_proj, B, E_DIM, KV_DIM);
    linear(cublas_handle, rms_norms, weights.w_v[layer], v_proj, B, E_DIM, KV_DIM);

    ropeDecode(q_proj, gpu_positions, B, E_DIM);
    ropeDecode(k_proj, gpu_positions, B, KV_DIM);

    kvCacheScatter(kv_cache, k_proj, v_proj, gpu_slots, gpu_positions, B, layer);

    void** k_ptrs = gpu_k_ptrs + (size_t)layer*batch_ptrs;
    void** v_ptrs = gpu_v_ptrs + (size_t)layer*batch_ptrs;

    cublasGemmBatchedEx(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        max_len, GQA_Q_TO_K_RATIO, HEAD_DIM,
        &scale,
        (const void* const*)k_ptrs,     CUDA_R_16BF, KV_DIM,
        (const void* const*)gpu_q_ptrs, CUDA_R_16BF, HEAD_DIM,
        &zero,
        (void* const*)gpu_s_ptrs,       CUDA_R_16BF, MAX_SEQ_LEN,
        batch_ptrs, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    decodeSoftmax(attn_scores, gpu_positions, B, max_len);

    cublasGemmBatchedEx(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HEAD_DIM, GQA_Q_TO_K_RATIO, max_len,
        &one,
        (const void* const*)v_ptrs,     CUDA_R_16BF, KV_DIM,
        (const void* const*)gpu_s_ptrs, CUDA_R_16BF, MAX_SEQ_LEN,
        &zero,
        (void* const*)gpu_o_ptrs,       CUDA_R_16BF, HEAD_DIM,
        batch_ptrs, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

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

  std::vector<int> sampled;
  sampleTokens(logits, buf.gpu_sampled_tokens, buf.gpu_rand, B, cfg.top_k, cfg.temperature, sampled);

  for(int b=0; b<B; b++) {
    int s = active[b];
    if(isStopToken(sampled[b])) {
      slots[s].stopped = true;
      continue;
    }
    slots[s].seq_len += 1;
    slots[s].last_token = sampled[b];
    slots[s].generated.push_back(sampled[b]);
  }
}
