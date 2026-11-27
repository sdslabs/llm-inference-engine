#include "model.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

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
  int* gpu_block_table = buf.gpu_block_table;
  int* gpu_input_tokens = buf.gpu_input_tokens;
  bf16* kv_cache = buf.kv_cache;
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
  int* gpu_positions = buf.prefill_gpu_positions;
  int* gpu_slots = buf.prefill_gpu_slots;

  static std::vector<int> token_slots(MAX_PROMPT_LEN);
  std::fill(token_slots.begin(), token_slots.begin() + prompt_len, slot);
  cudaMemcpy(gpu_slots, token_slots.data(), prompt_len*sizeof(int), cudaMemcpyHostToDevice);

  static std::vector<int> token_positions = []() {
      std::vector<int> v(MAX_PROMPT_LEN);
      std::iota(v.begin(), v.end(), 0); 
      return v;
  }();
  cudaMemcpy(gpu_positions, token_positions.data(), prompt_len*sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(gpu_input_tokens, prompt.data(), prompt_len*sizeof(int), cudaMemcpyHostToDevice);
  embeddingGather(gpu_input_tokens, hidden_state, weights.embed_tokens, prompt_len);

  for(int layer=0; layer<N_LAYERS; layer++) {
    rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], prompt_len);

    linear(cublas_handle, rms_norms, weights.w_q[layer], q_proj, prompt_len, E_DIM, E_DIM );
    linear(cublas_handle, rms_norms, weights.w_k[layer], k_proj, prompt_len, E_DIM, KV_DIM);
    linear(cublas_handle, rms_norms, weights.w_v[layer], v_proj, prompt_len, E_DIM, KV_DIM);

    rope(q_proj, prompt_len, E_DIM );
    rope(k_proj, prompt_len, KV_DIM);

    kvCacheScatter(kv_cache, gpu_block_table, k_proj, v_proj, layer, gpu_slots, gpu_positions, prompt_len);

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
            Weights& weights, cublasHandle_t cublas_handle, Buffers& buf, DecodeBatch& decode_batch) {
  bf16* kv_cache = buf.kv_cache;
  int* gpu_block_table = buf.gpu_block_table;
  int* gpu_input_tokens = buf.gpu_input_tokens;
  int* gpu_positions = buf.gpu_positions;
  int* gpu_slots = buf.gpu_slots;
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
  bf16* logits = buf.logits;

  std::vector<int> active = decode_batch.active; // active[b] is the slot that row b belongs to
  std::vector<int> batch_tokens = decode_batch.batch_tokens;
  std::vector<int> batch_positions = decode_batch.batch_positions;
  
  int B = active.size();
  if(B == 0) return;

  cudaMemcpy(gpu_input_tokens, batch_tokens.data(), B*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_positions, batch_positions.data(), B*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_slots, active.data(), B*sizeof(int), cudaMemcpyHostToDevice);
  embeddingGather(gpu_input_tokens, hidden_state, weights.embed_tokens, B);

  for(int layer=0; layer<N_LAYERS; layer++) {
    rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], B);

    linear(cublas_handle, rms_norms, weights.w_q[layer], q_proj, B, E_DIM, E_DIM);
    linear(cublas_handle, rms_norms, weights.w_k[layer], k_proj, B, E_DIM, KV_DIM);
    linear(cublas_handle, rms_norms, weights.w_v[layer], v_proj, B, E_DIM, KV_DIM);

    ropeDecode(q_proj, gpu_positions, B, E_DIM);
    ropeDecode(k_proj, gpu_positions, B, KV_DIM);

    kvCacheScatter(kv_cache, gpu_block_table, k_proj, v_proj, layer, gpu_slots, gpu_positions, B);

    pagedGQA(kv_cache, gpu_block_table, q_proj, attn_out, layer, gpu_slots, gpu_positions, B);

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
