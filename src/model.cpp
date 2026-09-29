#include "model.h"
#include <cuda_runtime.h>
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

const char* finishReason(const SlotState& s, int max_new_tokens) {
  if(s.generated.empty()) return nullptr;
  int t = s.generated.back();
  if(t == END_OF_TEXT_TOKEN_ID || t == EOT_ID_TOKEN_ID) return "stop";
  if((int)s.generated.size() >= max_new_tokens) return "length";
  if(s.seq_len >= MAX_SEQ_LEN) return "context";
  return nullptr;
}

bool isFinished(const SlotState& s, int max_new_tokens) {
  return finishReason(s, max_new_tokens) != nullptr;
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

void projectLogits(cublasHandle_t cublas_handle, const bf16* hidden, Weights& weights,
                   bf16* logits, int rows) {
  linear(cublas_handle, hidden, weights.embed_tokens, logits, rows, E_DIM, VOCAB_SIZE);
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
  bf16* attn_scores = buf.attn_scores;
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
}

void prefill(std::vector<int>& prompt, int prompt_len, int slot, const Config& cfg,
             Weights& weights, cublasHandle_t cublas_handle, Buffers& buf,
             std::vector<SlotState>& slots) {
  if(prompt_len > MAX_PROMPT_LEN) {
    std::cerr << "Prompt of " << prompt_len << " tokens exceeds MAX_PROMPT_LEN\n";
    return;
  }

  forwardPrefill(prompt, prompt_len, slot, weights, cublas_handle, buf);

  // generation only needs the distribution after the last prompt token
  bf16* last = buf.rms_norms + (prompt_len-1)*E_DIM;
  linear(cublas_handle, last, weights.embed_tokens, buf.logits, 1, E_DIM, VOCAB_SIZE);

  std::vector<int> sampled;
  sampleTokens(buf.logits, buf.gpu_sampled_tokens, buf.gpu_rand, 1, cfg.top_k, cfg.temperature, sampled);

  slots[slot].active = true;
  slots[slot].seq_len = prompt_len;
  slots[slot].last_token = sampled[0];
  slots[slot].generated.clear();
  slots[slot].generated.push_back(sampled[0]);
}

void decode(std::vector<SlotState>& slots, const Config& cfg,
            Weights& weights, cublasHandle_t cublas_handle, Buffers& buf) {
  bf16* kv_cache = buf.kv_cache;
  int* gpu_input_tokens = buf.gpu_input_tokens;
  int* gpu_positions = buf.gpu_positions;
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

  std::vector<int> sampled;
  sampleTokens(logits, buf.gpu_sampled_tokens, buf.gpu_rand, B, cfg.top_k, cfg.temperature, sampled);

  for(int b=0; b<B; b++) {
    int s = active[b];
    slots[s].seq_len += 1;
    slots[s].last_token = sampled[b];
    slots[s].generated.push_back(sampled[b]);
  }
}
