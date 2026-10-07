#pragma once
#include <cublas_v2.h>
#include <filesystem>
#include <vector>
#include "kernels.cuh"

int checkGPUStatus();

struct ModelConfig {
  int num_layers = N_LAYERS;
  int hidden_size = E_DIM;
  int intermediate_size = INTERMEDIATE_DIM;
  int num_attention_heads = NUM_Q_HEADS;
  int num_key_value_heads = NUM_K_HEADS;
  int head_dim = HEAD_DIM;
  int vocab_size = VOCAB_SIZE;

  float rms_norm_eps = 1e-5f;

  float rope_theta = 500000.0f;
  float rope_factor = 32.0f;
  float rope_low_freq_factor = 1.0f;
  float rope_high_freq_factor = 4.0f;
  int rope_original_max_position = 8192;

  int bos_token_id = BOS_TOKEN_ID;
  std::vector<int> stop_token_ids = {END_OF_TEXT_TOKEN_ID, EOT_ID_TOKEN_ID};
};

int loadModelConfig(ModelConfig& mc, const std::filesystem::path& model_path);

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

int loadWeights(Weights& weights, std::filesystem::path model_path);

// scratch memory, allocated once at startup and reused by every request
struct Buffers {
  bf16* kv_cache = nullptr;             // MAX_SEQUENCES x N_LAYERS x 2 x MAX_SEQ_LEN x KV_DIM

  int* gpu_input_tokens = nullptr;
  int* gpu_positions = nullptr;         // MAX_SEQUENCES, cache position of each row
  bf16* hidden_state = nullptr;         // carried across all N_LAYERS
  bf16* rms_norms = nullptr;            // rmsNorm output, never in place

  bf16* q_proj = nullptr;
  bf16* k_proj = nullptr;
  bf16* v_proj = nullptr;
  bf16* attn_scores = nullptr;
  bf16* attn_out = nullptr;             // scores @ V
  bf16* o_proj = nullptr;               // after w_o

  bf16* gate = nullptr;
  bf16* up = nullptr;
  bf16* down = nullptr;

  bf16* logits = nullptr;               // one row per active slot, or per prompt token when scoring
  int* gpu_sampled_tokens = nullptr;
  float* gpu_rand = nullptr;

  int* gpu_targets = nullptr;           // score mode only
  float* gpu_logprobs = nullptr;        // score mode only

  int allocate(bool score_mode = false);
  void free();
};
