#pragma once
#include <cublas_v2.h>
#include <filesystem>
#include "kernels.cuh"

int checkGPUStatus();

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

  bf16* logits = nullptr;               // one row per active slot
  int* gpu_sampled_tokens = nullptr;
  float* gpu_rand = nullptr;

  int allocate();
  void free();
};
