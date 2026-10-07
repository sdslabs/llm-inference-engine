#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

using bf16 = __nv_bfloat16;

// unit conversion
constexpr int B_TO_MB = 1024 * 1024;
constexpr int B_TO_GB = 1024 * 1024 * 1024;

// model architecture. 
constexpr int N_LAYERS = 16;
constexpr int EMBEDDING_LENGTH = 2048;   // model dim, width of the residual stream
constexpr int E_DIM = EMBEDDING_LENGTH;  // should be Even
constexpr int INTERMEDIATE_DIM = 8192;   // MLP width, gate and up project up to this
constexpr int KV_DIM = 512;              // NUM_K_HEADS * HEAD_DIM, width of one K or V row
constexpr int HEAD_DIM = 64;             // per head width, E_DIM / NUM_Q_HEADS
constexpr float SQRT_HEAD_DIM = 8;       // QK scores are divided by this before softmax
constexpr int VOCAB_SIZE = 128256;       // logits width, embed_tokens is reused as the output projection

// grouped query attention, 4 query heads share each K and V head
constexpr int NUM_Q_HEADS = 32;
constexpr int NUM_K_HEADS = 8;
constexpr int GQA_Q_TO_K_RATIO = 4;            // query head index / this = its K head index

// .env is the only definition of these. nob passes them all as -D defines
#ifndef ENGINE_MAX_SEQUENCES
#error "ENGINE_MAX_SEQUENCES not defined, build with ./nob"
#endif
#ifndef ENGINE_MAX_PROMPT_LEN
#error "ENGINE_MAX_PROMPT_LEN not defined, build with ./nob"
#endif
#ifndef ENGINE_MAX_SEQ_LEN
#error "ENGINE_MAX_SEQ_LEN not defined, build with ./nob"
#endif
#ifndef ENGINE_MAX_NUM_THREAD
#error "ENGINE_MAX_NUM_THREAD not defined, build with ./nob"
#endif
#ifndef ENGINE_MAX_TOP_K
#error "ENGINE_MAX_TOP_K not defined, build with ./nob"
#endif
#ifndef ENGINE_DEFAULT_TOP_K
#error "ENGINE_DEFAULT_TOP_K not defined, build with ./nob"
#endif
#ifndef ENGINE_DEFAULT_TEMPERATURE
#error "ENGINE_DEFAULT_TEMPERATURE not defined, build with ./nob"
#endif
#ifndef ENGINE_DEFAULT_MAX_NEW_TOKENS
#error "ENGINE_DEFAULT_MAX_NEW_TOKENS not defined, build with ./nob"
#endif

constexpr int TOP_K = ENGINE_DEFAULT_TOP_K;

constexpr int BOS_TOKEN_ID = 128000;         // <|begin_of_text|>, prepended to every prompt
constexpr int END_OF_TEXT_TOKEN_ID = 128001; // <|end_of_text|>, stops decode
constexpr int EOT_ID_TOKEN_ID = 128009;      // <|eot_id|>, stops decode

// sizes fixed arrays and shared memory, so these stay compile time
constexpr int MAX_SEQUENCES = ENGINE_MAX_SEQUENCES;
constexpr int MAX_PROMPT_LEN = ENGINE_MAX_PROMPT_LEN;
constexpr int MAX_SEQ_LEN = ENGINE_MAX_SEQ_LEN;
constexpr int MAX_NUM_THREAD = ENGINE_MAX_NUM_THREAD;
constexpr int MAX_TOP_K = ENGINE_MAX_TOP_K;
constexpr float INF = 1e18;          // stand in for infinity in the causal mask

// starting values for the CLI flags, overridable per run
constexpr int MAX_NEW_TOKENS_GENERATED = ENGINE_DEFAULT_MAX_NEW_TOKENS;
constexpr int DEFAULT_TOP_K = ENGINE_DEFAULT_TOP_K;
constexpr float TEMPERATURE = ENGINE_DEFAULT_TEMPERATURE;

static_assert(MAX_SEQ_LEN <= MAX_NUM_THREAD,
              "decodeSoftmax reduces one row per block, so a sequence cannot be "
              "longer than one block of threads");
static_assert(MAX_PROMPT_LEN <= MAX_NUM_THREAD,
              "softmax and causalMask launch one thread per prompt position");
static_assert(MAX_PROMPT_LEN <= MAX_SEQ_LEN,
              "a prompt has to fit in the KV cache");
static_assert((size_t)MAX_PROMPT_LEN * MAX_PROMPT_LEN >= (size_t)MAX_SEQUENCES * MAX_SEQ_LEN,
              "attn_scores is sized for prefill but decode needs "
              "MAX_SEQUENCES * NUM_Q_HEADS * MAX_SEQ_LEN");
static_assert(E_DIM / 2 <= MAX_NUM_THREAD,
              "rmsNorm, embeddingGather and residualAdd launch E_DIM/2 threads");
static_assert(INTERMEDIATE_DIM % MAX_NUM_THREAD == 0,
              "siluKernel strides by MAX_NUM_THREAD up to INTERMEDIATE_DIM");
static_assert(DEFAULT_TOP_K >= 1 && DEFAULT_TOP_K <= MAX_TOP_K,
              "DEFAULT_TOP_K must be a value --top-k would accept");
static_assert(MAX_TOP_K <= MAX_NUM_THREAD,
              "topKSample reduces over one block per row");
static_assert(TEMPERATURE > 0.0f, "temperature divides the logits");
static_assert(MAX_NEW_TOKENS_GENERATED >= 1, "must generate at least one token");

inline float *d_cos_table = nullptr; // [max_seq_len, head_dim]
inline float *d_sin_table = nullptr; // [max_seq_len, head_dim]


// prefill
void embeddingGather(int* gpu_input_tokens, __nv_bfloat16* gpu_input_embeds, __nv_bfloat16* embed_tokens, int num_input_tokens);
void rmsNorm(bf16* input, bf16* output, bf16* norm_weights, int num_tokens);
void setRmsNormEps(float eps);
void silu(bf16* a, bf16* b, int num_tokens);
void residualAdd(bf16* input, bf16* residual, int num_tokens);
void softmax(bf16* input, int num_tokens);
void init_rope_frequencies(int head_dim, int max_seq_len, float rope_theta,float factor, float low_freq_factor,float high_freq_factor, int original_max_len);
void free_rope_frequencies(void);
void rope(bf16* input, int num_tokens, int proj_dim);
void causalMask(bf16* input, int num_tokens);

// decode
void decodeSoftmax(bf16* attention_scores, int seq_len);
void ropeDecode(bf16* input, const int* positions, int num_rows, int proj_dim);

void topKSample(const bf16* logits, int* sampled_tokens, const float* uniform_rand,
                int num_rows, int k, float temperature);

void logProbs(const bf16* logits, const int* targets, float* out_logprobs, int num_rows);
