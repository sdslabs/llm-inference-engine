#include "score.h"
#include "kernels.cuh"
#include "model.h"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

using json = nlohmann::json;

int scoreSequence(const std::vector<int>& tokens, const std::string& id,
                  Weights& weights, cublasHandle_t cublas_handle, Buffers& buf,
                  std::ostream& out) {
  int num_tokens = (int)tokens.size();
  int num_targets = num_tokens - 1;

  if(num_targets < 1) {
    std::cerr << "Sequence " << id << " needs at least 2 tokens to score\n";
    return 0;
  }
  if(num_tokens > MAX_PROMPT_LEN) {
    std::cerr << "Sequence " << id << " of " << num_tokens
              << " tokens exceeds MAX_PROMPT_LEN\n";
    return 0;
  }

  auto start = std::chrono::steady_clock::now();

  forwardPrefill(tokens, num_tokens, 0, weights, cublas_handle, buf);

  const float alpha = 1.0f, beta = 0.0f;
  cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
               VOCAB_SIZE, num_targets, E_DIM,
               &alpha, weights.embed_tokens, CUDA_R_16BF, E_DIM,
                       buf.rms_norms,        CUDA_R_16BF, E_DIM,
               &beta,  buf.logits,           CUDA_R_16BF, VOCAB_SIZE,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  cudaMemcpy(buf.gpu_targets, tokens.data() + 1, num_targets*sizeof(int),
             cudaMemcpyHostToDevice);
  logProbs(buf.logits, buf.gpu_targets, buf.gpu_logprobs, num_targets);

  std::vector<float> logprobs(num_targets);
  cudaMemcpy(logprobs.data(), buf.gpu_logprobs, num_targets*sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  double elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();

  double sum = 0.0;
  for(float lp : logprobs) sum += lp;

  json record;
  record["type"] = "score";
  record["id"] = id;
  record["num_tokens"] = num_tokens;
  record["num_scored"] = num_targets;
  record["sum_logprob"] = sum;
  record["mean_logprob"] = sum / num_targets;
  record["logprobs"] = logprobs;
  record["latency_ms"] = elapsed;

  out << record.dump() << "\n";
  out.flush();

  return num_targets;
}
