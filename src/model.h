#pragma once
#include <cublas_v2.h>
#include <string>
#include <vector>
#include "kernels.cuh"
#include "config.h"
#include "runtime.h"

struct SlotState {
  bool active = false;
  bool stopped = false;
  int prompt_len = 0;
  int seq_len = 0;
  int last_token = 0;
  std::vector<int> generated;
  std::string id;
  std::string prompt_text;
};

bool isStopToken(int token);
void setStopTokens(const std::vector<int>& ids);

void seedSampler(unsigned long long seed);
const char* finishReason(const SlotState& s, int max_new_tokens);

void forwardPrefill(const std::vector<int>& prompt, int prompt_len, int slot,
                    Weights& weights, cublasHandle_t cublas_handle, Buffers& buf);

void prefill(std::vector<int>& prompt, int prompt_len, int slot, const Config& cfg,
             Weights& weights, cublasHandle_t cublas_handle, Buffers& buf,
             std::vector<SlotState>& slots);

void decode(std::vector<SlotState>& slots, const Config& cfg,
            Weights& weights, cublasHandle_t cublas_handle, Buffers& buf);
