#pragma once
#include <cublas_v2.h>
#include <string>
#include <vector>
#include "kernels.cuh"
#include "config.h"
#include "runtime.h"

struct SlotState {
  bool active = false;
  int seq_len = 0;
  int last_token = 0;
  std::vector<int> generated;
  std::string prompt_text;
};

void seedSampler(unsigned long long seed);
bool isFinished(const SlotState& s, int max_new_tokens);

void prefill(std::vector<int>& prompt, int prompt_len, int slot, const Config& cfg,
             Weights& weights, cublasHandle_t cublas_handle, Buffers& buf,
             std::vector<SlotState>& slots);

void decode(std::vector<SlotState>& slots, const Config& cfg,
            Weights& weights, cublasHandle_t cublas_handle, Buffers& buf);
