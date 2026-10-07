#pragma once
#include <cublas_v2.h>
#include <ostream>
#include <string>
#include <vector>
#include "runtime.h"

int scoreSequence(const std::vector<int>& tokens, const std::string& id,
                  Weights& weights, cublasHandle_t cublas_handle, Buffers& buf,
                  std::ostream& out);
