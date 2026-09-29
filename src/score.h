#pragma once
#include <cublas_v2.h>
#include <ostream>
#include <string>
#include <vector>
#include "runtime.h"

// teacher forced scoring. one forward pass over the whole sequence, emits
// log P(tokens[i+1] | tokens[0..i]) for every position. no sampling involved.
void scoreSequence(const std::vector<int>& tokens, const std::string& id,
                   Weights& weights, cublasHandle_t cublas_handle, Buffers& buf,
                   std::ostream& out);
