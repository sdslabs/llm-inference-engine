#pragma once
#include <vector>
#include "kernels.cuh"

enum class PopResult : int {
  success,
  InsufficientBlocks,
};

class FreeStack {
  private:
  std::vector<int> free_stack; 
  std::vector<int>::iterator top; // this iterator points to the top element in the stack
  public:
  FreeStack(int sz); // sz is the total number of blocks carved out of the kv cache
  [[nodiscard]] PopResult multipop(int amt, int* dest);
  void multipush(int amt, int* src);
};
