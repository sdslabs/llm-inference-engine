#pragma once
#include <vector>

enum class AllocResult : int {
  Success,
  InsufficientBlocks,
};

class FreeStack {
  private:
  std::vector<int> free_stack_; 
  std::vector<int>::iterator top_; // this iterator points to the top element in the stack
  
  public:
  FreeStack(int sz); // sz is the total number of blocks carved out of the kv cache
  [[nodiscard]] AllocResult multipop(int amt, int* dest);
  void multipush(int amt, int* src);
};

class BlockManager {
  private:
  FreeStack free_blocks_;
  int max_sequences_;
  int max_blocks_per_seq_;
  std::vector<int> using_count_; //book-keeping data structure which stores how many blocks have been allocated for a certain slot
  std::vector<int> block_table_;

  public:
  BlockManager(int max_sequences, int max_blocks_per_seq);
  [[nodiscard]] AllocResult allocate(int slot, int amt);
  void deallocate(int slot);
};
