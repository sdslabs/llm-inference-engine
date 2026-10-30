//change the build system files
#include <cassert>
#include <numeric>
#include <algorithm>
#include "block_manager.h"

FreeStack::FreeStack(int sz) : free_stack_(sz), top_(free_stack_.end() - 1) {
    assert(sz > 0);
    std::iota(free_stack_.begin(), free_stack_.end(), 0);
}

[[nodiscard]] AllocResult FreeStack::multipop(int amt, int* dest) {
    assert(amt > 0);
    
    if(top_ - free_stack_.begin() >= amt - 1) {
        std::copy(top_ - amt + 1, top_ + 1, dest);
        top_ -= amt;
        return AllocResult::Success;
    }
    else {
        return AllocResult::InsufficientBlocks;
    }
}

void FreeStack::multipush(int amt, int* src) {
    assert(amt > 0);
    assert(top_ + amt < free_stack_.end());

    std::copy(src, src + amt, top_ + 1);
    top_ += amt;
}

BlockManager::BlockManager(int max_sequences, int max_blocks_per_seq)
    : free_blocks_(max_sequences * max_blocks_per_seq),
      max_sequences_(max_sequences),
      max_blocks_per_seq_(max_blocks_per_seq),
      using_count_(max_sequences, 0),
      block_table_(max_sequences * max_blocks_per_seq, 0) {}

[[nodiscard]] AllocResult BlockManager::allocate(int slot, int amt) { // amt is the number of blocks to allocate
    assert(slot >= 0 && slot < max_sequences_);
    assert(amt > 0 && using_count_[slot] + amt <= max_blocks_per_seq_);

    int* block_table_entry = block_table_.data() + slot * max_blocks_per_seq_ + using_count_[slot];
    AllocResult res = free_blocks_.multipop(amt, block_table_entry);
    if(res == AllocResult::Success) {
        using_count_[slot] += amt;
    }

    return res;
}

void BlockManager::deallocate(int slot) {
    assert(slot >= 0 && slot < max_sequences_);
    assert(using_count_[slot] > 0 && "double-free, or slot was never allocated");

    int* block_table_slot = block_table_.data() + slot * max_blocks_per_seq_;
    free_blocks_.multipush(using_count_[slot], block_table_slot);
    using_count_[slot] = 0;
}