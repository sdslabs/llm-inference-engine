//change the build system files
#include <cassert>
#include <numeric>
#include <algorithm>
#include "block_manager.h"

FreeStack::FreeStack(int sz) : free_stack(sz), top(free_stack.end() - 1) {
    assert(sz > 0);
    std::iota(free_stack.begin(), free_stack.end(), 0);
}

[[nodiscard]] PopResult FreeStack::multipop(int amt, int* dest) {
    assert(amt > 0);
    if(top - free_stack.begin() >= amt - 1) {
        std::copy(top - amt + 1, top + 1, dest);
        top -= amt;
        return PopResult::success;
    }
    else {
        return PopResult::InsufficientBlocks;
    }
}

void FreeStack::multipush(int amt, int* src) {
    assert(amt > 0);
    assert(top + amt < free_stack.end());
    std::copy(src, src + amt, top + 1);
    top += amt;
}

