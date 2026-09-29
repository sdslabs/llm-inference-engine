#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include "kernels.cuh"

struct Config {
  std::filesystem::path model_path;
  std::filesystem::path tokenizer_path;
  std::string prompt_text;

  std::vector<std::vector<int>> prompts;
  std::vector<std::string> prompt_ids;

  int max_new_tokens = MAX_NEW_TOKENS_GENERATED;
  int top_k = TOP_K;
  float temperature = TEMPERATURE;
  bool greedy = false;

  unsigned long long seed = 0;
  bool seeded = false;

  bool score_mode = false;
  std::filesystem::path output_path;
};

enum class ParseResult { Ok, Error, Exit };

ParseResult parseArgs(int argc, char* argv[], Config& cfg);
