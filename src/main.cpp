#include <tokenizers_cpp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernels.cuh"
#include "config.h"
#include "runtime.h"
#include "model.h"
#include "score.h"
#include "telemetry.h"

struct Request {
  std::string id;
  std::vector<int> tokens;
  std::string text;
};

// Load Hugging Face tokenizer.json
std::unique_ptr<tokenizers::Tokenizer> loadNativeTokenizer(const std::string& tokenizer_json_path) {
  std::ifstream file(tokenizer_json_path);
  if(!file.is_open()) {
    throw std::runtime_error("Failed to open tokenizer file: " + tokenizer_json_path);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();

  return tokenizers::Tokenizer::FromBlobJSON(buffer.str());
}

int main(int argc, char* argv[]) {
  Config cfg;
  ParseResult parsed = parseArgs(argc, argv, cfg);
  if(parsed == ParseResult::Exit) return 0;
  if(parsed == ParseResult::Error) return 1;

  if(!cfg.seeded) cfg.seed = std::random_device{}();
  seedSampler(cfg.seed);
  std::cerr << "seed " << cfg.seed << ", top_k " << cfg.top_k
            << ", temperature " << cfg.temperature
            << ", max_new_tokens " << cfg.max_new_tokens << "\n";

  std::ofstream output_file;
  std::ostream* out = &std::cout;
  if(!cfg.output_path.empty()) {
    output_file.open(cfg.output_path);
    if(!output_file.is_open()) {
      std::cerr << "Cannot open output file " << cfg.output_path << "\n";
      return 1;
    }
    out = &output_file;
  }

  auto tokenizer = loadNativeTokenizer(cfg.tokenizer_path);

  ModelConfig model_cfg;
  if(loadModelConfig(model_cfg, cfg.model_path)) return 1;
  setStopTokens(model_cfg.stop_token_ids);
  setRmsNormEps(model_cfg.rms_norm_eps);

  std::queue<Request> pending;
  for(size_t i=0; i<cfg.prompts.size(); i++) {
    std::string id = std::to_string(i);
    std::vector<int> tokens = tokenizer->Encode(cfg.prompts[i]);
    if(tokens.empty() || tokens[0] != model_cfg.bos_token_id) {
      tokens.insert(tokens.begin(), model_cfg.bos_token_id);
    }

    int len = (int)tokens.size();
    if(len > MAX_PROMPT_LEN) {
      std::cerr << cfg.prompts_path.string() << ": prompt " << id << " encodes to "
                << len << " tokens, exceeds MAX_PROMPT_LEN of " << MAX_PROMPT_LEN << "\n";
      return 1;
    }
    if(cfg.score_mode && len < 2) {
      std::cerr << cfg.prompts_path.string() << ": prompt " << id << " encodes to "
                << len << " tokens, scoring needs at least 2\n";
      return 1;
    }

    pending.push(Request{std::move(id), std::move(tokens), cfg.prompts[i]});
  }

  Weights weights;
  if(loadWeights(weights, cfg.model_path)) return 1;

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  init_rope_frequencies(model_cfg.head_dim, MAX_SEQ_LEN, model_cfg.rope_theta,
                        model_cfg.rope_factor, model_cfg.rope_low_freq_factor,
                        model_cfg.rope_high_freq_factor,
                        model_cfg.rope_original_max_position);

  Buffers buf;
  if(buf.allocate(cfg.score_mode)) return 1;

  std::vector<SlotState> slots(MAX_SEQUENCES);
  int num_requests = (int)pending.size();
  Recorder recorder(*out);

  auto retire = [&](int s, const char* reason) {
    std::string generated_text = tokenizer->Decode(slots[s].generated);
    recorder.onFinish(s, slots[s], reason, generated_text);

    std::cerr << "slot " << s << " done, " << slots[s].generated.size()
              << " tokens, " << reason << "\n";
    std::cerr << slots[s].prompt_text << generated_text << "\n";

    slots[s] = SlotState{};
  };

  while(cfg.score_mode && !pending.empty()) {
    Request request = pending.front();
    pending.pop();
    int num_scored = scoreSequence(request.tokens, request.id, weights, cublas_handle, buf, *out);
    if(num_scored > 0) recorder.onScore(num_scored);
  }

  while(!cfg.score_mode) {
    for(int s=0; s<MAX_SEQUENCES && !pending.empty(); s++) {
      if(slots[s].active) continue;

      Request request = pending.front();
      pending.pop();
      int prompt_len = request.tokens.size();

      prefill(request.tokens, prompt_len, s, cfg, weights, cublas_handle, buf, slots);
      if(!slots[s].active) continue;

      slots[s].id = request.id;
      slots[s].prompt_text = request.text;
      recorder.onPrefill(s, slots[s]);
      std::cerr << "slot " << s << " prefilled " << prompt_len << " tokens\n";

      const char* reason = finishReason(slots[s], cfg.max_new_tokens);
      if(reason) retire(s, reason);
    }

    int num_active = 0;
    for(const SlotState& s : slots) if(s.active) num_active++;
    if(num_active == 0) {
      if(pending.empty()) break;
      continue;
    }

    decode(slots, cfg, weights, cublas_handle, buf);
    recorder.onDecodeStep(slots);

    for(int s=0; s<MAX_SEQUENCES; s++) {
      if(!slots[s].active) continue;
      const char* reason = finishReason(slots[s], cfg.max_new_tokens);
      if(reason) retire(s, reason);
    }
  }

  cudaDeviceSynchronize();
  recorder.writeRun(cfg, num_requests);

  buf.free();
  free_rope_frequencies();
  cublasDestroy(cublas_handle);

  return 0;
}
