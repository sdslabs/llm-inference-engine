#pragma once
#include <chrono>
#include <ostream>
#include <string>
#include <vector>
#include "config.h"
#include "model.h"

struct SlotTiming {
  double arrival_ms = 0.0;
  double ttft_ms = 0.0;
  std::vector<double> token_times_ms;
};

// timestamps every request and emits one JSONL record per completion.
// the pipeline never reads any of this back, it only calls the hooks.
struct Recorder {
  explicit Recorder(std::ostream& out);

  void onPrefill(int slot, const SlotState& slot_state);
  void onDecodeStep(const std::vector<SlotState>& slots);
  void onFinish(int slot, const SlotState& slot_state, const char* reason,
                const std::string& text);
  void onScore(int num_scored);
  void writeRun(const Config& cfg, int num_requests);

private:
  double nowMs() const;

  std::ostream& out;
  std::chrono::steady_clock::time_point start;
  SlotTiming timing[MAX_SEQUENCES];
  int completed = 0;
  long long generated_tokens = 0;
  long long scored_tokens = 0;
};
