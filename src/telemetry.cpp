#include "telemetry.h"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>

using json = nlohmann::json;

Recorder::Recorder(std::ostream& out) : out(out), start(std::chrono::steady_clock::now()) {}

double Recorder::nowMs() const {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
}

void Recorder::onPrefill(int slot, const SlotState& slot_state) {
  cudaDeviceSynchronize();
  double now = nowMs();

  SlotTiming& t = timing[slot];
  t.arrival_ms = 0.0;
  t.ttft_ms = now;
  t.token_times_ms.clear();
  if(!slot_state.generated.empty()) t.token_times_ms.push_back(now);
}

void Recorder::onDecodeStep(const std::vector<SlotState>& slots) {
  cudaDeviceSynchronize();
  double now = nowMs();

  for(int s=0; s<MAX_SEQUENCES; s++) {
    if(slots[s].active) timing[s].token_times_ms.push_back(now);
  }
}

void Recorder::onFinish(int slot, const SlotState& slot_state, const char* reason,
                        const std::string& text) {
  const SlotTiming& t = timing[slot];
  double e2e = t.token_times_ms.empty() ? 0.0 : t.token_times_ms.back();
  int decode_steps = (int)slot_state.generated.size() - 1;

  json record;
  record["type"] = "request";
  record["id"] = slot_state.id;
  record["slot"] = slot;
  record["prompt"] = slot_state.prompt_text;
  record["prompt_tokens"] = slot_state.seq_len - (int)slot_state.generated.size() + 1;
  record["generated_tokens"] = (int)slot_state.generated.size();
  record["tokens"] = slot_state.generated;
  record["text"] = text;
  record["finish_reason"] = reason;
  record["arrival_ms"] = t.arrival_ms;
  record["ttft_ms"] = t.ttft_ms - t.arrival_ms;
  record["e2e_ms"] = e2e - t.arrival_ms;
  record["tpot_ms"] = decode_steps > 0 ? (e2e - t.ttft_ms) / decode_steps : 0.0;
  record["token_times_ms"] = t.token_times_ms;

  out << record.dump() << "\n";
  out.flush();

  completed++;
  generated_tokens += (long long)slot_state.generated.size();
}

void Recorder::writeRun(const Config& cfg, int num_requests) {
  cudaDeviceSynchronize();
  double wall = nowMs();

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);

  int memory_clock_khz = 0, memory_bus_width = 0;
  cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&memory_bus_width, cudaDevAttrGlobalMemoryBusWidth, 0);

  json device;
  device["name"] = prop.name;
  device["compute_capability"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
  device["sm_count"] = prop.multiProcessorCount;
  device["memory_clock_khz"] = memory_clock_khz;
  device["memory_bus_width_bits"] = memory_bus_width;
  device["peak_bandwidth_bytes_per_sec"] =
      (uint64_t)2 * memory_clock_khz * 1000 * (memory_bus_width / 8);
  device["total_memory_bytes"] = (uint64_t)total_mem;

  json record;
  record["type"] = "run";
  record["mode"] = cfg.score_mode ? "score" : "generate";
  record["model"] = cfg.model_path.string();
  record["device"] = device;
  record["seed"] = cfg.seed;
  record["greedy"] = cfg.greedy;
  record["top_k"] = cfg.top_k;
  record["temperature"] = cfg.temperature;
  record["max_new_tokens"] = cfg.max_new_tokens;
  record["max_sequences"] = MAX_SEQUENCES;
  record["num_requests"] = num_requests;
  record["completed_requests"] = completed;
  record["generated_tokens"] = generated_tokens;
  record["wall_ms"] = wall;
  record["kv_bytes_per_token"] = (uint64_t)N_LAYERS * 2 * KV_DIM * sizeof(bf16);
  record["gpu_used_bytes"] = (uint64_t)(total_mem - free_mem);

  out << record.dump() << "\n";
  out.flush();
}
