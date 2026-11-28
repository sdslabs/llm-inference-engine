#include <chrono>
#include <condition_variable>
#include <mutex>
#include <ostream>
#include <queue>
#include <tokenizers_cpp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include "kernels.cuh"
#include "config.h"
#include "runtime.h"
#include "model.h"
#include "score.h"
#include "telemetry.h"
#include "queue.h"

#include "../external/httplib.h"
#include "../external/json.hpp"

using json = nlohmann::json;

// Global thread-safe queue & engine state flag
RequestQueue g_request_queue;
std::atomic<bool> g_engine_running{true};

// Helper struct for bridging GPU engine callbacks to HTTP response threads
struct ResponseToken {
  std::string text;
  bool is_finished;
};

class TokenQueue {
private:
  std::queue<ResponseToken> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;

public:
  void push(ResponseToken t) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(std::move(t));
    }
    cv_.notify_one();
  }

  bool pop(ResponseToken& t) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]{ return !queue_.empty(); });
    t = std::move(queue_.front());
    queue_.pop();
    return true;
  }
};

// Message format for Llama 3.2 chat header 
std::string apply_chat_template(const json& messages) {
  std::string prompt = "";
  for(const auto& msg: messages) {
    std::string role = msg["role"];
    std::string content = msg["content"];
    prompt += "<|start_header_id|>" + role + "<|end_header_id|>\n\n" + content + "<|eot_id|>";
  }
  prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return prompt;
}

void run_engine_loop (
  const Config& cfg,
  const ModelConfig& model_cfg,
  Weights& weights,
  cublasHandle_t cublas_handle,
  Buffers& buf,
  tokenizers::Tokenizer* tokenizer,
  Recorder& recorder
) {
  std::vector<SlotState> slots(MAX_SEQUENCES);

  auto retire_slot = [&](int s, const char* reason) {
    std::string final_text = tokenizer->Decode(slots[s].generated);

    recorder.onFinish(s, slots[s], reason, final_text);

    // signals client thread that generation is done for this req
    if (slots[s].on_token_callback) {
      slots[s].on_token_callback("", true);
    }

    std::cerr << "[Engine] Slot " << s << " finished (" << reason << ")\n";
    slots[s] = SlotState{};
  };


  while(g_engine_running) {
    // Ingesting incoming requests
    for(int s{0}; s < MAX_SEQUENCES; s++) {
      if(slots[s].active) continue;

      ClientRequest req;
      // Non-blocking check for new requests
      if(!g_request_queue.pop(req, false)) break;

      int prompt_len = req.tokens.size();

      prefill(req.tokens, prompt_len, s, cfg, weights, cublas_handle, buf, slots);
      
      if(!slots[s].active) {
        if(req.on_token) {
          req.on_token("\n[Error: Request dropped during prefill (likely exceeds max context length)]", true);
        }
        std::cerr << "[Engine] Request " << req.id << " dropped during prefill.\n";
        continue;
      }

      slots[s].id = req.id;
      slots[s].prompt_text = req.raw_text;
      slots[s].on_token_callback = req.on_token; // Store callback

      // transfer time stamp from queue to the active slot
      slots[s].arrival_time = req.arrival_time;

      recorder.onPrefill(s, slots[s]);
      
      std::cerr << "[Engine] Slot " << s << " prefilled " << prompt_len << " tokens\n";
      
      const char* reason = finishReason(slots[s], req.max_tokens);

      if(!slots[s].generated.empty() && slots[s].on_token_callback) {
        bool is_stop = (reason != nullptr && std::string(reason) == "stop");
        if(!is_stop) {
          int first_token_id = slots[s].generated.back();
          std::string token_str = tokenizer->Decode({first_token_id});
          slots[s].on_token_callback(token_str, false);
        }
      }

      if(reason) retire_slot(s, reason);
    }

    int num_active = 0;
    for(const SlotState& s: slots) {
      if(s.active) num_active++;
    }

    if(num_active == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    decode(slots, cfg, weights, cublas_handle, buf);

    recorder.onDecodeStep(slots);

    for(int s{0}; s < MAX_SEQUENCES; s++) {
      if(!slots[s].active) continue;

      const char* reason = finishReason(slots[s], cfg.max_new_tokens);

      // Stream last token to the request callback
      if(!slots[s].generated.empty() && slots[s].on_token_callback) {
        bool is_stop = (reason != nullptr && std::string(reason) == "stop");
        if(!is_stop) {
          int last_token_id = slots[s].generated.back();
          std::string token_str = tokenizer->Decode({last_token_id});
          slots[s].on_token_callback(token_str, false);
        }
      }

      if(reason) retire_slot(s, reason);
    }
  }
}

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
    
  if(cfg.score_mode) {
    std::cerr << "Running in Scoring Mode\n";
    Recorder recorder(*out);
      
    int prompt_idx = 0;
    for(const auto& prompt_text: cfg.prompts) {
      std::string id = std::to_string(prompt_idx++);

      std::vector<int> tokens = tokenizer->Encode(prompt_text);
      if(tokens.empty() || tokens[0] != model_cfg.bos_token_id) {
        tokens.insert(tokens.begin(), model_cfg.bos_token_id);
      }
        
      int len = (int)tokens.size();
      if(len > MAX_PROMPT_LEN) {
        std::cerr << cfg.prompts_path->string() << ": prompt " << id << " encodes to "
                  << len << " tokens, exceeds MAX_PROMPT_LEN of " << MAX_PROMPT_LEN << "\n";
        return 1;
      }
      if(cfg.score_mode && len < 2) {
        std::cerr << cfg.prompts_path->string() << ": prompt " << id << " encodes to "
                  << len << " tokens, scoring needs at least 2\n";
        return 1;
      }


      int num_scored = scoreSequence(tokens, id, weights, cublas_handle, buf, *out);
      if(num_scored > 0) recorder.onScore(num_scored);
    }

    cudaDeviceSynchronize();
    buf.free();
    free_rope_frequencies();
    cublasDestroy(cublas_handle);
    return 0;
  }

  // recorder for the interactive loop
  Recorder recorder(cfg.server_mode ? std::cerr : *out);

  // Spawn prefill/decode loop thread
  std::thread engine_thread(run_engine_loop, std::ref(cfg), std::ref(model_cfg), std::ref(weights), cublas_handle, std::ref(buf), tokenizer.get(), std::ref(recorder));
  
  if(cfg.server_mode) {
    // Server Mode
    httplib::Server server;
    std::atomic<int> req_counter{0};

    server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
      json request_body = json::parse(req.body);
      std::string raw_prompt = apply_chat_template(request_body["messages"]);

      auto token_queue = std::make_shared<TokenQueue>();

      ClientRequest client_req;
      client_req.id = "req_" + std::to_string(req_counter++);
      client_req.raw_text = raw_prompt;
      client_req.tokens = tokenizer->Encode(raw_prompt);

      if(client_req.tokens.empty() || client_req.tokens[0] != model_cfg.bos_token_id) {
        client_req.tokens.insert(client_req.tokens.begin(), model_cfg.bos_token_id);
      }

      client_req.max_tokens = request_body.value("max_tokens", cfg.max_new_tokens);

      client_req.on_token = [token_queue](const std::string& token_text, bool is_finished) {
        token_queue->push({token_text, is_finished});
      };

      client_req.arrival_time = std::chrono::steady_clock::now();

      g_request_queue.push(std::move(client_req));

      res.set_chunked_content_provider(
        "text/event-stream",
        [token_queue](size_t offset, httplib::DataSink& sink){
          ResponseToken t;
          if(token_queue->pop(t)) {
            if(!t.is_finished) {
              json chunk = {{"choices", {{{"delta", {{"content", t.text}}}}}}};
              std::string sse_payload = "data: " + chunk.dump() + "\n\n";
              sink.write(sse_payload.c_str(), sse_payload.size());
              return true;
            } else {
              sink.write("data: [DONE]\n\n", 14);
              sink.done();
              return false;
            }
          }
          return true;
        }
      );
    });

    std::cerr << "Server listening on http://0.0.0.0:8080\n";
    server.listen("0.0.0.0", 8080);
  } else if(!cfg.prompts.empty()) {
    std::cerr << "Engine in batch mode. Processing " << cfg.prompts.size() << " prompts\n";

    std::atomic<int> completed_requests{0};
    int total_requests = static_cast<int>(cfg.prompts.size());

    for(size_t i{0}; i<cfg.prompts.size(); i++) {
      ClientRequest req;
      req.id = std::to_string(i);
      req.raw_text = cfg.prompts[i];
      req.tokens = tokenizer->Encode(req.raw_text);

      if(req.tokens.empty() || req.tokens[0] != model_cfg.bos_token_id) {
        req.tokens.insert(req.tokens.begin(), model_cfg.bos_token_id);
      }

      req.max_tokens = cfg.max_new_tokens;
      req.on_token = [&completed_requests, out](const std::string& token_text, bool is_finished) {
        if(!is_finished) {
          *out << token_text << std::flush;
        } else {
          *out << "\n" << std::flush;
          completed_requests++;
        }
      };

      req.arrival_time = std::chrono::steady_clock::now();

      g_request_queue.push(std::move(req));
    }

    while(completed_requests < total_requests) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    recorder.writeRun(cfg, total_requests);
  } else {

    std::cerr << "Engine started in terminal mode.\n Type a prompt and press Enter (or 'exit' to quit):\n";
    
    // Request handler (terminal stdin for now)
    std::string input_line;
    int req_id = 0;
    while (std::getline(std::cin, input_line)) {
      if(input_line == "exit" || input_line == "quit") break;
      if(input_line.empty()) continue;
      
      ClientRequest req;
      req.id = std::to_string(req_id++);
      req.raw_text = input_line;
      req.tokens = tokenizer->Encode(input_line);
      
      if(req.tokens.empty() || req.tokens[0] != model_cfg.bos_token_id) {
        req.tokens.insert(req.tokens.begin(), model_cfg.bos_token_id);
      }
      
      req.max_tokens = cfg.max_new_tokens;
      
      // what happens when a new token is generated for this request is below
      req.on_token = [](const std::string& token_text, bool is_finished) {
        if(!is_finished) {
          std::cout << token_text << std::flush;
        } else {
          std::cout << "\n> " << std::flush;
        }
      };
      
      req.arrival_time = std::chrono::steady_clock::now();

      g_request_queue.push(std::move(req));
    }
    
  }


  std::cerr << "\nShutting down engine\n";
  g_engine_running = false;

  // Pushing an empty dummy request to wake up the queue if it is sleeping
  g_request_queue.push(ClientRequest{});
  engine_thread.join();

  cudaDeviceSynchronize();

  buf.free();
  free_rope_frequencies();
  cublasDestroy(cublas_handle);

  return 0;
}
