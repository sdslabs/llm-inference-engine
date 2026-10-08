#include "config.h"
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

static void usage(const char* prog) {
  std::cerr
    << "Usage: " << prog << " <model.safetensors> <tokenizer.json> <prompts.txt> [options]\n\n"
    << "  prompts.txt              one prompt per line, plain text. blank lines are skipped.\n"
    << "                           \\n \\r \\t \\\\ are unescaped, so a prompt can span lines\n\n"
    << "  --max-new-tokens <n>     default " << MAX_NEW_TOKENS_GENERATED << "\n"
    << "  --top-k <n>              default " << DEFAULT_TOP_K << ", max " << MAX_TOP_K << "\n"
    << "  --temperature <f>        default " << TEMPERATURE << "\n"
    << "  --greedy                 argmax sampling, same as --top-k 1\n"
    << "  --seed <n>               seed the sampler, default nondeterministic\n"
    << "  --score                  emit per token logprobs instead of generating\n"
    << "  --output <file.jsonl>    default stdout\n";
}

static std::string unescape(const std::string& line) {
  std::string out;
  out.reserve(line.size());

  for(size_t i=0; i<line.size(); i++) {
    if(line[i] != '\\' || i+1 >= line.size()) {
      out.push_back(line[i]);
      continue;
    }
    char next = line[++i];
    switch(next) {
      case 'n':  out.push_back('\n'); break;
      case 'r':  out.push_back('\r'); break;
      case 't':  out.push_back('\t'); break;
      case '\\': out.push_back('\\'); break;
      default:   out.push_back('\\'); out.push_back(next); break;
    }
  }
  return out;
}

static int loadPrompts(const fs::path& path, Config& cfg) {
  std::ifstream file(path);
  if(!file.is_open()) {
    std::cerr << "Cannot open prompts file " << path << "\n";
    return 1;
  }

  std::string line;
  while(std::getline(file, line)) {
    if(!line.empty() && line.back() == '\r') line.pop_back();
    if(line.find_first_not_of(" \t") == std::string::npos) continue;
    cfg.prompts.push_back(unescape(line));
  }

  if(file.bad()) {
    std::cerr << "Error reading prompts file " << path << "\n";
    return 1;
  }
  if(cfg.prompts.empty()) {
    std::cerr << "No prompts found in " << path << "\n";
    return 1;
  }
  return 0;
}

ParseResult parseArgs(int argc, char* argv[], Config& cfg) {
  std::vector<std::string> positional;
  bool missing_value = false;

  for(int i=1; i<argc; i++) {
    std::string arg = argv[i];

    auto value = [&](const char* name) -> std::string {
      if(i+1 >= argc) {
        std::cerr << name << " requires a value\n";
        missing_value = true;
        return std::string();
      }
      return argv[++i];
    };

    try {
      if(arg == "--help" || arg == "-h") {
        usage(argv[0]);
        return ParseResult::Exit;
      } else if(arg == "--max-new-tokens") {
        std::string v = value("--max-new-tokens");
        if(!v.empty()) cfg.max_new_tokens = std::stoi(v);
      } else if(arg == "--top-k") {
        std::string v = value("--top-k");
        if(!v.empty()) cfg.top_k = std::stoi(v);
      } else if(arg == "--temperature") {
        std::string v = value("--temperature");
        if(!v.empty()) cfg.temperature = std::stof(v);
      } else if(arg == "--seed") {
        std::string v = value("--seed");
        if(!v.empty()) { cfg.seed = std::stoull(v); cfg.seeded = true; }
      } else if(arg == "--output") {
        cfg.output_path = value("--output");
      } else if(arg == "--greedy") {
        cfg.greedy = true;
      } else if(arg == "--score") {
        cfg.score_mode = true;
      } else if(arg.rfind("--", 0) == 0) {
        std::cerr << "Unknown option " << arg << "\n";
        usage(argv[0]);
        return ParseResult::Error;
      } else {
        positional.push_back(arg);
      }
    } catch(const std::exception& e) {
      std::cerr << "Bad value for " << arg << ": " << e.what() << "\n";
      return ParseResult::Error;
    }

    if(missing_value) return ParseResult::Error;
  }

  if(positional.size() != 3) {
    usage(argv[0]);
    return ParseResult::Error;
  }
  cfg.model_path = positional[0];
  cfg.tokenizer_path = positional[1];
  cfg.prompts_path = positional[2];

  if(loadPrompts(cfg.prompts_path, cfg)) return ParseResult::Error;

  if(cfg.greedy) cfg.top_k = 1;

  if(cfg.max_new_tokens < 1) {
    std::cerr << "--max-new-tokens must be at least 1\n";
    return ParseResult::Error;
  }
  if(cfg.top_k < 1 || cfg.top_k > MAX_TOP_K) {
    std::cerr << "--top-k must be between 1 and " << MAX_TOP_K << "\n";
    return ParseResult::Error;
  }
  if(cfg.temperature <= 0.0f) {
    std::cerr << "--temperature must be positive\n";
    return ParseResult::Error;
  }

  return ParseResult::Ok;
}
