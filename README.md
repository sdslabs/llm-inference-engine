# Nucleus

An LLM inference engine for the Llama 3.2 architecture, written from scratch in
C++/CUDA. cuBLAS does the GEMMs; everything else is hand-written kernels. No
Python in the serving path.

Reference hardware: RTX 4050 Laptop (6 GB, 192 GB/s). 213 tok/s at batch 4,
88% memory bandwidth utilisation at batch 1, and within 0.05% perplexity of the
fp32 HuggingFace reference. Full numbers in [eval/README.md](eval/README.md).

## Build and run

```bash
git submodule update --init --recursive
cd external/tokenizers-cpp && cmake -B build . && cmake --build build -j   # needs a Rust toolchain
cd ../..

cc -o nob nob.c      # once, nob rebuilds itself after this
./nob                # -> build/engine     (./nob clean removes build/)

./build/engine python/models/llama-3.2-1b/model.safetensors \
               python/models/llama-3.2-1b/tokenizer.json \
               prompts.txt --greedy --max-new-tokens 64
```

`prompts.txt` is one prompt per line, plain text, with `\n` `\r` `\t` `\\`
unescaped so a prompt can span lines. Records go to stdout as JSONL, or to
`--output <file>`; human logging goes to stderr. `--score` emits per-token
logprobs instead of generating. `--help` lists the rest.

## Layout

| Path | |
|---|---|
| `src/kernels.cu`, `.cuh` | every custom kernel, and all compile-time limits |
| `src/runtime.*` | safetensors loading, `config.json` parsing, scratch allocation |
| `src/model.*` | `forwardPrefill`, `prefill`, `decode`, slot state |
| `src/config.*` | CLI parsing, prompt file loading |
| `src/score.cpp` | `--score` mode, per-token logprobs |
| `src/telemetry.*` | TTFT / TPOT / ITL timing, JSONL records |
| `src/main.cpp` | tokenizer, request queue, slot admission loop |
| `eval/` | correctness and speed harness, see its README |
| `nob.c`, `nob.h` | build system ([nob](https://github.com/tsoding/nob.h)) |
| `.env` | sole definition of the compile-time limits and defaults |

## How it works

**Weights** are read straight out of `model.safetensors` by HuggingFace tensor
name, bf16, no conversion step. `embed_tokens` is reused as the output
projection, since Llama 3.2 1B ties them.

**Prefill** runs one request at a time over the whole prompt. Per layer:
RMSNorm, Q/K/V projections, RoPE, K/V written into the request's KV cache slot,
attention, then the SwiGLU MLP. Grouped-query attention is exploited by batching
the 32 per-head GEMMs into 8 strided-batched cuBLAS calls, one per K/V group,
with the shared K broadcast at stride 0.

**Decode** packs every active sequence into dense batch rows and runs one step
for all of them, reading each sequence's history from its own cache slot. Only
the attention GEMMs loop per row, because the cache pointers differ.

**Continuous batching** is a fixed array of `MAX_SEQUENCES` slots, each owning a
contiguous KV cache lane. Each iteration fills idle slots from the queue, runs
one decode step, then retires finished sequences. A slot freed this iteration is
refilled on the next, so arrivals join an in-flight batch without draining it.

**Sampling** is top-k with temperature, done on the GPU over the full 128k
vocabulary with deterministic tie-breaking. `--greedy` is `--top-k 1`.

## Configuration

Three separate sources, by lifetime:

- **`.env`** — build target and compile-time limits (`MAX_SEQUENCES`,
  `MAX_PROMPT_LEN`, `MAX_SEQ_LEN`, `MAX_NUM_THREAD`, `MAX_TOP_K`) plus the
  sampling defaults. These size `__shared__` arrays and fixed arrays, so they
  cannot be runtime values. `nob` passes them as `-D` defines and fails if one
  is unset; `src/kernels.cuh` `#error`s rather than carry a fallback, and
  `static_assert`s the relationships between them. Shell exports override the
  file: `MAX_SEQUENCES=8 ./nob`.
- **`config.json`** next to the model — rope parameters, `rms_norm_eps`, BOS/EOS
  ids, read at startup. The architecture fields are validated against the
  compiled constants, so a mismatched checkpoint is a startup error rather than
  silent garbage. This is the vendor file from the checkpoint; it is never
  written.
- **CLI flags** — `--top-k`, `--temperature`, `--max-new-tokens`, `--seed`,
  override the defaults per run.

## Known limits

- Prefill is one request at a time, so TTFT for the last slot admitted includes
  every earlier prefill. Chunked prefill is the main structural work left.
- Attention is unfused: prefill materialises the full
  `NUM_Q_HEADS x prompt_len^2` score matrix and makes separate masking and
  softmax passes over it.
- The KV cache is a flat slab with one fixed-size lane per slot, so a short
  sequence reserves as much as a long one. No paging or prefix sharing.
- Architecture constants are compile-time, so only Llama 3.2 1B works without a
  rebuild. Validated, not adapted.
- CUDA errors are logged but do not affect the exit code.
- `MAX_SEQ_LEN` is capped at `MAX_NUM_THREAD` (1024) because the softmax
  reductions use one block per row.
