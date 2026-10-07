# eval

Test harness for the engine. Each script measures one thing and prints it.

Run everything from the repo root. `engine.py` is a shared client, not a script.

## Setup

```bash
git submodule update --init --recursive
cd external/tokenizers-cpp && cmake -B build . && cmake --build build -j   # needs a Rust toolchain
cd ../.. && ./build.sh                                                     # produces build/engine

python/venv/bin/pip install datasets         # for perplexity --dataset and tasks
python/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

python/venv/bin/hf download meta-llama/Llama-3.2-1B \
  --local-dir python/models/llama-3.2-1b     # gated, needs `hf auth login`
```

`build.sh` does not build the submodule, it only links against
`external/tokenizers-cpp/build/libtokenizers_{c,cpp}.a`.

`python/models/llama-3.2-1b/` must hold `model.safetensors`, `config.json` and
`tokenizer.json`. The engine reads the safetensors directly, the reference loads
the same directory through transformers, so both sides always use the same
weights. Override with `MODEL_DIR=<path>` if needed.

---

## bench.py — speed

Latency and throughput across batch size, prompt length and output length.
Needs no reference model.

Reports TTFT / inter-token latency / end-to-end at p50, p90, p99, tokens per
second, and **MBU** (memory bandwidth utilisation) against a roofline computed
from the GPU's actual clock and bus width. Decode is memory bound, so MBU is the
number to optimise.

```bash
python/venv/bin/python eval/bench.py
python/venv/bin/python eval/bench.py --batch 1,4 --prompt-len 128 --new-tokens 64 --reps 5
python/venv/bin/python eval/bench.py --json results.json
```

---

## test_logits.py — prefill numerics

Scores the same sequences through the engine and the fp32 reference, compares
log P at every position. Catches numeric drift anywhere in the forward pass.

Fails on max abs diff > 0.25, mean abs diff > 0.06, or Pearson r < 0.9995.
Thresholds sit about 2x above the observed bf16 noise floor.

```bash
python/venv/bin/python eval/test_logits.py
python/venv/bin/python eval/test_logits.py --tolerance 0.3 --json logits.json
```

---

## test_generation.py — decode path

Greedy decode vs reference greedy decode. The only test that exercises decode
(KV cache writes, `ropeDecode` positions, `decodeSoftmax`); `test_logits.py`
covers prefill only.

Reports **mean first divergence** rather than exact match. Divergence is
expected as bf16 rounding compounds; what matters is where. Divergence at token
3 is a bug, at token 19 it is normal. Fails below 0.6 x `--max-new-tokens`.

```bash
python/venv/bin/python eval/test_generation.py
python/venv/bin/python eval/test_generation.py --max-new-tokens 40 --json gen.json
```

---

## perplexity.py — whole-system correctness

Windows a corpus, scores every window, reports `exp(-mean logprob)`.

The absolute number reflects the model and the windowing, not the engine. The
**delta against the reference** is the signal: near zero means the engine
reproduces the reference faithfully.

`--stride` below `--window` gives each prediction more context and lowers
perplexity; only positions no earlier window already predicted are counted.

```bash
python/venv/bin/python eval/perplexity.py                                # WikiText-2 test
python/venv/bin/python eval/perplexity.py --limit 30 --compare           # vs reference (slow)
python/venv/bin/python eval/perplexity.py --window 512 --stride 256
python/venv/bin/python eval/perplexity.py --text corpus.txt              # local file
```

---

## tasks.py — task accuracy

Multiple choice by loglikelihood scoring: every candidate continuation is
scored, highest wins. No generation involved.

`acc` uses total continuation logprob, `acc_norm` uses logprob per token, which
corrects the bias against longer answers. Compare `acc_norm` against published
Llama-3.2-1B numbers.

```bash
python/venv/bin/python eval/tasks.py --task hellaswag --limit 300
python/venv/bin/python eval/tasks.py --task arc_easy --limit 300
python/venv/bin/python eval/tasks.py --task arc_challenge --limit 300 --json arc.json
```

---

## Current numbers

RTX 4050 Laptop (6 GB, 192 GB/s), Llama-3.2-1B bf16.

| Measurement | Value |
|---|---|
| Perplexity delta vs reference | +0.10% (17.9934 vs 17.9761) |
| Logprob correlation | r >= 0.99992, 8/8 sequences pass |
| Greedy exact match, `--max-new-tokens 20` | 6/8, mean first divergence 17.9 |
| WikiText-2 perplexity | 17.92 (512-token non-overlapping windows) |
| HellaSwag acc_norm | 44.67% (300 questions) |
| ARC-Easy acc | 65.67% (300 questions) |
| Throughput batch 1 / batch 4 | 53.8 / 186.2 tok/s |
| MBU batch 1 / batch 4 | 86% / 80% |

## Limits

- `MAX_PROMPT_LEN` 512, `MAX_SEQUENCES` 4, `MAX_SEQ_LEN` 1024, all compile time.
  Set them in `.env` and rebuild; `src/kernels.cuh` only holds `#ifndef`
  fallbacks that `build.sh` overrides. The scripts read the same `.env` and
  error early, so editing it without rebuilding makes the two sides disagree.
- `MAX_SEQ_LEN` cannot exceed `MAX_NUM_THREAD`, because the softmax reductions
  use one block per row.
- The engine tokenizes prompts itself now, so the harness sends text, not token
  ids. Prompts go one per line with `\n` escaped; `engine.escape_line` does the
  packing. Each suite cross checks the engine's token count against its own and
  fails on a mismatch, which is what catches tokenizer drift.
- `tasks.py` takes the continuation boundary from the tokenization of
  `context + choice`, not from `len(encode(context))`, since BPE can merge
  across the join.
- `perplexity.py` drops any window whose text does not re-tokenize to the ids it
  came from, and reports how many. Zero on WikiText-2 in practice.
- `arrival_ms` is always 0. Correct for offline batch; timed arrivals would need
  a load generator.
- `test_logits.py` compares log P at the observed token, not the full 128k
  distribution. True KL needs a `--dump-logits` flag the engine does not have.
- `topKSample` with k > 1 has no statistical test. Only greedy is verified.
