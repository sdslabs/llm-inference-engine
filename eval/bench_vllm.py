#!/usr/bin/env python3
"""vLLM benchmark mirroring eval/bench.py methodology.

Same sweep grid (batch x prompt_len x new_tokens), same prompt generator, same
metrics: TTFT / ITL / E2E at p50, p90, p99, TPOT mean, throughput tok/s.

A batch of N is N requests sent concurrently, so vLLM's scheduler batches them
the way the engine batches its prompt file. Greedy, ignore_eos so every request
generates exactly new_tokens. Uses /v1/completions with raw prompts (base model,
no chat template), matching bench.py.

Per-request timing comes from streaming chunks: first chunk => TTFT, gaps
between chunks => ITL (matches the engine's token_times_ms).

Start the server first, e.g.:
  vllm serve python/models/llama-3.2-1b --served-model-name llama-3.2-1b \
    --dtype bfloat16 --max-model-len 1024 --max-num-seqs 4
"""
import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "python" / "models" / "llama-3.2-1b"

WORDS = (
    "time year people way day man thing woman life child world school state family "
    "student group country problem hand part place case week company system program "
    "question work government number night point home water room mother area money "
    "story fact month lot right study book eye job word business issue side kind head "
    "house service friend father power hour game line end member law car city community "
    "name president team minute idea body information parent face level office door "
    "health person art war history party result change morning reason research girl "
    "moment air teacher force education season player market report street while"
).split()

tok = None


def text_of_token_length(target, rng):
    """Real text encoding to exactly `target` tokens, BOS included (matches bench.py)."""
    if target < 2:
        return None
    words = [rng.choice(WORDS) for _ in range(target)]
    for _ in range(60):
        text = " ".join(words)
        length = len(tok(text)["input_ids"])
        if length == target:
            return text
        if length < target:
            words.extend(rng.choice(WORDS) for _ in range(target - length))
        else:
            trim = min(length - target, len(words) - 1)
            words = words[: len(words) - trim]
    return None


def percentiles(values, points=(50, 90, 99)):
    if not values:
        return {f"p{p}": 0.0 for p in points}
    arr = np.array(values, dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in points}


def run_one_streaming(url, model, text, max_tokens, t0):
    """Stream one completion. Times are ms since t0, the batch start."""
    times = []
    prompt_tokens = completion_tokens = None
    with requests.post(url, json={
        "model": model, "prompt": text, "max_tokens": max_tokens,
        "temperature": 0.0, "seed": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: "):]
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"]["prompt_tokens"]
                completion_tokens = chunk["usage"]["completion_tokens"]
            if chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                times.append((time.perf_counter() - t0) * 1000)
    return {"times": times, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens if completion_tokens is not None else len(times)}


def run_batch(url, model, prompts, new_tokens):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        results = list(pool.map(
            lambda p: run_one_streaming(url, model, p, new_tokens, t0), prompts))
    wall_ms = (time.perf_counter() - t0) * 1000
    return results, wall_ms


def run_config(url, model, batch, prompt_len, new_tokens, reps, rng):
    ttfts, itls, e2es, tpots = [], [], [], []
    wall_ms = []
    prompt_tokens = []
    generated_total = 0

    for _ in range(reps):
        prompts = []
        for _ in range(batch):
            text = text_of_token_length(prompt_len, rng)
            if text is None:
                sys.exit(f"could not build a prompt of exactly {prompt_len} tokens")
            prompts.append(text)

        results, wall = run_batch(url, model, prompts, new_tokens)
        wall_ms.append(wall)
        for res in results:
            times = res["times"]
            if res["prompt_tokens"] is not None and res["prompt_tokens"] != prompt_len:
                sys.exit(f"server counted {res['prompt_tokens']} prompt tokens, expected {prompt_len}")
            prompt_tokens.append(res["prompt_tokens"] or prompt_len)
            generated_total += res["completion_tokens"]
            if not times:
                continue
            ttfts.append(times[0])
            e2es.append(times[-1])
            gaps = [b - a for a, b in zip(times, times[1:])]
            itls.extend(gaps)
            if gaps:
                tpots.append((times[-1] - times[0]) / len(gaps))

    total_wall_s = sum(wall_ms) / 1000.0
    return {
        "batch": batch,
        "prompt_len": prompt_len,
        "mean_prompt_tokens": statistics.mean(prompt_tokens),
        "new_tokens": new_tokens,
        "reps": reps,
        "ttft_ms": percentiles(ttfts),
        "itl_ms": percentiles(itls),
        "e2e_ms": percentiles(e2es),
        "tpot_ms_mean": statistics.mean(tpots) if tpots else 0.0,
        "throughput_tok_s": generated_total / total_wall_s if total_wall_s > 0 else 0.0,
        "engine": "vllm",
    }


def main():
    global tok
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", default="llama-3.2-1b", help="served model name")
    parser.add_argument("--batch", default="1,2,4")
    parser.add_argument("--prompt-len", default="32,128,512")
    parser.add_argument("--new-tokens", default="64")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="write full results here")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    batches = [int(x) for x in args.batch.split(",")]
    prompt_lens = [int(x) for x in args.prompt_len.split(",")]
    new_tokens = [int(x) for x in args.new_tokens.split(",")]

    rng = random.Random(args.seed)

    # warmup: first requests pay for CUDA graph capture / allocator growth
    warm = text_of_token_length(max(prompt_lens), random.Random(1))
    run_batch(args.url, args.model, [warm] * max(batches), 16)

    results = []
    header = (f"{'batch':>5} {'plen':>5} {'gen':>5} "
              f"{'TTFT p50':>9} {'p99':>8} {'ITL p50':>8} {'p99':>8} "
              f"{'TPOT':>7} {'E2E p50':>8} {'tok/s':>8}")
    print(header)
    print("-" * len(header))

    for batch in batches:
        for prompt_len in prompt_lens:
            for gen in new_tokens:
                r = run_config(args.url, args.model, batch, prompt_len, gen, args.reps, rng)
                results.append(r)
                print(f"{batch:>5} {prompt_len:>5} {gen:>5} "
                      f"{r['ttft_ms']['p50']:>9.1f} {r['ttft_ms']['p99']:>8.1f} "
                      f"{r['itl_ms']['p50']:>8.2f} {r['itl_ms']['p99']:>8.2f} "
                      f"{r['tpot_ms_mean']:>7.2f} {r['e2e_ms']['p50']:>8.1f} "
                      f"{r['throughput_tok_s']:>8.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
