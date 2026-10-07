#!/usr/bin/env python3
"""Latency and throughput benchmark.

Sweeps batch size, prompt length and output length, then reports percentile
latencies and memory bandwidth utilisation. Needs no reference model.
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine


def percentiles(values, points=(50, 90, 99)):
    if not values:
        return {f"p{p}": 0.0 for p in points}
    array = np.array(values, dtype=float)
    return {f"p{p}": float(np.percentile(array, p)) for p in points}


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


def text_of_token_length(target, rng):
    """Text that encodes to exactly `target` tokens, BOS included.

    The engine tokenizes prompts itself now, so a benchmark prompt has to be
    real text. Content is still irrelevant to timing, only the token count is.
    """
    if target < 2:
        return None

    words = [rng.choice(WORDS) for _ in range(target)]
    for _ in range(60):
        text = " ".join(words)
        length = len(engine.encode(text))
        if length == target:
            return text
        if length < target:
            words.extend(rng.choice(WORDS) for _ in range(target - length))
        else:
            trim = min(length - target, len(words) - 1)
            words = words[:len(words) - trim]
    return None


def run_config(batch, prompt_len, new_tokens, reps, rng):
    ttfts, tpots, e2es, itls = [], [], [], []
    wall_ms = []
    prompt_tokens = []
    generated_total = 0
    device = None
    weights_bytes = engine.MODEL.stat().st_size
    kv_bytes_per_token = None

    for _ in range(reps):
        prompts = []
        for i in range(batch):
            text = text_of_token_length(prompt_len, rng)
            if text is None:
                sys.exit(f"could not build a prompt of exactly {prompt_len} tokens")
            prompts.append((f"r{i}", text))

        records = engine.run(prompts, greedy=True, max_new_tokens=new_tokens, seed=0)

        requests = engine.of_type(records, "request")
        run = engine.run_record(records)
        device = run["device"]
        kv_bytes_per_token = run["kv_bytes_per_token"]
        wall_ms.append(run["wall_ms"])
        generated_total += run["generated_tokens"]

        for request in requests:
            prompt_tokens.append(request["prompt_tokens"])
            ttfts.append(request["ttft_ms"])
            e2es.append(request["e2e_ms"])
            if request["tpot_ms"] > 0:
                tpots.append(request["tpot_ms"])
            times = request["token_times_ms"]
            itls.extend(b - a for a, b in zip(times, times[1:]))

    total_wall_s = sum(wall_ms) / 1000.0
    throughput = generated_total / total_wall_s if total_wall_s > 0 else 0.0

    # decode is memory bound. every step streams all weights once plus the live
    # KV for each sequence in the batch
    peak_bw = device["peak_bandwidth_bytes_per_sec"]
    mean_prompt_tokens = statistics.mean(prompt_tokens) if prompt_tokens else prompt_len
    mean_seq_len = mean_prompt_tokens + new_tokens / 2
    step_bytes = weights_bytes + batch * mean_seq_len * kv_bytes_per_token
    median_step_s = statistics.median(itls) / 1000.0 if itls else 0.0
    mbu = step_bytes / (median_step_s * peak_bw) if median_step_s > 0 else 0.0
    roofline_ms = step_bytes / peak_bw * 1000.0

    return {
        "batch": batch,
        "prompt_len": prompt_len,
        "mean_prompt_tokens": mean_prompt_tokens,
        "new_tokens": new_tokens,
        "reps": reps,
        "ttft_ms": percentiles(ttfts),
        "itl_ms": percentiles(itls),
        "e2e_ms": percentiles(e2es),
        "tpot_ms_mean": statistics.mean(tpots) if tpots else 0.0,
        "throughput_tok_s": throughput,
        "roofline_step_ms": roofline_ms,
        "mbu": mbu,
        "device": device,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="1,2,4",
                        help="comma separated batch sizes (engine caps at MAX_SEQUENCES)")
    parser.add_argument("--prompt-len", default="32,128,512")
    parser.add_argument("--new-tokens", default="64")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="write full results here")
    args = parser.parse_args()

    batches = [int(x) for x in args.batch.split(",")]
    prompt_lens = [int(x) for x in args.prompt_len.split(",")]
    new_tokens = [int(x) for x in args.new_tokens.split(",")]

    over = [b for b in batches if b > engine.MAX_SEQUENCES]
    if over:
        parser.error(f"batch {over} exceeds engine MAX_SEQUENCES={engine.MAX_SEQUENCES}")
    over = [p for p in prompt_lens if p > engine.MAX_PROMPT_LEN]
    if over:
        parser.error(f"prompt-len {over} exceeds engine MAX_PROMPT_LEN={engine.MAX_PROMPT_LEN}")

    rng = random.Random(args.seed)
    results = []

    header = (f"{'batch':>5} {'plen':>5} {'gen':>5} "
              f"{'TTFT p50':>9} {'p99':>8} {'ITL p50':>8} {'p99':>8} "
              f"{'tok/s':>8} {'roofline':>9} {'MBU':>6}")
    print(header)
    print("-" * len(header))

    for batch in batches:
        for prompt_len in prompt_lens:
            for gen in new_tokens:
                result = run_config(batch, prompt_len, gen, args.reps, rng)
                results.append(result)
                print(f"{batch:>5} {prompt_len:>5} {gen:>5} "
                      f"{result['ttft_ms']['p50']:>9.1f} {result['ttft_ms']['p99']:>8.1f} "
                      f"{result['itl_ms']['p50']:>8.2f} {result['itl_ms']['p99']:>8.2f} "
                      f"{result['throughput_tok_s']:>8.1f} "
                      f"{result['roofline_step_ms']:>9.2f} "
                      f"{result['mbu']*100:>5.1f}%")

    if results:
        device = results[0]["device"]
        print(f"\ndevice: {device['name']}, peak bandwidth "
              f"{device['peak_bandwidth_bytes_per_sec']/1e9:.1f} GB/s")
        print(f"weights: {engine.MODEL.stat().st_size/1e9:.2f} GB")
        print("MBU = bytes streamed per decode step / (measured step time x peak bandwidth).")
        print("Decode is memory bound, so MBU is the number to optimise.")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
