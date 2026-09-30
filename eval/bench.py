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


def synthetic_prompt(length, rng):
    """Random in-vocabulary tokens. Content is irrelevant to timing."""
    return [engine.BOS_TOKEN] + [rng.randint(1000, 120000) for _ in range(length - 1)]


def run_config(batch, prompt_len, new_tokens, reps, rng):
    ttfts, tpots, e2es, itls = [], [], [], []
    wall_ms = []
    generated_total = 0
    device = None
    weights_bytes = engine.MODEL.stat().st_size
    kv_bytes_per_token = None

    for _ in range(reps):
        prompts = [(f"r{i}", synthetic_prompt(prompt_len, rng)) for i in range(batch)]
        records = engine.run(prompts=prompts, greedy=True, max_new_tokens=new_tokens, seed=0)

        requests = engine.of_type(records, "request")
        run = engine.run_record(records)
        device = run["device"]
        kv_bytes_per_token = run["kv_bytes_per_token"]
        wall_ms.append(run["wall_ms"])
        generated_total += run["generated_tokens"]

        for request in requests:
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
    mean_seq_len = prompt_len + new_tokens / 2
    step_bytes = weights_bytes + batch * mean_seq_len * kv_bytes_per_token
    median_step_s = statistics.median(itls) / 1000.0 if itls else 0.0
    mbu = step_bytes / (median_step_s * peak_bw) if median_step_s > 0 else 0.0
    roofline_ms = step_bytes / peak_bw * 1000.0

    return {
        "batch": batch,
        "prompt_len": prompt_len,
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
