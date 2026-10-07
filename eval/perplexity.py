#!/usr/bin/env python3
"""L4 perplexity, the single best correctness signal for an inference engine.

Windows a corpus into MAX_PROMPT_LEN chunks, scores each through the engine, and
reports exp(-mean logprob). The absolute number is not the point, the delta
against the HF reference on the same windows is.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine


def load_text(args):
    if args.text:
        return Path(args.text).read_text(encoding="utf-8")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("--dataset needs `datasets`:\n"
                 "  python/venv/bin/pip install datasets\n"
                 "or pass --text <file> instead")
    data = load_dataset(args.dataset, args.dataset_config, split=args.split)
    return "\n\n".join(data["text"])


def windows(tokens, size, stride):
    """Yields (chunk, scored) pairs. With stride < size each window re-reads
    earlier tokens purely as context, so only the positions no previous window
    already predicted are counted. Scoring the whole window would count those
    twice and give a perplexity comparable to nothing.

    Note a token at the very start of a window has no preceding context inside
    it, so with stride == size one position per window boundary is unscorable.
    That is inherent to non overlapping windowing, not a bug."""
    out = []
    covered = 0
    limit = len(tokens) - 1
    for start in range(0, limit, stride):
        chunk = tokens[start:start + size]
        if len(chunk) < 2:
            break
        last = start + len(chunk) - 1
        first_new = max(covered + 1, start + 1)
        scored = last - first_new + 1
        if scored > 0:
            out.append((chunk, scored))
            covered = last
        if last >= limit:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="local text file")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--window", type=int, default=engine.MAX_PROMPT_LEN - 1)
    parser.add_argument("--stride", type=int, default=0,
                        help="0 means non overlapping (stride = window)")
    parser.add_argument("--limit", type=int, default=0, help="cap number of windows")
    parser.add_argument("--compare", action="store_true",
                        help="also compute reference perplexity (needs torch, slow)")
    parser.add_argument("--json", help="write full results here")
    args = parser.parse_args()

    if args.window > engine.MAX_PROMPT_LEN - 1:
        parser.error(f"--window must leave room for the BOS the engine prepends, "
                     f"so at most MAX_PROMPT_LEN-1={engine.MAX_PROMPT_LEN - 1}")
    stride = args.stride or args.window

    print("tokenizing corpus...")
    text = load_text(args)
    tokens = engine.tokenizer().encode(text, add_special_tokens=False)
    chunks = windows(tokens, args.window, stride)
    if args.limit:
        chunks = chunks[:args.limit]

    sequences = []
    scored_counts = []
    token_ids = []
    rejected = 0
    for index, (chunk, scored) in enumerate(chunks):
        window_text = engine.tokenizer().decode(chunk)
        full_ids = engine.encode(window_text)
        if full_ids[1:] != list(chunk):
            rejected += 1
            continue
        sequences.append((f"w{index}", window_text))
        scored_counts.append(scored)
        token_ids.append(full_ids)

    if not sequences:
        sys.exit("no windows survived the decode/encode round trip")
    if rejected:
        print(f"dropped {rejected}/{len(chunks)} windows whose text does not "
              f"re-tokenize to the same ids")

    total_positions = sum(scored_counts)
    print(f"{len(tokens)} tokens -> {len(sequences)} windows of <={args.window} "
          f"(stride {stride}, {total_positions} scored positions)")

    results = engine.score(sequences)

    engine_sum = 0.0
    engine_count = 0
    for (ident, _), scored, full_ids in zip(sequences, scored_counts, token_ids):
        record = results.get(ident)
        if record is None:
            sys.exit(f"engine returned no score for {ident}")
        if record["num_tokens"] != len(full_ids):
            sys.exit(f"{ident}: engine tokenized to {record['num_tokens']} tokens, "
                     f"harness to {len(full_ids)}. Tokenizer mismatch")
        engine_sum += sum(record["logprobs"][-scored:])
        engine_count += scored

    engine_ppl = math.exp(-engine_sum / engine_count)
    print(f"\nengine     perplexity: {engine_ppl:.4f}  "
          f"(mean logprob {engine_sum/engine_count:+.4f}, {engine_count} positions)")

    summary = {
        "windows": len(sequences),
        "windows_dropped": rejected,
        "window": args.window,
        "stride": stride,
        "positions": engine_count,
        "engine_ppl": engine_ppl,
        "engine_mean_logprob": engine_sum / engine_count,
    }

    if args.compare:
        print("loading HF reference (fp32, CPU), this is slow...")
        model = engine.reference_model()
        reference_sum = 0.0
        reference_count = 0
        for index, (full_ids, scored) in enumerate(zip(token_ids, scored_counts)):
            logprobs = engine.reference_logprobs(model, full_ids)
            reference_sum += sum(logprobs[-scored:])
            reference_count += scored
            print(f"  window {index+1}/{len(sequences)}", end="\r", flush=True)
        print(" " * 40, end="\r")

        reference_ppl = math.exp(-reference_sum / reference_count)
        delta = engine_ppl - reference_ppl
        print(f"reference  perplexity: {reference_ppl:.4f}")
        print(f"delta                : {delta:+.4f} ({delta/reference_ppl*100:+.2f}%)")
        print("\nA delta near zero means the engine reproduces the reference "
              "faithfully.\nA large positive delta means real numeric damage "
              "somewhere in the forward pass.")
        summary["reference_ppl"] = reference_ppl
        summary["delta"] = delta

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
