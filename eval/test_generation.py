#!/usr/bin/env python3
"""L3 sequence parity: engine greedy decode vs HF greedy decode.

This is the only test that exercises the decode path (KV cache writes,
ropeDecode positions, decodeSoftmax). test_logits.py only covers prefill.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine

PROMPTS = [
    "The capital of France is",
    "Once upon a time there was a",
    "The three primary colors are",
    "def quicksort(arr):",
    "Water freezes at",
    "The largest planet in our solar system is",
    "In order to make a cup of tea, first",
    "The first president of the United States was",
]


def reference_greedy(model, tokens, max_new_tokens):
    import torch

    generated = []
    ids = torch.tensor([tokens])
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids).logits[0, -1]
            nxt = int(torch.argmax(logits))
            if nxt in engine.STOP_TOKENS:
                break
            generated.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return generated


def first_divergence(a, b):
    """Index of the first differing token, or None if one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-divergence-ratio", type=float, default=0.6,
                        help="fail if mean first divergence < ratio * max_new_tokens")
    parser.add_argument("--json", help="write full results here")
    args = parser.parse_args()

    sequences = [(f"g{i}", text) for i, text in enumerate(PROMPTS)]

    print(f"generating {len(sequences)} sequences through the engine (greedy)...")
    records = engine.generate(sequences, greedy=True,
                              max_new_tokens=args.max_new_tokens, seed=0)
    by_id = {r["id"]: r for r in records}

    print("loading HF reference (fp32, CPU)...")
    model = engine.reference_model()

    rows = []
    divergences = []
    exact = 0
    failures = 0

    for ident, text in sequences:
        record = by_id.get(ident)
        if record is None:
            print(f"  {ident}: MISSING from engine output")
            failures += 1
            continue

        tokens = engine.encode(text)
        if record["prompt_tokens"] != len(tokens):
            print(f"  FAIL {ident}: engine tokenized the prompt to "
                  f"{record['prompt_tokens']} tokens, harness to {len(tokens)}. "
                  f"Tokenizer mismatch, continuations are not comparable")
            failures += 1
            continue

        engine_tokens = record["tokens"]
        reference_tokens = reference_greedy(model, tokens, args.max_new_tokens)

        index = first_divergence(engine_tokens, reference_tokens)
        matched = index is None and len(engine_tokens) == len(reference_tokens)
        if matched:
            exact += 1
        divergences.append(len(engine_tokens) if index is None else index)

        rows.append({
            "id": ident,
            "prompt": text,
            "engine_tokens": engine_tokens,
            "reference_tokens": reference_tokens,
            "first_divergence": index,
            "exact_match": matched,
        })

        where = "exact match" if index is None else f"diverges at {index}"
        print(f"  {ident}  {where:<18} {text[:40]!r}")
        if index is not None:
            tok = engine.tokenizer()
            print(f"        engine: {tok.decode(engine_tokens[max(0, index-2):index+3])!r}")
            print(f"        ref   : {tok.decode(reference_tokens[max(0, index-2):index+3])!r}")

    if divergences and not failures:
        mean_div = statistics.mean(divergences)
        print(f"\nexact matches      : {exact}/{len(rows)}")
        print(f"mean first divergence: {mean_div:.1f} tokens (of {args.max_new_tokens})")
        print(f"min first divergence : {min(divergences)}")
        floor = args.min_divergence_ratio * args.max_new_tokens
        ok = mean_div >= floor
        print(f"threshold            : {floor:.1f} "
              f"({args.min_divergence_ratio} x {args.max_new_tokens})")
        print("PASS" if ok else
              f"FAIL, mean divergence below {floor:.1f} suggests a real bug")
    else:
        if failures:
            print(f"\n{failures}/{len(sequences)} sequences could not be compared")
        ok = False

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
