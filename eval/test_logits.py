#!/usr/bin/env python3
"""L2 logit parity: engine token logprobs vs the HF reference.

Scores the same sequences through both and compares log P(token) position by
position. Catches numeric drift anywhere in the forward pass.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine

PROMPTS = [
    "The capital of France is Paris, and the capital of Germany is Berlin.",
    "In 1969 humans first walked on the surface of the Moon.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The mitochondrion is often called the powerhouse of the cell because it",
    "She sold seashells by the seashore, and the shells she sold were surely seashells.",
    "Quantum entanglement describes a physical phenomenon in which pairs of particles",
    "To be, or not to be, that is the question: whether tis nobler in the mind to suffer",
]


def compare(engine_lp, reference_lp):
    a = np.array(engine_lp, dtype=np.float64)
    b = np.array(reference_lp, dtype=np.float64)
    diff = np.abs(a - b)
    # pearson r, guarding the degenerate constant-vector case
    if a.std() > 0 and b.std() > 0:
        correlation = float(np.corrcoef(a, b)[0, 1])
    else:
        correlation = float("nan")
    return {
        "positions": int(a.size),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "rel_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        "correlation": correlation,
        "engine_sum": float(a.sum()),
        "reference_sum": float(b.sum()),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tolerance", type=float, default=0.25,
                        help="max absolute logprob difference allowed per position")
    parser.add_argument("--mean-tolerance", type=float, default=0.06,
                        help="max mean absolute difference allowed per sequence")
    parser.add_argument("--min-correlation", type=float, default=0.9995,
                        help="min pearson r between engine and reference logprobs")
    parser.add_argument("--json", help="write full results here")
    args = parser.parse_args()

    sequences = [(f"p{i}", engine.encode(text)) for i, text in enumerate(PROMPTS)]

    print(f"scoring {len(sequences)} sequences through the engine...")
    engine_results = engine.score(sequences)

    print("loading HF reference (fp32, CPU)...")
    model = engine.reference_model()

    rows = []
    failures = 0
    for ident, tokens in sequences:
        record = engine_results.get(ident)
        if record is None:
            print(f"  {ident}: MISSING from engine output")
            failures += 1
            continue

        reference_lp = engine.reference_logprobs(model, tokens)
        stats = compare(record["logprobs"], reference_lp)
        stats["id"] = ident
        stats["ok"] = (stats["max_abs_diff"] <= args.tolerance
                       and stats["mean_abs_diff"] <= args.mean_tolerance
                       and stats["correlation"] >= args.min_correlation)
        rows.append(stats)
        if not stats["ok"]:
            failures += 1

        flag = "ok  " if stats["ok"] else "FAIL"
        print(f"  {flag} {ident}  n={stats['positions']:>3}  "
              f"max={stats['max_abs_diff']:.4f}  mean={stats['mean_abs_diff']:.4f}  "
              f"relL2={stats['rel_l2']:.4f}  r={stats['correlation']:.6f}")

    if rows:
        print(f"\nworst max_abs_diff : {max(r['max_abs_diff'] for r in rows):.4f}"
              f"  (tolerance {args.tolerance})")
        print(f"worst mean_abs_diff: {max(r['mean_abs_diff'] for r in rows):.4f}"
              f"  (tolerance {args.mean_tolerance})")
        print(f"min correlation    : {min(r['correlation'] for r in rows):.6f}"
              f"  (tolerance {args.min_correlation})")

    print("\nNote: this compares log P at the observed token, not the full 128k "
          "distribution.\nTrue KL divergence needs a --dump-logits flag the engine "
          "does not have yet.")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")

    print(f"\n{len(rows) - failures}/{len(sequences)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
