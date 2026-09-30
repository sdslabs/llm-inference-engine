#!/usr/bin/env python3
"""Multiple choice task accuracy via loglikelihood scoring.

For each question, every candidate continuation is scored and the highest
scoring one wins. No generation involved, which is why score mode is enough.

Reports two numbers, both standard:
  acc      argmax of total continuation logprob
  acc_norm argmax of logprob per continuation token, which corrects the bias
           against longer answers
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine

TASKS = {
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "config": None,
        "split": "validation",
    },
    "arc_easy": {
        "path": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "split": "test",
    },
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
    },
}


def load_items(task, limit):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("needs `datasets`:\n  python/venv/bin/pip install datasets")

    spec = TASKS[task]
    data = load_dataset(spec["path"], spec["config"], split=spec["split"])
    if limit:
        data = data.select(range(min(limit, len(data))))

    items = []
    for row in data:
        if task == "hellaswag":
            context = row["ctx"]
            choices = row["endings"]
            answer = int(row["label"])
        else:
            context = "Question: " + row["question"] + "\nAnswer:"
            choices = row["choices"]["text"]
            labels = row["choices"]["label"]
            answer = labels.index(row["answerKey"])
            choices = [" " + c for c in choices]
        items.append({"context": context, "choices": choices, "answer": answer})
    return items


def build_sequences(items):
    """One sequence per (question, choice). Records where the continuation starts
    so its logprobs can be sliced out of the full sequence score."""
    sequences = []
    meta = []
    skipped = 0

    for q, item in enumerate(items):
        context_ids = engine.encode(item["context"])
        entries = []
        for c, choice in enumerate(item["choices"]):
            choice_ids = engine.tokenizer().encode(choice, add_special_tokens=False)
            tokens = context_ids + choice_ids
            if len(tokens) > engine.MAX_PROMPT_LEN or not choice_ids:
                entries = None
                break
            ident = f"q{q}c{c}"
            sequences.append((ident, tokens))
            entries.append({"id": ident,
                            "context_len": len(context_ids),
                            "choice_len": len(choice_ids)})
        if entries is None:
            skipped += 1
            meta.append(None)
        else:
            meta.append({"answer": item["answer"], "entries": entries})
    return sequences, meta, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="hellaswag")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--json", help="write per question results here")
    args = parser.parse_args()

    print(f"loading {args.task}...")
    items = load_items(args.task, args.limit)
    sequences, meta, skipped = build_sequences(items)

    if skipped:
        print(f"skipped {skipped} questions that exceed MAX_PROMPT_LEN="
              f"{engine.MAX_PROMPT_LEN}")
    print(f"scoring {len(sequences)} continuations for {len(items)-skipped} questions...")

    results = engine.score(sequences)

    correct = 0
    correct_norm = 0
    total = 0
    rows = []

    for question in meta:
        if question is None:
            continue
        totals, norms = [], []
        for entry in question["entries"]:
            record = results.get(entry["id"])
            if record is None:
                sys.exit(f"engine returned no score for {entry['id']}")
            # logprobs[i] is for token i+1, so the continuation starts at
            # index context_len - 1
            choice_lp = record["logprobs"][entry["context_len"] - 1:]
            total_lp = sum(choice_lp)
            totals.append(total_lp)
            norms.append(total_lp / max(1, len(choice_lp)))

        predicted = max(range(len(totals)), key=totals.__getitem__)
        predicted_norm = max(range(len(norms)), key=norms.__getitem__)
        answer = question["answer"]

        total += 1
        correct += predicted == answer
        correct_norm += predicted_norm == answer
        rows.append({"answer": answer, "pred": predicted, "pred_norm": predicted_norm,
                     "totals": totals, "norms": norms})

    if not total:
        sys.exit("no questions scored")

    print(f"\ntask      : {args.task}")
    print(f"questions : {total}")
    print(f"acc       : {correct/total*100:.2f}%  ({correct}/{total})")
    print(f"acc_norm  : {correct_norm/total*100:.2f}%  ({correct_norm}/{total})")
    print(f"random    : {100/len(rows[0]['totals']):.2f}%")
    print("\nCompare acc_norm against published Llama-3.2-1B numbers. A large "
          "shortfall\nmeans the engine is damaging the distribution, not that the "
          "model is weak.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"task": args.task, "acc": correct/total,
             "acc_norm": correct_norm/total, "questions": rows}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
