#!/usr/bin/env python3
"""Run the full pipeline: prompt -> tokens -> engine -> tokens -> text.

Usage:
    python python/run.py "The capital of France is"

Paths can be overridden with the ENGINE and MODEL environment variables.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("ENGINE", ROOT / "engine"))
MODEL = Path(os.environ.get("MODEL", ROOT / "python" / "models" / "llama-3.2-1b" / "model.safetensors"))
TOKENIZER = os.environ.get("TOKENIZER", "meta-llama/Llama-3.2-1B")

# "slot 0 done, 20 tokens: 128000 791 6864"
DONE_LINE = re.compile(r"^slot \d+ done, \d+ tokens:((?: \d+)*)\s*$")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} \"<prompt>\"")
    prompt = sys.argv[1]

    if not ENGINE.exists():
        sys.exit(f"engine not found at {ENGINE}, build it first")
    if not MODEL.exists():
        sys.exit(f"model not found at {MODEL}, set MODEL=<path to .safetensors>")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt_ids = tokenizer.encode(prompt)

    result = subprocess.run(
        [str(ENGINE), str(MODEL), *map(str, prompt_ids)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        sys.exit(f"engine exited with {result.returncode}")

    generated = []
    for line in result.stdout.splitlines():
        match = DONE_LINE.match(line)
        if match:
            generated = [int(t) for t in match.group(1).split()]
            break

    if not generated:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        sys.exit("no completed slot in engine output")

    print(tokenizer.decode(prompt_ids + generated, skip_special_tokens=True, cleanup_tokenization_spaces=False))


if __name__ == "__main__":
    main()
