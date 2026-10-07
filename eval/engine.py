"""Shared client for driving the C++ engine binary.

Every other script in this directory goes through here so the subprocess and
JSONL handling lives in one place.

Environment overrides: ENGINE, MODEL, TOKENIZER, HF_MODEL.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> dict:
    """Minimal .env reader. Real environment variables win, so a shell export
    still overrides the file."""
    values = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            values[key] = value
    return values


_ENV = _load_dotenv(ROOT / ".env")


def _setting(name: str, default=None):
    return os.environ.get(name) or _ENV.get(name) or default


def _path_setting(name: str, default: Path) -> Path:
    value = _setting(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


ENGINE = _path_setting("ENGINE", ROOT / "build" / "engine")

MODEL_DIR = _path_setting("MODEL_DIR", ROOT / "python" / "models" / "llama-3.2-1b")
MODEL = _path_setting("MODEL", MODEL_DIR / "model.safetensors")
HF_MODEL = _setting("HF_MODEL") or str(MODEL_DIR)

MAX_PROMPT_LEN = int(_setting("MAX_PROMPT_LEN", 512))
MAX_SEQUENCES = int(_setting("MAX_SEQUENCES", 4))
MAX_SEQ_LEN = int(_setting("MAX_SEQ_LEN", 1024))
BOS_TOKEN = 128000
STOP_TOKENS = (128001, 128009)

_tokenizer = None
_tokenizer_json = None


class EngineError(RuntimeError):
    pass


def tokenizer_json() -> Path:
    """Path to tokenizer.json, resolved from the HF cache (no network if cached)."""
    global _tokenizer_json
    if _tokenizer_json is None:
        override = os.environ.get("TOKENIZER")
        if override:
            _tokenizer_json = Path(override)
        elif (MODEL_DIR / "tokenizer.json").exists():
            _tokenizer_json = MODEL_DIR / "tokenizer.json"
        else:
            from huggingface_hub import hf_hub_download
            _tokenizer_json = Path(hf_hub_download(HF_MODEL, "tokenizer.json"))
    return _tokenizer_json


def tokenizer():
    """HF tokenizer, used to build token id inputs and decode outputs."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    return _tokenizer


def encode(text: str, add_bos: bool = True) -> list[int]:
    ids = tokenizer().encode(text, add_special_tokens=False)
    if add_bos and (not ids or ids[0] != BOS_TOKEN):
        ids = [BOS_TOKEN] + ids
    return ids


def check_paths():
    if not ENGINE.exists():
        raise EngineError(f"engine not found at {ENGINE}, run ./build.sh")
    if not MODEL.exists():
        raise EngineError(f"model not found at {MODEL}, set MODEL=<path to .safetensors>")


def _flags_to_argv(flags: dict) -> list[str]:
    argv = []
    for key, value in flags.items():
        if value is None or value is False:
            continue
        name = "--" + key.replace("_", "-")
        if value is True:
            argv.append(name)
        else:
            argv += [name, str(value)]
    return argv


def escape_line(text: str) -> str:
    """Pack a prompt onto one line the way the engine's loadPrompts unescapes it."""
    out = []
    for char in text:
        if char == "\\":
            out.append("\\\\")
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        else:
            out.append(char)
    return "".join(out)


def run(prompts: list, timeout: int = 3600, **flags) -> list[dict]:
    """Invoke the engine once and return its parsed JSONL records.

    prompts is a list of prompt strings, or of (id, prompt string) pairs. They
    are written one per line to a temp text file which the engine tokenizes
    itself. The engine labels records by line index; ids given here are mapped
    back onto the returned records.

    Remaining keyword args become flags: greedy=True -> --greedy,
    max_new_tokens=8 -> --max-new-tokens 8.
    """
    check_paths()

    entries = []
    for index, entry in enumerate(prompts):
        ident, text = entry if isinstance(entry, tuple) else (str(index), entry)
        if not isinstance(text, str):
            raise EngineError(
                f"prompt {ident} is {type(text).__name__}, the engine takes text now")
        if not text.strip():
            raise EngineError(
                f"prompt {ident} is blank; the engine skips blank lines, which would "
                f"shift every later line index")
        length = len(encode(text))
        if length > MAX_PROMPT_LEN:
            raise EngineError(
                f"prompt {ident} encodes to {length} tokens, "
                f"engine MAX_PROMPT_LEN is {MAX_PROMPT_LEN}")
        entries.append((str(ident), text))

    if not entries:
        raise EngineError("no prompts given")

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="eval-prompts-")
        with os.fdopen(fd, "w") as handle:
            for _, text in entries:
                handle.write(escape_line(text) + "\n")

        cmd = [str(ENGINE), str(MODEL), str(tokenizer_json()), tmp_path]
        cmd += _flags_to_argv(flags)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise EngineError(
                f"engine exited {result.returncode}\ncmd: {' '.join(cmd)}\n{result.stderr}")

        records = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise EngineError(f"non-JSON line on stdout: {line[:200]}") from exc

        idents = [ident for ident, _ in entries]
        for record in records:
            if record.get("type") not in ("request", "score"):
                continue
            try:
                position = int(record["id"])
            except (KeyError, ValueError) as exc:
                raise EngineError(f"engine record has no line index id: {record}") from exc
            if not 0 <= position < len(idents):
                raise EngineError(
                    f"engine reported line index {position} for {len(idents)} prompts")
            record["id"] = idents[position]
        return records
    finally:
        if tmp_path:
            os.unlink(tmp_path)


def of_type(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r.get("type") == kind]


def run_record(records: list[dict]) -> dict:
    runs = of_type(records, "run")
    if not runs:
        raise EngineError("engine emitted no run record")
    return runs[0]


def generate(prompts: list, **flags) -> list[dict]:
    """Generate and return the request records only."""
    return of_type(run(prompts, **flags), "request")


def score(prompts: list, **flags) -> dict:
    """Score sequences, returning {id: record}."""
    records = of_type(run(prompts, score=True, **flags), "score")
    return {r["id"]: r for r in records}


def local_tensor(name: str):
    """Read one tensor out of the safetensors file the engine loads."""
    import json
    import struct

    import numpy as np

    with open(MODEL, "rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        if name not in header:
            raise EngineError(f"{MODEL} has no tensor {name}")
        start, end = header[name]["data_offsets"]
        handle.seek(8 + header_size + start)
        raw = handle.read(end - start)

    dtype = header[name]["dtype"]
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32)
    raise EngineError(f"unhandled dtype {dtype}")


def assert_same_weights(model, name: str = "model.norm.weight"):
    """The engine reads MODEL directly while the reference loads HF_MODEL. If
    those are different checkpoints every comparison downstream is meaningless,
    so fail loudly rather than reporting a bogus accuracy gap."""
    import numpy as np

    engine_weights = local_tensor(name)
    reference = model.state_dict()[name].detach().float().numpy().reshape(-1)
    delta = float(np.abs(engine_weights - reference).max())
    if delta > 1e-6:
        raise EngineError(
            f"WEIGHTS MISMATCH: engine and reference are different checkpoints.\n"
            f"  engine    MODEL={MODEL}\n"
            f"  reference HF_MODEL={HF_MODEL}\n"
            f"  {name} max|diff| = {delta}\n\n"
            f"Both should come from MODEL_DIR ({MODEL_DIR}). Unset any MODEL or\n"
            f"HF_MODEL override, or make sure that directory holds the config.json\n"
            f"and model.safetensors you intend to test.")


def reference_model(check_weights: bool = True):
    """HF model in fp32 on CPU, the ground truth these weights define."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise EngineError(
            "reference comparison needs torch. Install the CPU build:\n"
            "  python/venv/bin/pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu") from exc

    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, dtype=torch.float32)
    model.eval()
    if check_weights:
        assert_same_weights(model)
    return model


def reference_logprobs(model, tokens: list[int]) -> list[float]:
    """log P(tokens[i+1] | tokens[:i+1]) for every position, from the reference."""
    import torch

    with torch.no_grad():
        ids = torch.tensor([tokens])
        logits = model(ids).logits[0].float()
        logprobs = torch.log_softmax(logits[:-1], dim=-1)
        targets = ids[0, 1:]
        return logprobs.gather(1, targets.unsqueeze(1)).squeeze(1).tolist()
