"""Prefetch Experiment 7 local LLM-agent models into the Hugging Face cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", WORKSPACE_ROOT / "Models")).expanduser()
os.environ.setdefault("HF_HOME", str(MODELS_ROOT / ".cache" / "huggingface"))

from huggingface_hub import snapshot_download  # noqa: E402

MODELS = {
    "deepseek_r1_qwen32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistral4": "mistralai/Mistral-Small-4-119B-2603-NVFP4",
    "llama3_3": "meta-llama/Llama-3.3-70B-Instruct",
    "llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "qwythos9b": "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
    "qwen3_6": "Qwen/Qwen3.6-35B-A3B-FP8",
}

ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.jinja",
    "*.model",
    "*.tiktoken",
    "*.py",
    "LICENSE",
    "README.md",
    "merges.txt",
    "vocab.json",
    ".gitattributes",
]
IGNORE_PATTERNS = ["original/*"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch local vLLM model files for Experiment 7 LLM agents.")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--status-json", type=Path)
    return parser.parse_args()


def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def disk_payload(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": round(bytes_to_gib(usage.free), 2),
    }


def validate_snapshot(snapshot_path: str) -> dict[str, Any]:
    path = Path(snapshot_path)
    if not path.is_dir():
        raise RuntimeError(f"snapshot path is not a directory: {path}")
    config_files = ["config.json", "params.json"]
    config_present = [name for name in config_files if (path / name).exists()]
    if not config_present:
        raise RuntimeError(f"missing config.json or params.json in {path}")
    tokenizer_files = ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"]
    tokenizer_present = [name for name in tokenizer_files if (path / name).exists()]
    if not tokenizer_present:
        raise RuntimeError(f"missing tokenizer files in {path}")
    index_files = sorted(path.glob("*safetensors.index.json"))
    shard_count = 0
    if index_files:
        missing = []
        for index_file in index_files:
            data = json.loads(index_file.read_text(encoding="utf-8"))
            weight_map = data.get("weight_map", {})
            filenames = sorted({value for value in weight_map.values() if isinstance(value, str)})
            shard_count += len(filenames)
            missing.extend(name for name in filenames if not (path / name).exists())
        if missing:
            sample = ", ".join(missing[:5])
            raise RuntimeError(f"missing safetensors shards in {path}: {sample}")
    else:
        safetensors = sorted(path.glob("*.safetensors"))
        if not safetensors:
            raise RuntimeError(f"missing safetensors weights in {path}")
        shard_count = len(safetensors)
    return {
        "snapshot_path": str(path),
        "config_present": config_present,
        "tokenizer_present": tokenizer_present,
        "index_files": [item.name for item in index_files],
        "safetensors_shards": shard_count,
    }


def fetch_model(engine: str, model_id: str, local_files_only: bool) -> dict[str, Any]:
    snapshot_path = snapshot_download(
        model_id,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        local_files_only=local_files_only,
    )
    validation = validate_snapshot(snapshot_path)
    return {
        "engine": engine,
        "model_id": model_id,
        "status": "ready",
        **validation,
    }


def main() -> None:
    args = parse_args()
    hf_home = Path(os.environ["HF_HOME"]).expanduser()
    hf_home.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "hf_home": str(hf_home),
        "allow_patterns": ALLOW_PATTERNS,
        "ignore_patterns": IGNORE_PATTERNS,
        "disk_before": disk_payload(hf_home),
        "models": [],
    }
    exit_code = 0
    for engine in args.models:
        model_id = MODELS[engine]
        try:
            item = fetch_model(engine, model_id, args.local_files_only)
        except Exception as exc:  # keep going so partial prefetch state is recorded
            exit_code = 1
            item = {
                "engine": engine,
                "model_id": model_id,
                "status": "blocked",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        payload["models"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    payload["disk_after"] = disk_payload(hf_home)
    payload["status"] = "completed" if exit_code == 0 else "blocked_or_partial"
    payload["exit_code"] = exit_code
    if args.status_json:
        args.status_json.parent.mkdir(parents=True, exist_ok=True)
        args.status_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
