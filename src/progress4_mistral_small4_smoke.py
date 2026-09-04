"""Progress 4 smoke runner for the hardware-risk Mistral Small 4 route.

This script records environment, checkpoint, vLLM engine-init, and one-token
generation evidence without changing model checkpoints or experiment data.
It is intentionally narrow: failures are classified as runtime feasibility
blockers, not model-quality failures.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Experiment" / "progress4_mistral_small4_resolution"
DEFAULT_MODEL_ID = "mistralai/Mistral-Small-4-119B-2603"
NVFP4_MODEL_ID = "mistralai/Mistral-Small-4-119B-2603-NVFP4"
PACKAGE_NAMES = [
    "vllm",
    "transformers",
    "mistral_common",
    "torch",
    "bitsandbytes",
    "accelerate",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_command(args: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"args": args, "error": repr(exc)}
    return {
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"import_error": repr(exc)}
    return {
        "torch_version": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def hf_cache_root() -> Path:
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(env_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_dir(model_id: str) -> Path:
    return hf_cache_root() / f"models--{model_id.replace('/', '--')}"


def snapshot_score(path: Path) -> tuple[bool, bool, float, str]:
    config = path / "config.json"
    tokenizer = path / "tokenizer_config.json"
    try:
        mtime = max((item.stat().st_mtime for item in path.iterdir()), default=0.0)
    except OSError:
        mtime = 0.0
    return (config.exists(), tokenizer.exists(), mtime, path.name)


def resolve_snapshot(model_id: str) -> Path | None:
    root = model_cache_dir(model_id) / "snapshots"
    if not root.exists():
        return None
    snapshots = [path for path in root.iterdir() if path.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=snapshot_score, reverse=True)
    return snapshots[0]


def resolve_model_source(model: str, allow_download: bool) -> dict[str, Any]:
    path = Path(model).expanduser()
    if path.exists():
        return {
            "requested": model,
            "source": str(path.resolve()),
            "source_type": "local_path",
            "local_files_only": True,
            "cache_dir": None,
            "snapshot_found": True,
        }
    snapshot = resolve_snapshot(model)
    if snapshot is not None:
        return {
            "requested": model,
            "source": str(snapshot),
            "source_type": "huggingface_cache_snapshot",
            "local_files_only": True,
            "cache_dir": str(model_cache_dir(model)),
            "snapshot_found": True,
        }
    return {
        "requested": model,
        "source": model,
        "source_type": "huggingface_model_id",
        "local_files_only": not allow_download,
        "cache_dir": str(model_cache_dir(model)),
        "snapshot_found": False,
    }


def load_config_and_tokenizer(source: str, local_files_only: bool, trust_remote_code: bool) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    result: dict[str, Any] = {}
    config = AutoConfig.from_pretrained(
        source,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    result["config"] = {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "text_config_architectures": getattr(getattr(config, "text_config", None), "architectures", None),
        "quantization_config": getattr(config, "quantization_config", None),
    }
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    result["tokenizer"] = {
        "class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
    }
    return result


def classify_error(text: str) -> str:
    lowered = text.lower()
    if "outofmemoryerror" in lowered or "cuda out of memory" in lowered or "oom" in lowered:
        return "runtime_feasibility_oom_or_vram"
    if "nonetype" in lowered and "architectures" in lowered:
        return "runtime_feasibility_vllm_model_config_compatibility"
    if "keyerror" in lowered and "expert" in lowered:
        return "runtime_feasibility_weight_mapping_compatibility"
    if "quantization" in lowered and "bitsandbytes" in lowered:
        return "runtime_feasibility_quantization_conflict"
    if "local_files_only" in lowered or "couldn't find" in lowered or "not found" in lowered:
        return "runtime_feasibility_checkpoint_missing_or_download_required"
    return "runtime_feasibility_unknown"


def build_hf_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.hf_overrides_json:
        loaded = json.loads(args.hf_overrides_json)
        if not isinstance(loaded, dict):
            raise ValueError("--hf-overrides-json must decode to an object")
        return loaded
    return None


def refine_vllm_result_with_config(result: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    if result.get("blocker_type") != "runtime_feasibility_unknown":
        return result
    config = checks.get("config", {}) if isinstance(checks, dict) else {}
    model_type = config.get("model_type")
    text_architectures = config.get("text_config_architectures")
    error_summary = str(result.get("error_summary", ""))
    if model_type == "mistral3" and text_architectures is None and "Engine core initialization failed" in error_summary:
        result["blocker_type"] = "runtime_feasibility_vllm_model_config_compatibility"
        result["blocker_evidence"] = (
            "Transformers config resolved model_type=mistral3 with text_config.architectures=null; "
            "vLLM worker initialization failed before generation."
        )
    return result


def attempt_vllm(args: argparse.Namespace, source: str, raw_log_path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": source,
        "tokenizer": source,
        "runner": "generate",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
        "language_model_only": args.language_model_only,
        "limit_mm_per_prompt": {"image": 0},
    }
    hf_overrides = build_hf_overrides(args)
    if hf_overrides:
        kwargs["hf_overrides"] = hf_overrides

    public_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"hf_token"} and not (isinstance(value, str) and value.startswith("hf_"))
    }
    result: dict[str, Any] = {"engine_args": public_kwargs, "engine_init": "not_started"}

    try:
        with raw_log_path.open("w", encoding="utf-8") as raw_log:
            with contextlib.redirect_stdout(raw_log), contextlib.redirect_stderr(raw_log):
                from vllm import LLM, SamplingParams

                llm = LLM(**kwargs)
                result["engine_init"] = "passed"
                params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
                outputs = llm.generate([args.prompt], params)
        result["generation"] = {
            "status": "passed",
            "prompt": args.prompt,
            "outputs": [
                {
                    "text": item.outputs[0].text if item.outputs else "",
                    "finish_reason": item.outputs[0].finish_reason if item.outputs else None,
                }
                for item in outputs
            ],
        }
        return result
    except Exception:
        text = traceback.format_exc()
        with raw_log_path.open("a", encoding="utf-8") as raw_log:
            raw_log.write("\n\n=== progress4_smoke_traceback ===\n")
            raw_log.write(text)
        raw_text = raw_log_path.read_text(encoding="utf-8", errors="replace")
        result["engine_init"] = "failed"
        result["generation"] = {"status": "not_run"}
        result["error_summary"] = text.splitlines()[-1] if text.splitlines() else "unknown error"
        result["blocker_type"] = classify_error(raw_text)
        result["raw_log"] = str(raw_log_path)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Progress 4 Mistral Small 4 smoke checks.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--nvfp4", action="store_true", help="Use the official NVFP4 checkpoint id.")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--attempt-engine", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--language-model-only", action="store_true", default=True)
    parser.add_argument("--hf-overrides-json", default="")
    parser.add_argument("--prompt", default="Return the number 1 as Python code.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nvfp4:
        args.model = NVFP4_MODEL_ID

    run_id = f"{utc_timestamp()}_{args.model.split('/')[-1].lower().replace('-', '_')}"
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = output_dir / "vllm_engine_raw.log"

    model_source = resolve_model_source(args.model, args.allow_download)
    payload: dict[str, Any] = {
        "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "goal": "Progress 4 runtime feasibility smoke for Mistral Small 4 generator route.",
        "model": model_source,
        "packages": package_versions(),
        "torch": torch_info(),
        "gpu_before": run_command(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_bus_id", "--format=csv"]
        ),
        "checks": {},
        "result": {},
        "status": "started",
        "runtime_quality_classification": "runtime_or_hardware_blocker_not_model_quality_failure",
    }

    try:
        payload["checks"]["transformers_config_tokenizer"] = load_config_and_tokenizer(
            model_source["source"],
            local_files_only=model_source["local_files_only"],
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        text = traceback.format_exc()
        raw_log_path.write_text(text, encoding="utf-8")
        payload["checks"]["transformers_config_tokenizer"] = {
            "status": "failed",
            "error_summary": text.splitlines()[-1] if text.splitlines() else "unknown error",
            "blocker_type": classify_error(text),
            "raw_log": str(raw_log_path),
        }
        payload["status"] = "completed_with_runtime_blocker"
    else:
        payload["checks"]["transformers_config_tokenizer"]["status"] = "passed"
        if args.attempt_engine:
            payload["result"]["vllm"] = attempt_vllm(args, model_source["source"], raw_log_path)
            payload["result"]["vllm"] = refine_vllm_result_with_config(
                payload["result"]["vllm"],
                payload["checks"].get("transformers_config_tokenizer", {}),
            )
            payload["status"] = (
                "completed"
                if payload["result"]["vllm"].get("generation", {}).get("status") == "passed"
                else "completed_with_runtime_blocker"
            )
        else:
            payload["result"]["vllm"] = {"status": "not_run", "reason": "missing --attempt-engine"}
            payload["status"] = "completed_metadata_only"

    payload["gpu_after"] = run_command(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_bus_id", "--format=csv"]
    )
    summary_path = output_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "status": payload["status"]}, indent=2))

    if payload["status"] == "completed_with_runtime_blocker":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
