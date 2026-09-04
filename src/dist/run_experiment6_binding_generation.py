#!/usr/bin/env python3
"""Generate Experiment 6 binding predictions from prompt CSV inputs."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import faulthandler
import importlib
import json
import multiprocessing as mp
import queue as queue_module
import re
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from binding_extraction import extract_result_items, item_dict

csv.field_size_limit(sys.maxsize)
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
RETRIEVER_ROOT = REPO_ROOT / ".external/FINDER/Retriever Codes"

ROUTE_CSV_PATHS = {
    "narrative_original": WORKSPACE_ROOT / "data" / "finqa_original" / "narratives1_rel_fact_instruction.csv",
    "narrative_zero_shot": WORKSPACE_ROOT / "data" / "finqa_zero_shot" / "narratives1_rel_fact_instruction.csv",
    "narrative_many_shot": WORKSPACE_ROOT / "data" / "finqa_many_shot" / "narratives1_rel_fact_instruction.csv",
    "narrative_dynamic_shot": WORKSPACE_ROOT / "data" / "finqa_dynamic_shot" / "narratives1_rel_fact_instruction.csv",
    "btc_20241224": REPO_ROOT / "Experiment" / "btc_finflier_custom" / "btc_20241224_rel_fact_instruction.csv",
}
ROUTE_PROMPT_MODE = {
    "narrative_original": "original",
    "narrative_zero_shot": "zero-shot",
    "narrative_many_shot": "many-shot",
    "narrative_dynamic_shot": "dynamic-shot",
    "btc_20241224": "original",
}
BINDING_JSON_SYSTEM_PROMPT = """You are a financial data-binding assistant.
Return only strict JSON. Do not return Python code, markdown, explanations, or prose.
The JSON must be either {"Binding":[...]} or a list of binding objects.
Each binding object should use these keys when known: ObjectName, DataName, Position, Trend, Num, Text.
ObjectName should name the subject/entity/measure being described.
Trend should be a concise trend phrase; use "None" only when no trend exists.
Num should contain only measured data values asserted by the narrative; use [] when no number exists.
Do not put coordinate/index/category values such as years, BMI, row labels, or column labels in Num unless they are the measured value itself.
Preserve ObjectName as close as possible to the narrative phrasing, including qualifiers such as category, year, BMI, period, or condition.
Do not output hidden reasoning tags such as <think>.
Do not invent values absent from the chart data, narrative text, prompt, or candidate RetFact.
""".strip()

DIRECT_BINDING_SUFFIX = """

Return only strict JSON for the data-binding result. Do not output Python code. The preferred shape is:
{"Binding":[{"ObjectName":[],"DataName":"","Position":[{"Begin":[],"End":[]}],"Trend":"None","Num":[],"Text":""}]}"""

RETRIEVER_TO_BINDING_INSTRUCTION = """Convert the candidate RetFact/retriever output into a data-binding JSON result.
Use the chart data and narrative text as the authority. Use the retriever output only as candidate evidence.
For Num, keep only the chart/table value being asserted, not coordinate labels such as BMI/year/category.
Return only strict JSON in the shape {"Binding":[...]}."""
FAMILY_BASE_ENGINE = {
    "flan": "flan_t5_large",
    "mistral": "mistral_v0_3",
    "t5gemma2": "t5gemma_2_1b_1b",
}
BASE_RETRIEVER_SOURCE_FAMILIES = {
    "flan_t5_large": "flan",
    "mistral_v0_3": "mistral",
    "t5gemma_2_1b_1b": "t5gemma2",
}
FINETUNED_RETRIEVER_SOURCE_FAMILIES = {
    "finqa_flan_z": "flan",
    "finqa_flan_m": "flan",
    "finqa_flan_d": "flan",
    "finqa_mistral_z": "mistral",
    "finqa_mistral_m": "mistral",
    "finqa_mistral_d": "mistral",
    "finqa_t5gemma2_z": "t5gemma2",
    "finqa_t5gemma2_m": "t5gemma2",
    "finqa_t5gemma2_d": "t5gemma2",
}


@dataclass(frozen=True)
class MatrixCase:
    experiment_id: str
    source_id: str
    narrative_route: str


class RowTimeoutError(RuntimeError):
    pass


class MissingBindingGeneratorError(RuntimeError):
    def __init__(self, message: str, *, runtime: dict[str, Any] | None = None):
        super().__init__(message)
        self.runtime = runtime or {}
        self.failure_category = "runtime_blocked_missing_binding_generator"


class BindingGenerationRuntimeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        runtime: dict[str, Any] | None = None,
        failure_category: str = "blocked_model_generation_runtime",
    ):
        super().__init__(message)
        self.runtime = runtime or {}
        self.failure_category = failure_category


class ResponseModelIdentityError(RuntimeError):
    """Raised when a route returns a model other than its required identity."""

    failure_category = "runtime_blocked_model_identity"


def require_response_model_identity(engine: str, config: Any, metadata: Any) -> None:
    """Enforce the GPT-5.5 response identity instead of trusting request metadata."""
    if engine != "gpt5_5":
        return
    required = str(
        getattr(config, "actual_model", "")
        or getattr(config, "model", "")
        or "gpt-5.5"
    )
    observed = str(
        metadata.get("responseModel", "")
        if isinstance(metadata, dict)
        else ""
    )
    if observed != required:
        raise ResponseModelIdentityError(
            "GPT-5.5 response model identity mismatch: "
            f"required={required!r}, observed={observed!r}"
        )


def generation_failure_category(runtime: Any, exc: BaseException) -> str:
    """Prefer an explicit trust-boundary category over backend heuristics."""
    return str(
        getattr(exc, "failure_category", "")
        or runtime.classify_generation_exception(exc)
    )


def _parallel_generation_target(
    queue: Any,
    runtime_module_name: str,
    config: Any,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
) -> None:
    """Run one generation in an isolated child used by parallel rows."""
    try:
        runtime = importlib.import_module(runtime_module_name)
        outputs = runtime.generate_text(
            config,
            prompt,
            system_prompt,
            "greedy",
            max_tokens=max_tokens,
        )
        metadata = (
            runtime.last_generation_metadata()
            if hasattr(runtime, "last_generation_metadata")
            else {}
        )
        queue.put({"ok": True, "outputs": outputs, "metadata": metadata})
    except BaseException as exc:  # pragma: no cover - child process evidence path
        queue.put(
            {
                "ok": False,
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc()[-2000:],
            }
        )


SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "CHATMOCK_API_KEY",
    "VLLM_API_KEY",
    "HF_TOKEN",
)
RUNTIME_ENV_NAMES = (
    *SECRET_ENV_NAMES,
    "OPENAI_BASE_URL",
    "ALLOW_OPENAI_COMPATIBLE_EXECUTE",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
    "CHATMOCK_BASE_URL",
    "GPT5_3_CODEX_ROUTE",
    "GPT5_5_CODEX_ROUTE",
    "VLLM_BASE_URL",
    "VLLM_SERVED_MODEL_NAME",
    "QWEN3_6_MODEL",
    "LLAMA3_3_MODEL",
)
FATAL_ERROR_HINTS = (
    "credential_blocked",
    "missing_credentials",
    "missing=[",
    "missing adapter_config",
    "filenotfounderror",
    "no module named",
    "modulenotfounderror",
    "importerror",
    "package",
    "dependency",
    "not executable",
    "cuda out of memory",
    "outofmemory",
)
TRANSIENT_ERROR_HINTS = (
    "timeout",
    "timed out",
    "rate limit",
    "ratelimit",
    "http 429",
    "status code: 429",
    "too many requests",
    "quota",
    "temporarily",
    "temporary",
    "connection",
    "endpoint",
    "not ready",
    "service unavailable",
    "server disconnected",
    "read timed out",
    "connect timeout",
    "stopped after",
)


def public_env_status() -> dict[str, str]:
    return {name: ("set" if os.environ.get(name) else "unset") for name in RUNTIME_ENV_NAMES}


def truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_variables_md_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}", value)
    if match:
        existing = os.environ.get(match.group(1))
        return existing if existing else match.group(2)
    return value


def load_workspace_variables_md(args: argparse.Namespace) -> dict[str, Any]:
    if not args.load_variables_md:
        return {"enabled": False, "loaded": []}
    path = args.variables_md
    if not path.is_file():
        return {"enabled": True, "path": str(path), "loaded": [], "missing_file": True}
    allowed = set(RUNTIME_ENV_NAMES) | {
        "AZURE_OPENAI_GPT4_1_DEPLOYMENT",
        "AZURE_OPENAI_GPT5_5_DEPLOYMENT",
        "AZURE_OPENAI_GPT5_3_CODEX_DEPLOYMENT",
        "OPENAI_GPT5_5_MODEL",
        "OPENAI_GPT5_3_CODEX_MODEL",
        "GPT5_5_CODEX_ROUTE",
        "GPT5_3_CODEX_ROUTE",
    }
    loaded: list[str] = []
    override = truthy_env(os.environ.get("EXPERIMENT6_VARIABLES_MD_OVERRIDE"))
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, raw_value = line[7:].split("=", 1)
        key = key.strip()
        if key not in allowed:
            continue
        if os.environ.get(key) and not override:
            continue
        value = parse_variables_md_value(raw_value)
        if not value or value.startswith("<"):
            continue
        os.environ[key] = value
        loaded.append(key)
    return {"enabled": True, "path": str(path), "loaded": sorted(set(loaded)), "override": override}


def is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, RowTimeoutError):
        return True
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if any(hint in text for hint in FATAL_ERROR_HINTS):
        return False
    return any(hint in text for hint in TRANSIENT_ERROR_HINTS)


def rounded_wait_seconds(seconds: int) -> int:
    if seconds <= 0:
        return 0
    quantum = 600
    return ((seconds + quantum - 1) // quantum) * quantum


def resume_command(args: argparse.Namespace, case: MatrixCase) -> str:
    return (
        f"EXPERIMENT6_BINDING_MATRIX={case.experiment_id}:{case.source_id}:{case.narrative_route} "
        f"EXPERIMENT6_NUM_RUNS={args.num_runs} EXPERIMENT6_TOP_K={args.top_k} "
        f"EXPERIMENT6_RESUME_RUNS=1 EXPERIMENT6_RETRY_MAX={args.retry_max} "
        f"EXPERIMENT6_RETRY_WAIT_SECONDS={args.retry_wait_seconds} "
        f"EXPERIMENT6_DEBUG={1 if args.debug else 0} "
        f"EXPERIMENT6_GENERATION_MODE={args.mode} "
        f"bash dist/experiment_6_data_binding_evaluation.sh"
    )


def write_debug_error(
    args: argparse.Namespace,
    case: MatrixCase,
    run_number: int,
    attempt: int,
    exc: BaseException,
    *,
    transient: bool,
    wait_seconds: int,
) -> Path:
    debug_dir = args.pred_dir / "debug" / case.experiment_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"run_{run_number:02d}.attempt_{attempt:02d}.error.json"
    payload = {
        "time": utc_now(),
        "experiment_id": case.experiment_id,
        "source_id": case.source_id,
        "narrative_route": case.narrative_route,
        "run": run_number,
        "attempt": attempt,
        "transient": transient,
        "wait_seconds": wait_seconds,
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc()[-4000:],
        },
        "env_status": public_env_status(),
        "resume_command": resume_command(args, case),
    }
    write_json(path, payload)
    return path


def generate_text_with_timeout(
    generator_runtime: Any,
    config: Any,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    row_timeout_seconds: int,
    *,
    return_metadata: bool = False,
) -> Any:
    if row_timeout_seconds <= 0:
        outputs = generator_runtime.generate_text(
            config, prompt, system_prompt, "greedy", max_tokens=max_tokens,
        )
        metadata = (
            generator_runtime.last_generation_metadata()
            if hasattr(generator_runtime, "last_generation_metadata") else {}
        )
        return (outputs, metadata) if return_metadata else outputs

    queue: Any = mp.Queue(maxsize=1)

    def target() -> None:
        try:
            outputs = generator_runtime.generate_text(
                config,
                prompt,
                system_prompt,
                "greedy",
                max_tokens=max_tokens,
            )
            metadata = (
                generator_runtime.last_generation_metadata()
                if hasattr(generator_runtime, "last_generation_metadata") else {}
            )
            queue.put({"ok": True, "outputs": outputs, "metadata": metadata})
        except BaseException as exc:  # pragma: no cover - child process evidence path
            queue.put(
                {
                    "ok": False,
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback_tail": traceback.format_exc()[-2000:],
                }
            )

    process = mp.Process(target=target)
    process.start()
    process.join(row_timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise RowTimeoutError(f"generator row timed out after {row_timeout_seconds}s")
    if queue.empty():
        raise RuntimeError(f"generator row exited without output; exitcode={process.exitcode}")
    payload = queue.get()
    if payload.get("ok"):
        outputs = payload.get("outputs") or []
        metadata = payload.get("metadata") or {}
        return (outputs, metadata) if return_metadata else outputs
    raise RuntimeError(
        f"child generation failed: {payload.get('type')}: {payload.get('message')}\n{payload.get('traceback_tail')}"
    )


def generate_codes_with_timeout(generator_runtime: Any, config: Any, prompt: str, max_tokens: int, row_timeout_seconds: int) -> list[str]:
    return generate_text_with_timeout(
        generator_runtime,
        config,
        prompt,
        "Return only Python code that computes the answer.",
        max_tokens,
        row_timeout_seconds,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_matrix(raw: str) -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    for item in raw.split():
        parts = item.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid matrix item: {item}")
        cases.append(MatrixCase(parts[0], parts[1], parts[2]))
    return cases


def read_prompt_rows(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if limit and limit > 0:
        rows = rows[:limit]
    missing = {"Source", "input"} - set(rows[0]) if rows else {"Source", "input"}
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return rows


def normalized_source_id(source_id: str) -> str:
    return str(source_id or "").strip().lower()


def finetuned_retriever_family(source_id: str) -> str | None:
    return FINETUNED_RETRIEVER_SOURCE_FAMILIES.get(normalized_source_id(source_id))


def base_retriever_family(source_id: str) -> str | None:
    return BASE_RETRIEVER_SOURCE_FAMILIES.get(normalized_source_id(source_id))


def family_from_source_id(source_id: str) -> str | None:
    family = finetuned_retriever_family(source_id)
    if family is not None:
        return family
    family = base_retriever_family(source_id)
    if family is not None:
        return family
    return None


def is_finetuned_retriever_source(source_id: str) -> bool:
    return finetuned_retriever_family(source_id) is not None


def adapter_dir_for(source_id: str) -> Path:
    return REPO_ROOT / "Experiment" / source_id / "retriever" / "model"


def extract_pred_text(raw: str) -> str:
    text = str(raw).strip()
    if " Pred: " in text:
        return text.rsplit(" Pred: ", 1)[1].strip()
    return text


def read_raw_prediction_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            rows.append(extract_pred_text(line))
            continue
        if isinstance(payload, dict):
            rows.append(str(payload.get("predicted_label") or payload.get("prediction") or payload.get("output") or ""))
        else:
            rows.append(str(payload))
    return rows


def prediction_rows_from_texts(case: MatrixCase, prompt_rows: list[dict[str, str]], predictions: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for index, row in enumerate(prompt_rows):
        case_id = str(row.get("Source") or f"row_{index + 1}").strip()
        raw_prediction = predictions[index] if index < len(predictions) else ""
        record = {"case_id": case_id, "prediction": raw_prediction}
        items, _records, report = extract_result_items(record, fallback_case_id=case_id, row_number=index + 1, strict=False)
        output_rows.append(
            {
                "case_id": case_id,
                "items": [item_dict(item) for item in items],
                "raw_prediction": raw_prediction,
            }
        )
        reports.append(report)
    return output_rows, reports


def write_prediction_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"case_id": row["case_id"], "items": row["items"]}, ensure_ascii=False, sort_keys=True) + "\n")


def read_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} row must be object")
            case_id = str(payload.get("case_id") or "")
            items = payload.get("items")
            if not case_id or not isinstance(items, list):
                raise ValueError(f"{path}:{line_number} missing case_id/items")
            rows.append({"case_id": case_id, "items": items})
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cuda_library_dirs() -> list[str]:
    """Return CUDA library dirs shipped with the active Python environment."""
    roots = [
        Path(sys.prefix) / "lib" / "python3.10" / "site-packages" / "nvidia",
        Path(sys.prefix) / "lib" / "python3.11" / "site-packages" / "nvidia",
    ]
    dirs: list[str] = []
    for root in roots:
        for relative in (
            "cu13/lib",
            "cuda_runtime/lib",
            "nvjitlink/lib",
            "cublas/lib",
            "cusparse/lib",
            "curand/lib",
        ):
            candidate = root / relative
            if candidate.is_dir():
                dirs.append(str(candidate))
    return dirs


def run_command(command: list[str], log_path: Path, timeout_seconds: int, cuda_visible_devices: str) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Reduces CUDA allocator fragmentation (the OOM errors on this pipeline's
    # long sequences suggest fragmented free memory, not just insufficient
    # total memory); no effect on CPU-only runs. Doesn't override an
    # explicit caller setting.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    extra_library_dirs = cuda_library_dirs()
    if extra_library_dirs:
        existing_ld_library_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([*extra_library_dirs, existing_ld_library_path]).strip(":")
    execution_device = str(cuda_visible_devices or "")
    if execution_device:
        env["CUDA_VISIBLE_DEVICES"] = "" if execution_device == "cpu" else execution_device
        env.pop("LOCAL_RANK", None)
        env.pop("RANK", None)
        env.pop("WORLD_SIZE", None)
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        log_path.write_text(
            "command=" + " ".join(command) + "\n"
            + f"execution_device={execution_device or os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
            + f"cuda_visible_devices={env.get('CUDA_VISIBLE_DEVICES', '')}\n"
            + f"extra_library_dirs={':'.join(extra_library_dirs)}\n"
            + f"returncode=timeout_after_{timeout_seconds}s\n"
            + "\n[stdout]\n" + stdout
            + "\n[stderr]\n" + stderr,
            encoding="utf-8",
        )
        raise RuntimeError(f"command timed out after {timeout_seconds}s; see {log_path}") from exc
    log_path.write_text(
        "command=" + " ".join(command) + "\n"
        + f"execution_device={execution_device or os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
        + f"cuda_visible_devices={env.get('CUDA_VISIBLE_DEVICES', '')}\n"
        + f"extra_library_dirs={':'.join(extra_library_dirs)}\n"
        + f"returncode={completed.returncode}\n"
        + "\n[stdout]\n" + completed.stdout
        + "\n[stderr]\n" + completed.stderr,
        encoding="utf-8",
    )
    return completed


def run_retriever_case(
    case: MatrixCase,
    csv_path: Path,
    prompt_mode: str,
    args: argparse.Namespace,
    *,
    use_adapter: bool = True,
    raw_suffix: str = "",
    family_override: str | None = None,
    adapter_dir_override: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    family = family_override or family_from_source_id(case.source_id)
    if family is None:
        raise RuntimeError(f"Cannot infer retriever family from source_id={case.source_id}")
    adapter_dir = (
        adapter_dir_override or adapter_dir_for(case.source_id)
        if use_adapter
        else None
    )
    if (
        use_adapter
        and adapter_dir_override is None
        and not is_finetuned_retriever_source(case.source_id)
    ):
        raise RuntimeError(f"source_id={case.source_id} is not a fine-tuned retriever experiment id")
    if use_adapter and adapter_dir is not None and not (
        adapter_dir / "adapter_config.json"
    ).is_file():
        raise FileNotFoundError(f"missing adapter_config.json: {adapter_dir}")

    raw_dir = args.pred_dir / "raw"
    raw_stem = f"{case.experiment_id}{raw_suffix}"
    log_path = raw_dir / f"{raw_stem}.log"
    if family == "flan":
        raw_output = raw_dir / f"{raw_stem}.txt"
        command = [
            sys.executable,
            str(RETRIEVER_ROOT / "Flan" / "lora_flan_large_finqa_rel_fact.py"),
            "--mode",
            "infer",
            "--train-csv",
            str(csv_path),
            "--eval-csv",
            str(csv_path),
            "--input-csv",
            str(csv_path),
            "--output-txt",
            str(raw_output),
            "--prompt-mode",
            prompt_mode,
            "--max-infer-samples",
            str(args.limit if args.limit > 0 else -1),
            "--batch-size",
            str(args.batch_size),
            "--max-new-tokens",
            str(args.max_tokens),
            "--max-length",
            str(args.max_input_tokens),
            "--attention-query-chunk-size",
            str(getattr(args, "attention_query_chunk_size", 0)),
            "--structured-output",
            str(args.structured_output),
        ]
        # Force "generate" unconditionally: with adapter+prompt_mode="original",
        # "auto" resolves to "finder_logits_argmax" (keeps the full autograd
        # graph via model(**inputs, labels=labels), ~5x the RSS of .generate()
        # at long sequence lengths). No-op for every existing official prompt
        # mode, which already resolves to "generate" via "auto".
        command.extend(["--infer-method", "generate"])
        command.extend(["--adapter-dir", str(adapter_dir)] if use_adapter else ["--no-adapter"])
    elif family == "t5gemma2":
        raw_output = raw_dir / f"{raw_stem}.jsonl"
        command = [
            sys.executable,
            str(RETRIEVER_ROOT / "t5gemma-2" / "t5gemma-2_train.py"),
            "--mode",
            "infer",
            "--train-csv",
            str(csv_path),
            "--eval-csv",
            str(csv_path),
            "--input-csv",
            str(csv_path),
            "--output-dir",
            str(REPO_ROOT / "Experiment" / case.source_id / "retriever"),
            "--output-txt",
            str(raw_output),
            "--prompt-mode",
            prompt_mode,
            "--max-infer-samples",
            str(args.limit if args.limit > 0 else -1),
            "--batch-size",
            str(args.batch_size),
            "--max-new-tokens",
            str(args.max_tokens),
            "--output-format",
            "jsonl",
            "--max-length",
            str(args.max_input_tokens),
            "--structured-output",
            str(args.structured_output),
            "--cache-safe-input-tokens",
            str(args.t5gemma_cache_safe_input_tokens),
        ]
        command.extend(["--adapter-dir", str(adapter_dir)] if use_adapter else ["--no-adapter"])
    else:
        raw_output = raw_dir / f"{raw_stem}.txt"
        command = [
            sys.executable,
            str(RETRIEVER_ROOT / "Mistral" / "mistral_inference.py"),
            "--input-csv",
            str(csv_path),
            "--output-txt",
            str(raw_output),
            "--prompt-mode",
            prompt_mode,
            "--max-infer-samples",
            str(args.limit if args.limit > 0 else -1),
            "--batch-size",
            str(args.batch_size),
            "--max-new-tokens",
            str(args.max_tokens),
            "--max-input-length",
            str(args.max_input_tokens),
            "--structured-output",
            str(args.structured_output),
        ]
        append_label_descriptions = getattr(
            args, "append_label_descriptions", None
        )
        if append_label_descriptions is not None:
            command.extend([
                "--append-label-descriptions",
                "true" if append_label_descriptions else "false",
            ])
        command.extend(["--adapter-dir", str(adapter_dir)] if use_adapter else ["--no-adapter"])
    completed = run_command(command, log_path, args.case_timeout_seconds, args.cuda_visible_devices)
    if completed.returncode != 0:
        raise RuntimeError(f"retriever inference failed for {case.experiment_id}; see {log_path}")
    row_checkpoint = Path(str(raw_output) + ".checkpoint.jsonl")
    return read_raw_prediction_lines(raw_output), {
        "family": family,
        "actual_engine": str(adapter_dir) if use_adapter else case.source_id,
        "prediction_contract": "retfact_retriever",
        "adapter_dir": str(adapter_dir) if use_adapter else None,
        "use_adapter": use_adapter,
        "raw_output": str(raw_output),
        "log": str(log_path),
        "row_checkpoint": str(row_checkpoint),
        "row_checkpoint_rows": (
            sum(
                1
                for line in row_checkpoint.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if row_checkpoint.is_file()
            else 0
        ),
        "cuda_visible_devices": args.cuda_visible_devices,
        "execution_device": "cpu" if args.cuda_visible_devices == "cpu" else "cuda",
        "structured_output": str(args.structured_output),
        "max_input_tokens": int(args.max_input_tokens),
        "max_new_tokens": int(args.max_tokens),
        "attention_query_chunk_size": int(
            getattr(args, "attention_query_chunk_size", 0)
        ),
        "run_seed": (
            int(os.environ["EXPERIMENT6_RUN_SEED"])
            if os.environ.get("EXPERIMENT6_RUN_SEED")
            else None
        ),
    }


def normalize_no_adapter_engine(source_id: str) -> str:
    family = family_from_source_id(source_id)
    if family in FAMILY_BASE_ENGINE:
        return FAMILY_BASE_ENGINE[family]
    return source_id


def compact_for_prompt(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def row_data_text(row: dict[str, str]) -> tuple[str, str]:
    data = compact_for_prompt(
        row.get("Narrative_Data")
        or row.get("data")
        or row.get("Tables")
        or row.get("Table_Text")
        or ""
    )
    text = compact_for_prompt(
        row.get("Narrative_Text")
        or row.get("text")
        or row.get("Sentence")
        or row.get("Pre_Text")
        or row.get("Post_Text")
        or ""
    )
    return data, text


def direct_binding_prompt(row: dict[str, str]) -> str:
    base = compact_for_prompt(row.get("input") or "")
    if "Return only strict JSON" in base or "## Output contract" in base:
        return base
    return f"{base}{DIRECT_BINDING_SUFFIX}"


def retfact_from_prediction(raw_prediction: str) -> str:
    text = extract_pred_text(raw_prediction)
    try:
        payload = json.loads(text)
    except Exception:
        return text
    if isinstance(payload, dict):
        value = payload.get("RetFact") or payload.get("retfact") or payload.get("Rel_Fact")
        if value not in (None, ""):
            return compact_for_prompt(value)
    return text


def converter_binding_prompt(row: dict[str, str], retriever_prediction: str) -> str:
    data, narrative_text = row_data_text(row)
    candidate_retfact = retfact_from_prediction(retriever_prediction)
    return "\n\n".join(
        [
            RETRIEVER_TO_BINDING_INSTRUCTION,
            f"Source: {row.get('Source') or ''}",
            "Chart data:\n" + (data or "[missing]"),
            "Narrative text:\n" + (narrative_text or "[missing]"),
            "Candidate RetFact / retriever output:\n" + (candidate_retfact or "[empty]"),
        ]
    )


def resolve_generator_runtime(engine: str) -> tuple[Any, Any, str]:
    sys.path.insert(0, str(REPO_ROOT))
    import new_full_finqa_run as generator_runtime  # noqa: WPS433

    config = generator_runtime.resolve_engine(engine, credential_purpose="execute")
    route_status = generator_runtime.route_execution_status(config)
    if not config.available:
        raise RuntimeError(f"engine {engine} is not executable: {route_status}; missing={config.missing_credentials}")
    return generator_runtime, config, route_status


def generate_binding_predictions_with_engine(
    *,
    case: MatrixCase,
    prompt_rows: list[dict[str, str]],
    prompts: list[str],
    engine: str,
    args: argparse.Namespace,
    raw_path: Path,
    stage: str,
    max_tokens: int,
    row_timeout_seconds: int,
    retriever_predictions: list[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    generator_runtime, config, route_status = resolve_generator_runtime(engine)
    public_config = config.to_public_dict()
    execution_provenance: dict[str, Any] = {
        "bindingGeneratorParallelism": int(args.binding_generator_parallelism),
    }
    if int(args.binding_generator_parallelism) > 1:
        execution_provenance["parallelProcessStartMethod"] = "fork"
    service_max_num_seqs = os.environ.get("VLLM_MAX_NUM_SEQS")
    if service_max_num_seqs:
        execution_provenance["serviceMaxNumSeqs"] = int(service_max_num_seqs)
    runtime_profile = (
        os.environ.get("VLLM_RUNTIME_PROFILE")
        or os.environ.get("SGLANG_RUNTIME_PROFILE")
        or public_config.get("runtime_profile")
    )
    if runtime_profile:
        execution_provenance["runtimeProfile"] = runtime_profile
        public_config["runtime_profile"] = runtime_profile
    model_path_environment = {
        "llama4": "LLAMA4_MODEL_PATH",
        "qwen3_6": "QWEN3_6_MODEL",
        "mistral4": "MISTRAL_SMALL_MODEL",
    }.get(engine)
    actual_model = (
        os.environ.get(model_path_environment)
        if model_path_environment
        else None
    ) or public_config.get("actual_model")
    if actual_model:
        execution_provenance["actualModel"] = actual_model
        public_config["actual_model"] = actual_model
    quantization = (
        os.environ.get("VLLM_QUANTIZATION")
        or os.environ.get("SGLANG_QUANTIZATION")
        or os.environ.get("LLAMA_CPP_QUANT")
    )
    if quantization:
        execution_provenance["quantization"] = quantization
        public_config["quantization"] = quantization
    runtime_info = {
        "stage": stage,
        "engine": engine,
        "actual_engine": engine,
        "prediction_contract": "data_binding_generator",
        "route_status": route_status,
        "raw_output": str(raw_path),
        "system_prompt": "experiment6_binding_json_v1",
        "max_tokens": max_tokens,
        "row_timeout_seconds": row_timeout_seconds,
        "parallelism": args.binding_generator_parallelism,
        "execution": execution_provenance,
        "config": public_config,
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any] | None] = [None] * len(prompt_rows)
    predictions: list[str] = [""] * len(prompt_rows)

    def same_row_timeout_checkpoint(existing: dict[str, Any]) -> bool:
        if not str(existing.get("status", "")).startswith("runtime_blocked"):
            return False
        error = existing.get("error")
        if not isinstance(error, dict) or error.get("category") != "row_timeout":
            return False
        previous_timeout = error.get("row_timeout_seconds")
        if previous_timeout is None:
            match = re.search(r"timed out after (\d+)s", str(error.get("message", "")))
            previous_timeout = int(match.group(1)) if match else None
        try:
            return int(previous_timeout) == int(row_timeout_seconds)
        except (TypeError, ValueError):
            return False

    if raw_path.is_file():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(existing, dict):
                continue
            try:
                existing_index = int(existing.get("index"))
            except (TypeError, ValueError):
                continue
            if not 0 <= existing_index < len(prompt_rows):
                continue
            if existing.get("status") == "completed" and existing.get("prediction"):
                raw_rows[existing_index] = existing
                predictions[existing_index] = str(existing.get("prediction") or "")
            elif same_row_timeout_checkpoint(existing):
                raw_rows[existing_index] = existing

    def flush_raw_rows() -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=raw_path.parent,
                prefix=f".{raw_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                for index, raw_row in enumerate(raw_rows):
                    if raw_row is None:
                        raw_row = build_raw_row(
                            index,
                            "",
                            "not_started",
                            {"category": "not_started_after_runtime_block"},
                        )
                    handle.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, raw_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def build_raw_row(
        index: int,
        prediction: str,
        status: str,
        error: dict[str, Any] | None,
        response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = prompt_rows[index]
        return {
            "index": index,
            "case_id": row.get("Source"),
            "stage": stage,
            "status": status,
            "input_prompt": prompts[index],
            "retriever_prediction": retriever_predictions[index] if retriever_predictions and index < len(retriever_predictions) else None,
            "prediction": prediction,
            "response": response,
            "execution": (
                None if status == "not_started" else execution_provenance
            ),
            "error": error,
        }

    def attach_execution_metadata(response: Any) -> dict[str, Any]:
        response_metadata = dict(response or {})
        response_metadata["executionBatch"] = {
            key: value
            for key, value in execution_provenance.items()
            if key in {
                "bindingGeneratorParallelism",
                "parallelProcessStartMethod",
                "serviceMaxNumSeqs",
            }
        }
        if runtime_profile:
            response_metadata["runtimeProfile"] = runtime_profile
        return response_metadata

    def generate_one(index: int) -> tuple[int, str, dict[str, Any]]:
        prompt = prompts[index]
        outputs, response = generate_text_with_timeout(
            generator_runtime,
            config,
            prompt,
            BINDING_JSON_SYSTEM_PROMPT,
            max_tokens,
            row_timeout_seconds,
            return_metadata=True,
        )
        require_response_model_identity(engine, config, response)
        return (
            index,
            outputs[0] if outputs else "",
            attach_execution_metadata(response),
        )

    first_error: dict[str, Any] | None = next(
        (
            dict(row["error"])
            for row in raw_rows
            if isinstance(row, dict)
            and str(row.get("status", "")).startswith("runtime_blocked")
            and isinstance(row.get("error"), dict)
        ),
        None,
    )
    completed = sum(1 for row in raw_rows if isinstance(row, dict) and row.get("status") == "completed" and row.get("prediction"))
    pending_indexes = [index for index, row in enumerate(raw_rows) if row is None]
    if args.binding_generator_parallelism > 1 and pending_indexes:
        runtime_module_name = getattr(generator_runtime, "__name__", "")
        if not runtime_module_name:
            raise RuntimeError(
                "parallel row generation requires an importable runtime module"
            )
        process_context = mp.get_context("fork")
        waiting = deque(pending_indexes)
        active: dict[int, tuple[Any, Any, float]] = {}

        def stop_process(process: Any) -> None:
            if process.is_alive():
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)

        def record_parallel_error(
            index: int,
            exc: BaseException,
            traceback_tail: str = "",
        ) -> None:
            nonlocal first_error
            error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "category": (
                    "row_timeout"
                    if isinstance(exc, RowTimeoutError)
                    else generation_failure_category(generator_runtime, exc)
                ),
                "row_timeout_seconds": (
                    row_timeout_seconds if isinstance(exc, RowTimeoutError) else None
                ),
                "traceback_tail": traceback_tail or traceback.format_exc()[-2000:],
            }
            raw_rows[index] = build_raw_row(index, "", "runtime_blocked", error)
            flush_raw_rows()
            first_error = first_error or error

        while waiting or active:
            while waiting and len(active) < args.binding_generator_parallelism:
                index = waiting.popleft()
                result_queue = process_context.Queue(maxsize=1)
                process = process_context.Process(
                    target=_parallel_generation_target,
                    args=(
                        result_queue,
                        runtime_module_name,
                        config,
                        prompts[index],
                        BINDING_JSON_SYSTEM_PROMPT,
                        max_tokens,
                    ),
                )
                process.start()
                active[index] = (process, result_queue, time.monotonic())

            progressed = False
            now = time.monotonic()
            for index, (process, result_queue, started_at) in list(active.items()):
                payload: dict[str, Any] | None = None
                try:
                    payload = result_queue.get_nowait()
                except queue_module.Empty:
                    if not process.is_alive():
                        try:
                            payload = result_queue.get(timeout=1)
                        except queue_module.Empty:
                            record_parallel_error(
                                index,
                                RuntimeError(
                                    "generator row exited without output; "
                                    f"exitcode={process.exitcode}"
                                ),
                            )
                    elif (
                        row_timeout_seconds > 0
                        and now - started_at >= row_timeout_seconds
                    ):
                        stop_process(process)
                        record_parallel_error(
                            index,
                            RowTimeoutError(
                                f"generator row timed out after {row_timeout_seconds}s"
                            ),
                        )
                    else:
                        continue

                if payload is not None:
                    process.join(5)
                    if payload.get("ok"):
                        outputs = payload.get("outputs") or []
                        prediction = outputs[0] if outputs else ""
                        response = attach_execution_metadata(
                            payload.get("metadata") or {}
                        )
                        try:
                            require_response_model_identity(engine, config, response)
                        except ResponseModelIdentityError as exc:
                            record_parallel_error(index, exc)
                            stop_process(process)
                            result_queue.close()
                            del active[index]
                            progressed = True
                            continue
                        predictions[index] = prediction
                        raw_rows[index] = build_raw_row(
                            index, prediction, "completed", None, response,
                        )
                        completed += 1
                        flush_raw_rows()
                    else:
                        record_parallel_error(
                            index,
                            RuntimeError(
                                "child generation failed: "
                                f"{payload.get('type')}: {payload.get('message')}"
                            ),
                            str(payload.get("traceback_tail") or ""),
                        )
                stop_process(process)
                result_queue.close()
                del active[index]
                progressed = True

            if not progressed and active:
                time.sleep(0.05)
    elif pending_indexes:
        for index in pending_indexes:
            row = prompt_rows[index]
            try:
                _, prediction, response = generate_one(index)
                predictions[index] = prediction
                raw_rows[index] = build_raw_row(
                    index, prediction, "completed", None, response,
                )
                completed += 1
                flush_raw_rows()
            except Exception as exc:  # pragma: no cover - runtime evidence path
                error = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "category": "row_timeout" if isinstance(exc, RowTimeoutError) else generation_failure_category(generator_runtime, exc),
                    "row_timeout_seconds": (
                        row_timeout_seconds if isinstance(exc, RowTimeoutError) else None
                    ),
                    "traceback_tail": traceback.format_exc()[-2000:],
                }
                raw_rows[index] = build_raw_row(index, "", "runtime_blocked", error)
                flush_raw_rows()
                first_error = first_error or error
                continue

    flush_raw_rows()
    processed = sum(raw_row is not None for raw_row in raw_rows)
    runtime_blocked = sum(
        1
        for raw_row in raw_rows
        if isinstance(raw_row, dict)
        and str(raw_row.get("status", "")).startswith("runtime_blocked")
    )
    if processed != len(prompt_rows):
        failure_category = (first_error or {}).get("category") or "blocked_model_generation_runtime"
        raise BindingGenerationRuntimeError(
            f"engine {engine} stopped after processing {processed}/{len(prompt_rows)} rows "
            f"during {stage}; see {raw_path}",
            runtime={
                **runtime_info,
                "status": "runtime_blocked",
                "completed_rows": completed,
                "runtime_blocked_rows": runtime_blocked,
                "processed_rows": processed,
                "expected_rows": len(prompt_rows),
                "error": first_error,
            },
            failure_category=failure_category,
        )
    return predictions, {
        **runtime_info,
        "status": "runtime_blocked" if runtime_blocked else "completed",
        "completed_rows": completed,
        "runtime_blocked_rows": runtime_blocked,
        "processed_rows": processed,
        "expected_rows": len(prompt_rows),
        "error": first_error,
    }


def run_no_adapter_case(
    case: MatrixCase,
    csv_path: Path,
    prompt_mode: str,
    prompt_rows: list[dict[str, str]],
    args: argparse.Namespace,
    *,
    raw_suffix: str = "",
) -> tuple[list[str], dict[str, Any]]:
    if is_finetuned_retriever_source(case.source_id):
        raise RuntimeError(
            f"fine-tuned retriever source_id={case.source_id} cannot run in no-adapter mode; "
            "use Experiment 6 mixed or retriever mode"
        )
    if family_from_source_id(case.source_id) is not None:
        return run_retriever_case(case, csv_path, prompt_mode, args, use_adapter=False, raw_suffix=raw_suffix)

    engine = normalize_no_adapter_engine(case.source_id)
    raw_path = args.pred_dir / "raw" / f"{case.experiment_id}{raw_suffix}.jsonl"
    prompts = [direct_binding_prompt(row) for row in prompt_rows]
    return generate_binding_predictions_with_engine(
        case=case,
        prompt_rows=prompt_rows,
        prompts=prompts,
        engine=engine,
        args=args,
        raw_path=raw_path,
        stage="direct_binding_generation",
        max_tokens=args.max_tokens,
        row_timeout_seconds=args.row_timeout_seconds,
    )


def run_binding_converter_case(
    case: MatrixCase,
    prompt_rows: list[dict[str, str]],
    retriever_predictions: list[str],
    retriever_runtime: dict[str, Any],
    args: argparse.Namespace,
    *,
    raw_suffix: str = "",
) -> tuple[list[str], dict[str, Any]]:
    if not args.binding_conversion:
        raise MissingBindingGeneratorError(
            "runtime_blocked_missing_binding_generator: RetFact retriever output requires a formal "
            "RetFact-to-Binding conversion stage before Experiment 6 F-measure",
            runtime=retriever_runtime,
        )
    engine = args.binding_converter_engine
    raw_path = args.pred_dir / "raw" / f"{case.experiment_id}{raw_suffix}.binding_converter.jsonl"
    prompts = [converter_binding_prompt(row, retriever_predictions[index]) for index, row in enumerate(prompt_rows)]
    predictions, converter_runtime = generate_binding_predictions_with_engine(
        case=case,
        prompt_rows=prompt_rows,
        prompts=prompts,
        engine=engine,
        args=args,
        raw_path=raw_path,
        stage="retfact_to_binding_conversion",
        max_tokens=args.binding_converter_max_tokens,
        row_timeout_seconds=args.binding_converter_row_timeout_seconds,
        retriever_predictions=retriever_predictions,
    )
    runtime = {
        "prediction_contract": "data_binding_generator",
        "binding_conversion": {
            "status": "completed",
            "converter_engine": engine,
            "converter_raw_output": str(raw_path),
        },
        "stages": [retriever_runtime, converter_runtime],
    }
    return predictions, runtime


def aggregate_top_k_rows(prompt_rows: list[dict[str, str]], run_rows: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    if not run_rows:
        return []
    limit = max(1, min(top_k, len(run_rows)))
    aggregated: list[dict[str, Any]] = []
    for index, prompt_row in enumerate(prompt_rows):
        case_id = str(prompt_row.get("Source") or f"row_{index + 1}").strip()
        seen: set[tuple[str, str]] = set()
        items: list[dict[str, str]] = []
        for rows in run_rows[:limit]:
            if index >= len(rows):
                continue
            for item in rows[index].get("items", []):
                key = (str(item.get("type") or ""), str(item.get("text") or ""))
                if key[0] and key[1] and key not in seen:
                    seen.add(key)
                    items.append({"type": key[0], "text": key[1]})
        aggregated.append({"case_id": case_id, "items": items})
    return aggregated


def run_pred_path(args: argparse.Namespace, case: MatrixCase, run_number: int) -> Path:
    if args.num_runs <= 1:
        return args.pred_dir / f"{case.experiment_id}.jsonl"
    return args.pred_dir / "runs" / f"run_{run_number:02d}" / f"{case.experiment_id}.jsonl"


def run_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def completed_run_payload(path: Path, prompt_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    metadata_path = run_metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = read_prediction_jsonl(path)
    except Exception:
        return None
    if metadata.get("status") != "completed":
        return None
    if int(metadata.get("rows", -1)) != len(prompt_rows):
        return None
    if len(rows) != len(prompt_rows):
        return None
    return rows, metadata


def runtime_sequence(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    def flatten(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            items: list[dict[str, Any]] = []
            for item in value:
                items.extend(flatten(item))
            return items
        if not isinstance(value, dict):
            return []
        items = [value]
        if isinstance(value.get("stages"), list):
            for stage in value["stages"]:
                items.extend(flatten(stage))
        return items

    return flatten(metadata.get("runtime"))


def metadata_uses_adapter(metadata: dict[str, Any]) -> bool:
    return any(runtime.get("use_adapter") is True for runtime in runtime_sequence(metadata))


def metadata_has_retfact_only_contract(metadata: dict[str, Any]) -> bool:
    conversion = metadata.get("binding_conversion") or {}
    if isinstance(conversion, dict) and conversion.get("status") == "completed":
        return False
    for runtime in runtime_sequence(metadata):
        runtime_conversion = runtime.get("binding_conversion") or {}
        if isinstance(runtime_conversion, dict) and runtime_conversion.get("status") == "completed":
            return False
    if metadata.get("prediction_contract") == "retfact_retriever_without_binding_conversion":
        return True
    if metadata.get("generation_mode") == "retriever" and not metadata.get("binding_conversion"):
        return True
    return any(
        runtime.get("prediction_contract") == "retfact_retriever"
        or runtime.get("family") in FAMILY_BASE_ENGINE
        for runtime in runtime_sequence(metadata)
    )


def completed_run_metadata_is_usable(metadata: dict[str, Any], case: MatrixCase) -> bool:
    if is_finetuned_retriever_source(case.source_id) and not metadata_uses_adapter(metadata):
        return False
    if metadata_has_retfact_only_contract(metadata):
        return False
    return True


def completed_metadata_is_usable(metadata: dict[str, Any], case: MatrixCase, args: argparse.Namespace) -> bool:
    if not metadata.get("formal_result") or metadata.get("runtime_blocked"):
        return False
    if int(metadata.get("num_runs", 1)) != args.num_runs:
        return False
    if int(metadata.get("top_k", 3)) != args.top_k:
        return False
    run_paths = [Path(path) for path in metadata.get("run_prediction_jsonls", [])]
    if len(run_paths) != args.num_runs:
        return False
    if any((not path.is_file()) or path.stat().st_size <= 0 for path in run_paths):
        return False
    if is_finetuned_retriever_source(case.source_id) and not metadata_uses_adapter(metadata):
        return False
    if metadata_has_retfact_only_contract(metadata):
        return False
    return True


def require_binding_generation_contract(case: MatrixCase, rows: list[dict[str, Any]], runtime: dict[str, Any]) -> None:
    if runtime.get("prediction_contract") != "retfact_retriever":
        return
    raise MissingBindingGeneratorError(
        "runtime_blocked_missing_binding_generator: RetFact retriever output requires a formal "
        "RetFact-to-Binding conversion stage before Experiment 6 F-measure; refusing to score "
        "the empty canonical Binding wrapper as a data-binding prediction",
        runtime={
            **runtime,
            "binding_conversion": {
                "status": "missing",
                "required": True,
                "reason": "retfact_retriever_output_is_not_data_binding_prediction",
            },
            "extracted_items": sum(len(row.get("items") or []) for row in rows),
        },
    )


def run_generation_once(
    case: MatrixCase,
    csv_path: Path,
    prompt_mode: str,
    prompt_rows: list[dict[str, str]],
    args: argparse.Namespace,
    run_number: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    raw_suffix = "" if args.num_runs <= 1 else f".run_{run_number:02d}"
    if args.mode == "retriever":
        prediction_texts, runtime = run_retriever_case(case, csv_path, prompt_mode, args, raw_suffix=raw_suffix)
    elif args.mode == "mixed" and is_finetuned_retriever_source(case.source_id):
        prediction_texts, runtime = run_retriever_case(case, csv_path, prompt_mode, args, raw_suffix=raw_suffix)
    else:
        prediction_texts, runtime = run_no_adapter_case(case, csv_path, prompt_mode, prompt_rows, args, raw_suffix=raw_suffix)
    if len(prediction_texts) != len(prompt_rows):
        raise RuntimeError(f"prediction row count mismatch: prompts={len(prompt_rows)} predictions={len(prediction_texts)}")
    if runtime.get("prediction_contract") == "retfact_retriever":
        prediction_texts, runtime = run_binding_converter_case(
            case,
            prompt_rows,
            prediction_texts,
            runtime,
            args,
            raw_suffix=raw_suffix,
        )
        if len(prediction_texts) != len(prompt_rows):
            raise RuntimeError(f"binding conversion row count mismatch: prompts={len(prompt_rows)} predictions={len(prediction_texts)}")
    rows, extraction_reports = prediction_rows_from_texts(case, prompt_rows, prediction_texts)
    require_binding_generation_contract(case, rows, runtime)
    return rows, runtime, extraction_reports


def run_generation_with_retry(
    case: MatrixCase,
    csv_path: Path,
    prompt_mode: str,
    prompt_rows: list[dict[str, str]],
    args: argparse.Namespace,
    run_number: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    attempts = max(0, args.retry_max) + 1
    wait_seconds = rounded_wait_seconds(args.retry_wait_seconds)
    for attempt in range(1, attempts + 1):
        try:
            rows, runtime, extraction_reports = run_generation_once(case, csv_path, prompt_mode, prompt_rows, args, run_number)
            runtime["attempt"] = attempt
            return rows, runtime, extraction_reports
        except Exception as exc:
            transient = is_transient_error(exc)
            if args.debug:
                write_debug_error(args, case, run_number, attempt, exc, transient=transient, wait_seconds=wait_seconds)
            should_retry = transient and attempt < attempts
            if not should_retry:
                raise
            print(
                json.dumps(
                    {
                        "time": utc_now(),
                        "status": "retry_wait",
                        "experiment_id": case.experiment_id,
                        "run": run_number,
                        "attempt": attempt,
                        "wait_seconds": wait_seconds,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)
    raise RuntimeError(f"unreachable retry loop for {case.experiment_id} run {run_number}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    variables_md_status = load_workspace_variables_md(args)
    cases = parse_matrix(args.matrix)
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    case_reports: list[dict[str, Any]] = []
    failures = 0
    for case in cases:
        if case.narrative_route not in ROUTE_CSV_PATHS:
            raise ValueError(f"Unsupported narrative_route={case.narrative_route}")
        csv_path = ROUTE_CSV_PATHS[case.narrative_route]
        prompt_rows = read_prompt_rows(csv_path, args.limit)
        pred_path = args.pred_dir / f"{case.experiment_id}.jsonl"
        metadata_path = pred_path.with_suffix(pred_path.suffix + ".metadata.json")
        prompt_mode = ROUTE_PROMPT_MODE[case.narrative_route]
        if args.resume_completed and pred_path.is_file() and metadata_path.is_file():
            try:
                existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_metadata = {}
            if completed_metadata_is_usable(existing_metadata, case, args):
                case_reports.append(
                    {
                        "experiment_id": case.experiment_id,
                        "source_id": case.source_id,
                        "narrative_route": case.narrative_route,
                        "prediction_jsonl": str(pred_path),
                        "metadata_json": str(metadata_path),
                        "status": "resumed_completed",
                    }
                )
                continue
        try:
            run_rows: list[list[dict[str, Any]]] = []
            run_reports: list[dict[str, Any]] = []
            run_prediction_jsonls: list[str] = []
            runtimes: list[dict[str, Any]] = []
            for run_number in range(1, args.num_runs + 1):
                current_pred_path = run_pred_path(args, case, run_number)
                existing_run = completed_run_payload(current_pred_path, prompt_rows) if args.resume_runs else None
                if existing_run is not None:
                    rows, run_metadata = existing_run
                    if not completed_run_metadata_is_usable(run_metadata, case):
                        existing_run = None
                if existing_run is not None:
                    rows, run_metadata = existing_run
                    extraction_reports = run_metadata.get("extraction_reports") or []
                    runtime = {"resumed": True, **(run_metadata.get("runtime") or {})}
                else:
                    rows, runtime, extraction_reports = run_generation_with_retry(
                        case,
                        csv_path,
                        prompt_mode,
                        prompt_rows,
                        args,
                        run_number,
                    )
                write_prediction_jsonl(current_pred_path, rows)
                write_json(
                    run_metadata_path(current_pred_path),
                    {
                        "created_at": utc_now(),
                        "status": "completed",
                        "formal_result": True,
                        "experiment_id": case.experiment_id,
                        "source_id": case.source_id,
                        "narrative_route": case.narrative_route,
                        "run": run_number,
                        "rows": len(rows),
                        "runtime": runtime,
                        "binding_conversion": runtime.get("binding_conversion") if isinstance(runtime, dict) else None,
                        "extraction_reports": extraction_reports,
                    },
                )
                run_rows.append(rows)
                run_reports.append({"run": run_number, "extraction_reports": extraction_reports})
                run_prediction_jsonls.append(str(current_pred_path))
                runtimes.append({"run": run_number, **runtime})
            if args.num_runs > 1:
                top_rows = aggregate_top_k_rows(prompt_rows, run_rows, args.top_k)
                write_prediction_jsonl(pred_path, top_rows)
                rows = top_rows
                extraction_reports = [report for run_report in run_reports for report in run_report["extraction_reports"]]
            else:
                rows = run_rows[0]
                extraction_reports = run_reports[0]["extraction_reports"]
            metadata = {
                "created_at": utc_now(),
                "formal_result": True,
                "controlled_smoke": False,
                "prediction_source": "model_generated_from_input_prompt",
                "generation_mode": args.mode,
                "experiment_id": case.experiment_id,
                "source_id": case.source_id,
                "narrative_route": case.narrative_route,
                "prompt_csv": str(csv_path),
                "rows": len(rows),
                "num_runs": args.num_runs,
                "top_k": args.top_k,
                "top_k_prediction_jsonl": str(pred_path),
                "run_prediction_jsonls": run_prediction_jsonls,
                "runtime": runtimes[0] if len(runtimes) == 1 else runtimes,
                "binding_conversion": (
                    runtimes[0].get("binding_conversion")
                    if len(runtimes) == 1 and isinstance(runtimes[0], dict)
                    else {"status": "completed"} if any(isinstance(item, dict) and item.get("binding_conversion", {}).get("status") == "completed" for item in runtimes)
                    else None
                ),
                "extraction_reports": extraction_reports,
                "resume_runs": args.resume_runs,
                "retry_max": args.retry_max,
                "retry_wait_seconds": rounded_wait_seconds(args.retry_wait_seconds),
                "debug": args.debug,
                "debug_dir": str(args.pred_dir / "debug" / case.experiment_id) if args.debug else None,
            }
            status = "completed"
        except Exception as exc:
            failures += 1
            empty_rows = [{"case_id": str(row.get("Source") or f"row_{index + 1}"), "items": []} for index, row in enumerate(prompt_rows)]
            write_prediction_jsonl(pred_path, empty_rows)
            runtime = getattr(exc, "runtime", None)
            failure_category = getattr(exc, "failure_category", "blocked_model_generation_runtime")
            metadata = {
                "created_at": utc_now(),
                "formal_result": False,
                "controlled_smoke": False,
                "prediction_source": "runtime_blocked_no_binding_result_fallback",
                "generation_mode": args.mode,
                "experiment_id": case.experiment_id,
                "source_id": case.source_id,
                "narrative_route": case.narrative_route,
                "prompt_csv": str(csv_path),
                "rows": len(empty_rows),
                "runtime_blocked": True,
                "failure_category": failure_category,
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "runtime": runtime,
                "resume_command": resume_command(args, case),
                "env_status": public_env_status(),
                "debug_dir": str(args.pred_dir / "debug" / case.experiment_id) if args.debug else None,
            }
            status = "runtime_blocked"
        write_json(metadata_path, metadata)
        case_reports.append(
            {
                "experiment_id": case.experiment_id,
                "source_id": case.source_id,
                "narrative_route": case.narrative_route,
                "prediction_jsonl": str(pred_path),
                "metadata_json": str(metadata_path),
                "status": status,
            }
        )
    report = {
        "time": utc_now(),
        "mode": args.mode,
        "pred_dir": str(args.pred_dir),
        "matrix": args.matrix.split(),
        "cases": case_reports,
        "failures": failures,
        "status": "completed" if failures == 0 else "completed_with_runtime_blocks",
        "variables_md_status": variables_md_status,
    }
    write_json(args.report_json or (args.pred_dir / "generation_report.json"), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retriever", "no-adapter", "mixed"], required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--case-timeout-seconds", type=int, default=0)
    parser.add_argument("--row-timeout-seconds", type=int, default=int(os.environ.get("EXPERIMENT6_ROW_TIMEOUT_SECONDS", "0") or "0"))
    parser.add_argument("--num-runs", type=int, default=int(os.environ.get("EXPERIMENT6_NUM_RUNS", "1") or "1"))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("EXPERIMENT6_TOP_K", "3") or "3"))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("EXPERIMENT6_CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--resume-completed", action="store_true", default=True)
    parser.add_argument("--resume-runs", type=int, default=int(os.environ.get("EXPERIMENT6_RESUME_RUNS", "1") or "1"))
    parser.add_argument("--retry-max", type=int, default=int(os.environ.get("EXPERIMENT6_RETRY_MAX", "0") or "0"))
    parser.add_argument("--retry-wait-seconds", type=int, default=int(os.environ.get("EXPERIMENT6_RETRY_WAIT_SECONDS", "600") or "600"))
    parser.add_argument("--binding-conversion", type=int, default=int(os.environ.get("EXPERIMENT6_BINDING_CONVERSION", "1") or "1"))
    parser.add_argument("--binding-converter-engine", default=os.environ.get("EXPERIMENT6_BINDING_CONVERTER_ENGINE", "gpt5_5"))
    parser.add_argument(
        "--binding-converter-max-tokens",
        type=int,
        default=int(os.environ.get("EXPERIMENT6_BINDING_CONVERTER_MAX_TOKENS", "1024") or "1024"),
    )
    parser.add_argument(
        "--binding-converter-row-timeout-seconds",
        type=int,
        default=int(os.environ.get("EXPERIMENT6_BINDING_CONVERTER_ROW_TIMEOUT_SECONDS", os.environ.get("EXPERIMENT6_ROW_TIMEOUT_SECONDS", "120")) or "120"),
    )
    parser.add_argument(
        "--binding-generator-parallelism",
        type=int,
        default=int(os.environ.get("EXPERIMENT6_BINDING_GENERATOR_PARALLELISM", "1") or "1"),
    )
    parser.add_argument(
        "--binding-generator-total-timeout-seconds",
        type=int,
        default=int(os.environ.get("EXPERIMENT6_BINDING_GENERATOR_TOTAL_TIMEOUT_SECONDS", "0") or "0"),
    )
    parser.add_argument(
        "--load-variables-md",
        type=int,
        default=int(os.environ.get("EXPERIMENT6_LOAD_VARIABLES_MD", "1") or "1"),
    )
    parser.add_argument(
        "--variables-md",
        type=Path,
        default=Path(os.environ.get("EXPERIMENT6_VARIABLES_MD", str(WORKSPACE_ROOT / "src" / "doc" / "workspace" / "variables.md"))),
    )
    parser.add_argument("--debug", type=int, default=int(os.environ.get("EXPERIMENT6_DEBUG", "0") or "0"))
    args = parser.parse_args()
    args.num_runs = max(1, args.num_runs)
    args.top_k = max(1, args.top_k)
    args.retry_max = max(0, args.retry_max)
    args.retry_wait_seconds = rounded_wait_seconds(max(0, args.retry_wait_seconds))
    args.binding_conversion = bool(args.binding_conversion)
    args.binding_converter_max_tokens = max(1, args.binding_converter_max_tokens)
    args.binding_converter_row_timeout_seconds = max(0, args.binding_converter_row_timeout_seconds)
    args.binding_generator_parallelism = max(1, args.binding_generator_parallelism)
    args.binding_generator_total_timeout_seconds = max(0, args.binding_generator_total_timeout_seconds)
    args.load_variables_md = bool(args.load_variables_md)
    args.resume_runs = bool(args.resume_runs)
    args.debug = bool(args.debug)
    return args


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
