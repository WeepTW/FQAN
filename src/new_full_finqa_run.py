"""Unified FinQA generator route for FINDER-style target computation.

The module centralizes LLM engine validation and loads credentials from
environment variables or git-ignored local files only. Execution is opt-in via
--execute; validation and prompt inspection are safe defaults for pipeline dry
runs.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any, Iterator

import func_timeout

from finqa_metrics import finqa_equal, floatify_ans


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_MODELS_ROOT = WORKSPACE_ROOT / "Models"
DEFAULT_CONDA_ENV = "fnqa"
DEFAULT_INPUT_JSON = (
    REPO_ROOT
    / "Experiment"
    / "finqa_flan_o"
    / "retriever"
    / "outputs"
    / "best_matched_with_retrieved_facts_and_questions.json"
)
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "Experiment" / "Final" / "Finqa" / "finqa_pipeline_generated.jsonl"
PERCENTAGE_EQUIVALENT_DIAGNOSTIC_NOTE = (
    "diagnostic only: finqa_equal(..., include_percentage=True) accepts "
    "reference/100, reference, and reference*100; formal FINDER EA uses "
    "include_percentage=False"
)
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


API_AUTH_ERROR_PATTERNS = (
    "401 unauthorized",
    "missing bearer",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "auth error",
)

API_QUOTA_ERROR_PATTERNS = (
    "insufficient_quota",
    "quota",
    "billing",
    "credit",
    "balance",
    "rate limit",
    "ratelimit",
    "too many requests",
    "resource exhausted",
    "http 429",
    "status code: 429",
)

ENGINE_ALIASES = {
    "deepseek": "deepseek_r1_qwen32b",
    "deepseek_r1_qwen32b": "deepseek_r1_qwen32b",
    "deepseek-r1-qwen32b": "deepseek_r1_qwen32b",
    "deepseek_r1_distill_qwen_32b": "deepseek_r1_qwen32b",
    "deepseek-r1-distill-qwen-32b": "deepseek_r1_qwen32b",
    "deepseek-ai/deepseek-r1-distill-qwen-32b": "deepseek_r1_qwen32b",
    "mistral4": "mistral4",
    "mistral": "mistral4",
    "mistral_small_4": "mistral4",
    "mistral-small-4": "mistral4",
    "qwen": "qwen3_6",
    "qwen3_6": "qwen3_6",
    "qwen3_6_35b_a3b_fp8": "qwen3_6",
    "qwen3.6-35b-a3b-fp8": "qwen3_6",
    "llama": "llama4",
    "llama3_3": "llama3_3",
    "llama-3.3": "llama3_3",
    "llama3.3": "llama3_3",
    "llama3_3_70b": "llama3_3",
    "llama-3.3-70b-instruct": "llama3_3",
    "meta-llama/llama-3.3-70b-instruct": "llama3_3",
    "llama4": "llama4",
    "llama-4": "llama4",
    "llama4_scout": "llama4",
    "llama4-scout": "llama4",
    "llama_4_scout": "llama4",
    "llama-4-scout-17b-16e-instruct": "llama4",
    "meta-llama/llama-4-scout-17b-16e-instruct": "llama4",
    "qwythos": "qwythos9b",
    "qwythos9b": "qwythos9b",
    "qwythos-9b": "qwythos9b",
    "qwythos_9b": "qwythos9b",
    "empero-ai/qwythos-9b-claude-mythos-5-1m": "qwythos9b",
    "gpt41": "gpt4_1",
    "gpt-4.1": "gpt4_1",
    "gpt4.1": "gpt4_1",
    "gpt4_1": "gpt4_1",
    "gpt-4": "gpt4_1",
    "gpt4": "gpt4_1",
    "gpt5_3_codex": "gpt5_3_codexS",
    "gpt5_3_codexs": "gpt5_3_codexS",
    "gpt5_3_codexS": "gpt5_3_codexS",
    "gpt53codex": "gpt5_3_codexS",
    "gpt53codexs": "gpt5_3_codexS",
    "gpt-5.3-codex-spark": "gpt5_3_codexS",
    "gpt55": "gpt5_5",
    "gpt-5.5": "gpt5_5",
    "gpt5_5": "gpt5_5",
}


FORMAL_GENERATOR_MODELS = {
    "deepseek_r1_qwen32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistral4": "mistralai/Mistral-Small-4-119B-2603-NVFP4",
    "qwen3_6": "Qwen/Qwen3.6-35B-A3B-FP8",
    "llama3_3": "meta-llama/Llama-3.3-70B-Instruct",
    "llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "qwythos9b": "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
    "gpt4_1": "gpt-4.1",
    "gpt5_3_codexS": "gpt-5.3-codex-spark",
    "gpt5_5": "gpt-5.5",
}

ROUTE_BACKENDS = {
    "local_vllm_openai_compatible": "vllm",
    "local_llama_cpp_openai_compatible": "llama_cpp",
    "local_sglang_openai_compatible": "sglang",
    "chatmock_openai_compatible": "chatmock",
    "closed_api_azure_openai": "azure",
    "closed_api_openai": "openai",
    "closed_api_openai_compatible": "openai_compatible",
    "codex_cli": "codex_cli",
}

CREDENTIAL_NAMES = [
    "MODELS_ROOT",
    "HF_HOME",
    "VLLM_BASE_URL",
    "VLLM_API_KEY",
    "VLLM_SERVED_MODEL_NAME",
    "MISTRAL_SMALL_MODEL",
    "MISTRAL_SMALL_MODEL_PATH",
    "MISTRAL_SMALL_RUNTIME_BACKEND",
    "MISTRAL_SMALL_LLAMA_CPP_FORMAL_MODEL",
    "LLAMA_CPP_BASE_URL",
    "LLAMA_CPP_API_KEY",
    "LLAMA_CPP_MODEL_PATH",
    "LLAMA_CPP_MODEL_ALIAS",
    "LLAMA_CPP_QUANT",
    "LLAMA_CPP_CTX_SIZE",
    "LLAMA_CPP_N_GPU_LAYERS",
    "LLAMA_CPP_TENSOR_SPLIT",
    "LLAMA_CPP_SPLIT_MODE",
    "LLAMA_CPP_PARALLEL",
    "LLAMA_CPP_BATCH_SIZE",
    "LLAMA_CPP_UBATCH_SIZE",
    "LLAMA_CPP_THREADS",
    "LLAMA_CPP_CACHE_TYPE_K",
    "LLAMA_CPP_CACHE_TYPE_V",
    "LLAMA_CPP_KV_OFFLOAD",
    "LLAMA_CPP_CPU_MOE",
    "LLAMA_CPP_N_CPU_MOE",
    "LLAMA_CPP_FIT_TARGET",
    "LLAMA_CPP_FIT_CTX",
    "LLAMA_CPP_OP_OFFLOAD",
    "LLAMA_CPP_FLASH_ATTN",
    "LLAMA_CPP_CACHE_RAM",
    "LLAMA_CPP_DEVICE",
    "LLAMA_CPP_MAIN_GPU",
    "LLAMA_CPP_NO_MMAP",
    "SGLANG_BASE_URL",
    "SGLANG_API_KEY",
    "SGLANG_MODEL_PATH",
    "SGLANG_SERVED_MODEL_NAME",
    "SGLANG_FORMAL_MODEL",
    "SGLANG_RUNTIME_PROFILE",
    "SGLANG_QUANTIZATION",
    "SGLANG_LOAD_FORMAT",
    "SGLANG_TP",
    "SGLANG_CONTEXT_LENGTH",
    "SGLANG_MEM_FRACTION_STATIC",
    "SGLANG_CHUNKED_PREFILL_SIZE",
    "SGLANG_MAX_RUNNING_REQUESTS",
    "SGLANG_DTYPE",
    "SGLANG_KV_CACHE_DTYPE",
    "SGLANG_ATTENTION_BACKEND",
    "SGLANG_SAMPLING_BACKEND",
    "LLAMA3_3_RUNTIME_BACKEND",
    "SGLANG_LLAMA_RUNTIME_BACKEND",
    "DEEPSEEK_R1_MODEL",
    "DEEPSEEK_MODEL_PATH",
    "QWEN3_6_MODEL",
    "QWEN3_6_35B_MODEL",
    "QWEN3_6_MODEL_PATH",
    "LLAMA3_3_MODEL",
    "LLAMA3_3_MODEL_PATH",
    "LLAMA4_MODEL",
    "LLAMA4_MODEL_PATH",
    "QWYTHOS_MODEL",
    "QWYTHOS_MODEL_PATH",
    "CHATMOCK_BASE_URL",
    "CHATMOCK_API_KEY",
    "CHATMOCK_DEFAULT_MODEL",
    "CHATMOCK_GPT5_3_CODEX_MODEL",
    "CHATMOCK_GPT5_5_MODEL",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_GPT4_1_DEPLOYMENT",
    "AZURE_OPENAI_GPT4_DEPLOYMENT",
    "AZURE_OPENAI_GPT5_3_CODEX_DEPLOYMENT",
    "AZURE_OPENAI_GPT5_5_DEPLOYMENT",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_GPT4_1_MODEL",
    "OPENAI_GPT4_MODEL",
    "OPENAI_GPT5_3_CODEX_MODEL",
    "OPENAI_GPT5_5_MODEL",
    "ALLOW_OPENAI_COMPATIBLE_EXECUTE",
    "ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL",
    "CODEX_CLI_PATH",
    "CODEX_API_KEY",
    "CODEX_CLI_ASSUME_AUTH",
    "CODEX_CLI_TRUST_AUTH_FILE",
    "CODEX_CLI_MODEL",
    "CODEX_CLI_GPT5_3_MODEL",
    "CODEX_CLI_GPT5_5_MODEL",
    "GPT5_3_CODEX_ROUTE",
    "GPT5_5_CODEX_ROUTE",
]

DEFAULT_LOCAL_CREDENTIAL_FILES = [
    REPO_ROOT / ".env.local",
    REPO_ROOT / ".env.runtime",
    REPO_ROOT / ".llm_credentials.json",
    REPO_ROOT / ".secrets" / "llm_credentials.json",
    REPO_ROOT / ".secrets" / ".env",
]


@dataclass(frozen=True)
class CredentialStore:
    values: dict[str, str]
    sources: dict[str, str]
    files_used: list[str]
    warnings: list[str]

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def missing(self, names: list[str]) -> list[str]:
        return [name for name in names if not self.get(name)]

    def source_map(self, names: list[str]) -> dict[str, str]:
        return {name: self.sources[name] for name in names if name in self.sources}


@dataclass(frozen=True)
class EngineConfig:
    requested_engine: str
    engine: str
    route: str
    model: str
    actual_model: str
    formal_model: str
    runtime_profile: str
    endpoint: str | None
    api_version: str | None
    api_key: str | None
    missing_credentials: list[str]
    credential_sources: dict[str, str]
    credential_files: list[str]
    credential_warnings: list[str]

    @property
    def available(self) -> bool:
        return not self.missing_credentials

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "requested_engine": self.requested_engine,
            "engine": self.engine,
            "route": self.route,
            "backend": ROUTE_BACKENDS.get(self.route, self.route),
            "model": self.model,
            "formal_model": self.formal_model,
            "actual_model": self.actual_model,
            "runtime_profile": self.runtime_profile,
            "endpoint_set": self.endpoint is not None,
            "api_version": self.api_version,
            "available": self.available,
            "missing_credentials": self.missing_credentials,
            "credential_sources": self.credential_sources,
            "credential_files": self.credential_files,
            "credential_warnings": self.credential_warnings,
        }


@dataclass(frozen=True)
class SamplingPolicy:
    source: str
    temperature: float
    top_p: float
    n: int = 1
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    reasoning_effort: str | None = None
    reasoning_effort_intent: str | None = None
    enable_thinking: bool | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "source": self.source,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "n": self.n,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "presence_penalty": self.presence_penalty,
                "repetition_penalty": self.repetition_penalty,
                "reasoning_effort": self.reasoning_effort,
                "reasoning_effort_intent": self.reasoning_effort_intent,
                "enable_thinking": self.enable_thinking,
            }.items()
            if value is not None
        }


def normalize_engine(name: str) -> str:
    key = name.strip().lower()
    try:
        return ENGINE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported generator engine: {name}") from exc


def model_matches_formal_model(model: str, formal_model: str) -> bool:
    model_value = model.strip()
    formal_value = formal_model.strip()
    model_lower = model_value.replace("\\", "/").lower().rstrip("/")
    formal_lower = formal_value.replace("\\", "/").lower().strip("/")
    if model_lower == formal_lower:
        return True
    if "/" in formal_lower and formal_lower in model_lower:
        return True
    if "/" not in formal_value:
        return False
    cache_fragment = f"models--{formal_value.replace('/', '--')}/snapshots/".lower()
    return cache_fragment in model_lower


def runtime_profile_for(engine: str, route: str, model: str, formal_model: str) -> str:
    if route in {"local_vllm_openai_compatible", "chatmock_openai_compatible"}:
        return "formal" if model_matches_formal_model(model, formal_model) else "fallback_smoke"
    if route == "local_llama_cpp_openai_compatible":
        quant = os.environ.get("LLAMA_CPP_QUANT")
        return f"llama_cpp_{quant}" if quant else "llama_cpp"
    if route == "local_sglang_openai_compatible":
        profile = os.environ.get("SGLANG_RUNTIME_PROFILE")
        quant = os.environ.get("SGLANG_QUANTIZATION")
        return profile or (f"sglang_{quant}" if quant else "sglang")
    return "formal"


def build_engine_config(
    *,
    requested_engine: str,
    engine: str,
    route: str,
    model: str,
    endpoint: str | None,
    api_version: str | None,
    api_key: str | None,
    missing_credentials: list[str],
    credential_sources: dict[str, str],
    credential_files: list[str],
    credential_warnings: list[str],
    actual_model: str | None = None,
    formal_model: str | None = None,
    runtime_profile: str | None = None,
) -> EngineConfig:
    resolved_formal_model = formal_model or FORMAL_GENERATOR_MODELS.get(engine, model)
    resolved_actual_model = actual_model or model
    return EngineConfig(
        requested_engine=requested_engine,
        engine=engine,
        route=route,
        model=model,
        actual_model=resolved_actual_model,
        formal_model=resolved_formal_model,
        runtime_profile=runtime_profile
        or runtime_profile_for(engine, route, resolved_actual_model, resolved_formal_model),
        endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
        missing_credentials=missing_credentials,
        credential_sources=credential_sources,
        credential_files=credential_files,
        credential_warnings=credential_warnings,
    )


def normalize_openai_base_url(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_env_content(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_wrapping_quotes(value.strip())
        if key:
            values[key] = value
    return values


def parse_credential_file(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("credential json must be a flat object")
        parsed: dict[str, str] = {}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) and value != "":
                parsed[str(key)] = str(value)
        return parsed
    return parse_env_content(text)


def is_git_ignored(path: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "check-ignore", str(path)],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0


def credential_file_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("LLM_CREDENTIAL_FILE")
    if env_path:
        resolved = Path(env_path).expanduser()
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        candidates.append(resolved.resolve())
    for path in DEFAULT_LOCAL_CREDENTIAL_FILES:
        candidates.append(path.resolve())
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def collect_credentials() -> CredentialStore:
    file_values: dict[str, str] = {}
    file_sources: dict[str, str] = {}
    files_used: list[str] = []
    warnings: list[str] = []

    for path in credential_file_candidates():
        if not path.exists():
            continue
        if not is_git_ignored(path):
            warnings.append(f"Rejected local credential file because it is not git-ignored: {path.name}")
            continue
        try:
            parsed = parse_credential_file(path)
        except Exception:
            warnings.append(f"Rejected local credential file because it could not be parsed: {path.name}")
            continue
        if not parsed:
            continue
        files_used.append(path.name)
        source_label = f"ignored_local_file:{path.name}"
        for key, value in parsed.items():
            if value and key not in file_values:
                file_values[key] = value
                file_sources[key] = source_label

    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for name in CREDENTIAL_NAMES:
        env_value = os.environ.get(name)
        if env_value:
            values[name] = env_value
            sources[name] = "env"
        elif name in file_values:
            values[name] = file_values[name]
            sources[name] = file_sources[name]

    return CredentialStore(values=values, sources=sources, files_used=files_used, warnings=warnings)


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def codex_cli_auth_file_available() -> bool:
    candidates = [
        Path.home() / ".codex" / "auth.json",
        Path.home() / ".config" / "codex" / "auth.json",
    ]
    return any(path.is_file() and path.stat().st_size > 0 for path in candidates)


def codex_cli_auth_missing(credentials: CredentialStore, credential_purpose: str) -> list[str]:
    if credential_purpose != "execute":
        return []
    if credentials.get("OPENAI_API_KEY") or credentials.get("CODEX_API_KEY"):
        return []
    if truthy(credentials.get("CODEX_CLI_ASSUME_AUTH")):
        return []
    if truthy(credentials.get("CODEX_CLI_TRUST_AUTH_FILE")) and codex_cli_auth_file_available():
        return []
    return ["OPENAI_API_KEY or CODEX_API_KEY or CODEX_CLI_ASSUME_AUTH=1"]


def resolve_engine(name: str, credential_purpose: str = "test") -> EngineConfig:
    if credential_purpose not in {"test", "execute"}:
        raise ValueError(f"Unsupported credential_purpose={credential_purpose}; use test or execute")
    engine = normalize_engine(name)
    credentials = collect_credentials()
    chatmock_base_url = normalize_openai_base_url(credentials.get("CHATMOCK_BASE_URL"))
    allow_diagnostic_chatmock_execute = truthy(credentials.get("ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL"))

    def chatmock_formal_blocked(route_env: str, default_model: str) -> EngineConfig:
        return build_engine_config(
            requested_engine=name,
            engine=engine,
            route="codex_cli",
            model=default_model,
            endpoint=None,
            api_version=None,
            api_key=None,
            missing_credentials=[
                f"{route_env}=chatmock is diagnostic-only for Experiment 7 formal execution; "
                "use codex_cli, openai, or azure, or set ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL=1 and label the run diagnostic"
            ],
            credential_sources=credentials.source_map(
                [
                    route_env,
                    "CHATMOCK_BASE_URL",
                    "CHATMOCK_API_KEY",
                    "ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL",
                ]
            ),
            credential_files=credentials.files_used,
            credential_warnings=credentials.warnings
            + ["ChatMock output must not be claimed as formal GPT-series EA."],
        )

    def vllm_engine(
        env_model_name: str,
        default_model: str,
        compat_env_model_names: list[str] | None = None,
        env_path_names: list[str] | None = None,
    ) -> EngineConfig:
        required = ["VLLM_BASE_URL"]
        model_env_names = [env_model_name] + list(compat_env_model_names or [])
        path_env_names = list(env_path_names or [])
        actual_model = next(
            (credentials.get(item) for item in path_env_names + model_env_names if credentials.get(item)),
            default_model,
        )
        served_model = credentials.get("VLLM_SERVED_MODEL_NAME") or engine
        return build_engine_config(
            requested_engine=name,
            engine=engine,
            route="local_vllm_openai_compatible",
            model=served_model,
            actual_model=actual_model,
            endpoint=credentials.get("VLLM_BASE_URL"),
            api_version=None,
            api_key=credentials.get("VLLM_API_KEY") or "EMPTY",
            missing_credentials=credentials.missing(required),
            credential_sources=credentials.source_map(
                required
                + ["VLLM_API_KEY", "VLLM_SERVED_MODEL_NAME", "MODELS_ROOT", "HF_HOME"]
                + path_env_names
                + model_env_names
            ),
            credential_files=credentials.files_used,
            credential_warnings=credentials.warnings,
        )


    def sglang_engine(
        env_model_name: str,
        default_formal_model: str,
        default_actual_model: str | None = None,
    ) -> EngineConfig:
        required = ["SGLANG_BASE_URL", "SGLANG_MODEL_PATH"]
        actual_model = credentials.get("SGLANG_MODEL_PATH") or default_actual_model or default_formal_model
        served_model = credentials.get("SGLANG_SERVED_MODEL_NAME") or engine
        formal_model = credentials.get("SGLANG_FORMAL_MODEL") or default_formal_model
        runtime_profile = credentials.get("SGLANG_RUNTIME_PROFILE") or runtime_profile_for(
            engine, "local_sglang_openai_compatible", actual_model, formal_model
        )
        return build_engine_config(
            requested_engine=name,
            engine=engine,
            route="local_sglang_openai_compatible",
            model=served_model,
            actual_model=actual_model,
            formal_model=formal_model,
            endpoint=normalize_openai_base_url(credentials.get("SGLANG_BASE_URL")),
            api_version=None,
            api_key=credentials.get("SGLANG_API_KEY") or "EMPTY",
            missing_credentials=credentials.missing(required),
            credential_sources=credentials.source_map(
                required
                + [
                    "SGLANG_API_KEY",
                    "SGLANG_SERVED_MODEL_NAME",
                    "SGLANG_FORMAL_MODEL",
                    "SGLANG_RUNTIME_PROFILE",
                    "SGLANG_QUANTIZATION",
                    "SGLANG_LOAD_FORMAT",
                    "SGLANG_TP",
                    "SGLANG_CONTEXT_LENGTH",
                    "SGLANG_MEM_FRACTION_STATIC",
                    "SGLANG_CHUNKED_PREFILL_SIZE",
                    "SGLANG_MAX_RUNNING_REQUESTS",
                    "SGLANG_DTYPE",
                    "SGLANG_KV_CACHE_DTYPE",
                    "SGLANG_ATTENTION_BACKEND",
                    "SGLANG_SAMPLING_BACKEND",
                    env_model_name,
                ]
            ),
            credential_files=credentials.files_used,
            credential_warnings=credentials.warnings,
            runtime_profile=runtime_profile,
        )

    def llama_cpp_engine() -> EngineConfig:
        required = ["LLAMA_CPP_BASE_URL", "LLAMA_CPP_MODEL_PATH"]
        actual_model = credentials.get("LLAMA_CPP_MODEL_PATH") or ""
        model_alias = credentials.get("LLAMA_CPP_MODEL_ALIAS") or engine
        quant = credentials.get("LLAMA_CPP_QUANT") or ""
        runtime_profile = f"llama_cpp_{quant}" if quant else "llama_cpp"
        default_llama_cpp_formal_model = (
            "unsloth/Llama-3.3-70B-Instruct-GGUF"
            if engine == "llama3_3"
            else "unsloth/Mistral-Small-4-119B-2603-GGUF"
        )
        formal_model = (
            credentials.get("LLAMA_CPP_FORMAL_MODEL")
            or credentials.get("MISTRAL_SMALL_LLAMA_CPP_FORMAL_MODEL")
            or default_llama_cpp_formal_model
        )
        return build_engine_config(
            requested_engine=name,
            engine=engine,
            route="local_llama_cpp_openai_compatible",
            model=model_alias,
            actual_model=actual_model or model_alias,
            formal_model=formal_model,
            endpoint=normalize_openai_base_url(credentials.get("LLAMA_CPP_BASE_URL")),
            api_version=None,
            api_key=credentials.get("LLAMA_CPP_API_KEY") or "EMPTY",
            missing_credentials=credentials.missing(required),
            credential_sources=credentials.source_map(
                required
                + [
                    "MISTRAL_SMALL_RUNTIME_BACKEND",
                    "LLAMA3_3_RUNTIME_BACKEND",
                    "LLAMA_CPP_FORMAL_MODEL",
                    "MISTRAL_SMALL_LLAMA_CPP_FORMAL_MODEL",
                    "LLAMA_CPP_API_KEY",
                    "LLAMA_CPP_MODEL_ALIAS",
                    "LLAMA_CPP_QUANT",
                    "LLAMA_CPP_CTX_SIZE",
                    "LLAMA_CPP_N_GPU_LAYERS",
                    "LLAMA_CPP_TENSOR_SPLIT",
                    "LLAMA_CPP_SPLIT_MODE",
                    "LLAMA_CPP_PARALLEL",
                    "LLAMA_CPP_BATCH_SIZE",
                    "LLAMA_CPP_UBATCH_SIZE",
                    "LLAMA_CPP_THREADS",
                    "LLAMA_CPP_CACHE_TYPE_K",
                    "LLAMA_CPP_CACHE_TYPE_V",
                    "LLAMA_CPP_KV_OFFLOAD",
                    "LLAMA_CPP_CPU_MOE",
                    "LLAMA_CPP_N_CPU_MOE",
                    "LLAMA_CPP_FIT_TARGET",
                    "LLAMA_CPP_FIT_CTX",
                    "LLAMA_CPP_OP_OFFLOAD",
                    "LLAMA_CPP_FLASH_ATTN",
                    "LLAMA_CPP_CACHE_RAM",
                    "LLAMA_CPP_DEVICE",
                    "LLAMA_CPP_MAIN_GPU",
                    "LLAMA_CPP_NO_MMAP",
                ]
            ),
            credential_files=credentials.files_used,
            credential_warnings=credentials.warnings,
            runtime_profile=runtime_profile,
        )

    if engine == "mistral4":
        backend = (credentials.get("MISTRAL_SMALL_RUNTIME_BACKEND") or "vllm").strip().lower().replace("-", "_")
        if backend in {"llama_cpp", "llamacpp", "local_llama_cpp_openai_compatible"}:
            return llama_cpp_engine()
        if backend in {"sglang", "local_sglang_openai_compatible"}:
            return sglang_engine("MISTRAL_SMALL_MODEL", "mistralai/Mistral-Small-4-119B-2603-NVFP4")
        if backend in {"vllm", "local_vllm_openai_compatible"}:
            return vllm_engine(
                "MISTRAL_SMALL_MODEL",
                "mistralai/Mistral-Small-4-119B-2603-NVFP4",
                env_path_names=["MISTRAL_SMALL_MODEL_PATH"],
            )
        return build_engine_config(
            requested_engine=name,
            engine=engine,
            route="local_vllm_openai_compatible",
            model=engine,
            actual_model="mistralai/Mistral-Small-4-119B-2603-NVFP4",
            endpoint=credentials.get("VLLM_BASE_URL"),
            api_version=None,
            api_key=credentials.get("VLLM_API_KEY") or "EMPTY",
            missing_credentials=[f"Unsupported MISTRAL_SMALL_RUNTIME_BACKEND={backend}"],
            credential_sources=credentials.source_map(["MISTRAL_SMALL_RUNTIME_BACKEND"]),
            credential_files=credentials.files_used,
            credential_warnings=credentials.warnings,
        )
    if engine == "deepseek_r1_qwen32b":
        return vllm_engine(
            "DEEPSEEK_R1_MODEL",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            env_path_names=["DEEPSEEK_MODEL_PATH"],
        )
    if engine == "qwen3_6":
        return vllm_engine(
            "QWEN3_6_MODEL",
            "Qwen/Qwen3.6-35B-A3B-FP8",
            ["QWEN3_6_35B_MODEL"],
            env_path_names=["QWEN3_6_MODEL_PATH"],
        )
    if engine == "llama3_3":
        backend = (credentials.get("LLAMA3_3_RUNTIME_BACKEND") or credentials.get("SGLANG_LLAMA_RUNTIME_BACKEND") or "vllm").strip().lower().replace("-", "_")
        if backend in {"llama_cpp", "llamacpp", "local_llama_cpp_openai_compatible"}:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="local_vllm_openai_compatible",
                model=engine,
                actual_model="meta-llama/Llama-3.3-70B-Instruct",
                endpoint=credentials.get("VLLM_BASE_URL"),
                api_version=None,
                api_key=credentials.get("VLLM_API_KEY") or "EMPTY",
                missing_credentials=[
                    "Llama GGUF/llama.cpp runnable route is disabled; use the official Llama 3.3 vLLM route or keep historical logs/docs only."
                ],
                credential_sources=credentials.source_map(["LLAMA3_3_RUNTIME_BACKEND"]),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        if backend in {"sglang", "local_sglang_openai_compatible"}:
            return sglang_engine("LLAMA3_3_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        return vllm_engine(
            "LLAMA3_3_MODEL",
            "meta-llama/Llama-3.3-70B-Instruct",
            env_path_names=["LLAMA3_3_MODEL_PATH"],
        )
    if engine == "llama4":
        return vllm_engine(
            "LLAMA4_MODEL",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            env_path_names=["LLAMA4_MODEL_PATH"],
        )
    if engine == "qwythos9b":
        return vllm_engine(
            "QWYTHOS_MODEL",
            "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
            env_path_names=["QWYTHOS_MODEL_PATH"],
        )
    if engine == "gpt4_1":
        openai_base_url = normalize_openai_base_url(credentials.get("OPENAI_BASE_URL"))
        openai_key = credentials.get("OPENAI_API_KEY")
        openai_model = credentials.get("OPENAI_GPT4_1_MODEL") or credentials.get("OPENAI_GPT4_MODEL") or "gpt-4.1"
        allow_compatible_execute = truthy(credentials.get("ALLOW_OPENAI_COMPATIBLE_EXECUTE"))
        azure_required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
        azure_deployment = (
            credentials.get("AZURE_OPENAI_GPT4_1_DEPLOYMENT")
            or credentials.get("AZURE_OPENAI_GPT4_DEPLOYMENT")
            or credentials.get("AZURE_OPENAI_DEPLOYMENT")
        )

        def compatible_config() -> EngineConfig:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="closed_api_openai_compatible",
                model=openai_model,
                endpoint=openai_base_url,
                api_version=None,
                api_key=openai_key,
                missing_credentials=[] if openai_key else ["OPENAI_API_KEY"],
                credential_sources=credentials.source_map(
                    [
                        "OPENAI_BASE_URL",
                        "OPENAI_API_KEY",
                        "OPENAI_GPT4_1_MODEL",
                        "OPENAI_GPT4_MODEL",
                        "ALLOW_OPENAI_COMPATIBLE_EXECUTE",
                    ]
                ),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )

        def official_openai_config() -> EngineConfig:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="closed_api_openai",
                model=openai_model,
                endpoint=None,
                api_version=None,
                api_key=openai_key,
                missing_credentials=[] if openai_key else ["OPENAI_API_KEY"],
                credential_sources=credentials.source_map(["OPENAI_API_KEY", "OPENAI_GPT4_1_MODEL", "OPENAI_GPT4_MODEL"]),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )

        def azure_config(extra_warnings: list[str] | None = None) -> EngineConfig:
            missing = credentials.missing(azure_required)
            if not azure_deployment:
                missing.append("AZURE_OPENAI_GPT4_1_DEPLOYMENT or AZURE_OPENAI_GPT4_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT")
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="closed_api_azure_openai",
                model=azure_deployment or "gpt-4.1",
                endpoint=credentials.get("AZURE_OPENAI_ENDPOINT"),
                api_version=credentials.get("AZURE_OPENAI_API_VERSION") or "2024-02-01",
                api_key=credentials.get("AZURE_OPENAI_API_KEY"),
                missing_credentials=missing,
                credential_sources=credentials.source_map(
                    azure_required
                    + [
                        "AZURE_OPENAI_API_VERSION",
                        "AZURE_OPENAI_GPT4_1_DEPLOYMENT",
                        "AZURE_OPENAI_GPT4_DEPLOYMENT",
                        "AZURE_OPENAI_DEPLOYMENT",
                        "OPENAI_BASE_URL",
                    ]
                ),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings + (extra_warnings or []),
            )

        def compatible_blocked_for_execution() -> EngineConfig:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="closed_api_openai",
                model=openai_model,
                endpoint=None,
                api_version=None,
                api_key=None,
                missing_credentials=[
                    "formal GPT-4.1 execution ignores OPENAI_BASE_URL; unset OPENAI_BASE_URL and set an official OPENAI_API_KEY, or set Azure OpenAI credentials"
                ],
                credential_sources=credentials.source_map(
                    [
                        "OPENAI_BASE_URL",
                        "OPENAI_API_KEY",
                        "OPENAI_GPT4_1_MODEL",
                        "OPENAI_GPT4_MODEL",
                        "AZURE_OPENAI_ENDPOINT",
                        "AZURE_OPENAI_API_KEY",
                        "AZURE_OPENAI_GPT4_1_DEPLOYMENT",
                    ]
                ),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings
                + [
                    "OPENAI_BASE_URL is reserved for test/validation by default because ChatAnywhere/GPT_API_free is quota-limited."
                ],
            )

        if credential_purpose == "test":
            if openai_base_url:
                return compatible_config()
            if openai_key:
                return official_openai_config()
            return azure_config()

        if openai_base_url:
            if allow_compatible_execute:
                return compatible_config()
            if all(credentials.get(name) for name in azure_required) and azure_deployment:
                return azure_config(
                    [
                        "OPENAI_BASE_URL ignored for credential_purpose=execute; ChatAnywhere/OpenAI-compatible endpoints are test-only by default."
                    ]
                )
            return compatible_blocked_for_execution()
        if openai_key:
            return official_openai_config()
        return azure_config()
    if engine == "gpt5_3_codexS":
        route_override = (
            credentials.get("GPT5_3_CODEX_ROUTE")
            or ("chatmock" if chatmock_base_url else "codex_cli")
        ).strip().lower()
        if (
            credential_purpose == "execute"
            and route_override in {"chatmock", "chatmock_openai_compatible"}
            and not allow_diagnostic_chatmock_execute
        ):
            return chatmock_formal_blocked("GPT5_3_CODEX_ROUTE", "gpt-5.3-codex-spark")
        if chatmock_base_url and route_override in {"chatmock", "chatmock_openai_compatible"}:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="chatmock_openai_compatible",
                model=credentials.get("CHATMOCK_GPT5_3_CODEX_MODEL")
                or credentials.get("CHATMOCK_DEFAULT_MODEL")
                or "gpt-5.3-codex-spark",
                endpoint=chatmock_base_url,
                api_version=None,
                api_key=credentials.get("CHATMOCK_API_KEY") or "key",
                missing_credentials=[],
                credential_sources=credentials.source_map(
                    [
                        "CHATMOCK_BASE_URL",
                        "CHATMOCK_API_KEY",
                        "CHATMOCK_GPT5_3_CODEX_MODEL",
                        "CHATMOCK_DEFAULT_MODEL",
                        "GPT5_3_CODEX_ROUTE",
                    ]
                ),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        if route_override in {"chatmock", "chatmock_openai_compatible"}:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="chatmock_openai_compatible",
                model=credentials.get("CHATMOCK_GPT5_3_CODEX_MODEL")
                or credentials.get("CHATMOCK_DEFAULT_MODEL")
                or "gpt-5.3-codex-spark",
                endpoint=None,
                api_version=None,
                api_key=None,
                missing_credentials=["CHATMOCK_BASE_URL"],
                credential_sources=credentials.source_map(
                    ["CHATMOCK_GPT5_3_CODEX_MODEL", "CHATMOCK_DEFAULT_MODEL", "GPT5_3_CODEX_ROUTE"]
                ),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        if route_override in {"api_key", "apikey", "openai_api_key", "openai", "closed_api_openai"}:
            openai_api_key = credentials.get("OPENAI_API_KEY") or credentials.get("CODEX_API_KEY")
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="closed_api_openai",
                model=credentials.get("OPENAI_GPT5_3_CODEX_MODEL")
                or credentials.get("CODEX_CLI_GPT5_3_MODEL")
                or "gpt-5.3-codex-spark",
                endpoint=None,
                api_version=None,
                api_key=openai_api_key,
                missing_credentials=[] if openai_api_key else ["OPENAI_API_KEY or CODEX_API_KEY"],
                credential_sources=credentials.source_map(
                    [
                        "OPENAI_API_KEY",
                        "CODEX_API_KEY",
                        "OPENAI_GPT5_3_CODEX_MODEL",
                        "CODEX_CLI_GPT5_3_MODEL",
                        "GPT5_3_CODEX_ROUTE",
                    ]
                ),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        if route_override in {"codex", "codex_cli", "cli"}:
            codex_path = credentials.get("CODEX_CLI_PATH") or shutil.which("codex")
            codex_api_key = credentials.get("OPENAI_API_KEY") or credentials.get("CODEX_API_KEY")
            missing = [] if codex_path else ["CODEX_CLI_PATH or codex on PATH"]
            missing.extend(codex_cli_auth_missing(credentials, credential_purpose))
            sources = credentials.source_map(
                [
                    "CODEX_CLI_PATH",
                    "CODEX_API_KEY",
                    "CODEX_CLI_ASSUME_AUTH",
                    "CODEX_CLI_TRUST_AUTH_FILE",
                    "CODEX_CLI_GPT5_3_MODEL",
                    "CODEX_CLI_MODEL",
                    "GPT5_3_CODEX_ROUTE",
                    "OPENAI_API_KEY",
                ]
            )
            if codex_path and "CODEX_CLI_PATH" not in sources:
                sources["CODEX_CLI_PATH"] = "PATH"
            if truthy(credentials.get("CODEX_CLI_TRUST_AUTH_FILE")) and codex_cli_auth_file_available() and not codex_api_key:
                sources["CODEX_CLI_AUTH_FILE"] = "local_codex_auth_file"
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="codex_cli",
                model=credentials.get("CODEX_CLI_GPT5_3_MODEL")
                or credentials.get("CODEX_CLI_MODEL")
                or "gpt-5.3-codex-spark",
                endpoint=codex_path,
                api_version=None,
                api_key=codex_api_key,
                missing_credentials=missing,
                credential_sources=sources,
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        if route_override not in {"azure", "azure_openai", "closed_api_azure_openai", "chatmock", "chatmock_openai_compatible"}:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="codex_cli",
                model=credentials.get("CODEX_CLI_GPT5_3_MODEL")
                or credentials.get("CODEX_CLI_MODEL")
                or "gpt-5.3-codex-spark",
                endpoint=credentials.get("CODEX_CLI_PATH"),
                api_version=None,
                api_key=None,
                missing_credentials=[f"Unsupported GPT5_3_CODEX_ROUTE={route_override}"],
                credential_sources=credentials.source_map(["GPT5_3_CODEX_ROUTE"]),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
        deployment = credentials.get("AZURE_OPENAI_GPT5_3_CODEX_DEPLOYMENT") or credentials.get(
            "AZURE_OPENAI_DEPLOYMENT"
        )
        missing = credentials.missing(required)
        if not deployment:
            missing.append("AZURE_OPENAI_GPT5_3_CODEX_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT")
        return build_engine_config(
            requested_engine=name,
            engine=engine,
            route="closed_api_azure_openai",
            model=deployment or "gpt-5.3-codex-spark",
            endpoint=credentials.get("AZURE_OPENAI_ENDPOINT"),
            api_version=credentials.get("AZURE_OPENAI_API_VERSION") or "2024-02-01",
            api_key=credentials.get("AZURE_OPENAI_API_KEY"),
            missing_credentials=missing,
            credential_sources=credentials.source_map(
                required
                + [
                    "AZURE_OPENAI_API_VERSION",
                    "AZURE_OPENAI_GPT5_3_CODEX_DEPLOYMENT",
                    "AZURE_OPENAI_DEPLOYMENT",
                ]
            ),
            credential_files=credentials.files_used,
            credential_warnings=credentials.warnings,
        )
    if engine == "gpt5_5":
        route_override = (
            credentials.get("GPT5_5_CODEX_ROUTE")
            or ("chatmock" if chatmock_base_url else "codex_cli")
        ).strip().lower()
        if (
            credential_purpose == "execute"
            and route_override in {"chatmock", "chatmock_openai_compatible"}
            and not allow_diagnostic_chatmock_execute
        ):
            return chatmock_formal_blocked("GPT5_5_CODEX_ROUTE", "gpt-5.5")
        openai_base_url = normalize_openai_base_url(credentials.get("OPENAI_BASE_URL"))
        openai_key = credentials.get("OPENAI_API_KEY")
        openai_compatible_aliases = {
            "openai_compatible",
            "openai-compatible",
            "closed_api_openai_compatible",
            "chatmock_via_openai_base_url",
        }

        def openai_compatible_config() -> EngineConfig:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="closed_api_openai_compatible",
                model=credentials.get("OPENAI_GPT5_5_MODEL")
                or credentials.get("CHATMOCK_GPT5_5_MODEL")
                or credentials.get("CHATMOCK_DEFAULT_MODEL")
                or "gpt-5.5",
                endpoint=openai_base_url,
                api_version=None,
                api_key=openai_key,
                missing_credentials=[] if openai_key else ["OPENAI_API_KEY"],
                credential_sources=credentials.source_map(
                    [
                        "OPENAI_BASE_URL",
                        "OPENAI_API_KEY",
                        "OPENAI_GPT5_5_MODEL",
                        "CHATMOCK_GPT5_5_MODEL",
                        "CHATMOCK_DEFAULT_MODEL",
                        "GPT5_5_CODEX_ROUTE",
                        "ALLOW_OPENAI_COMPATIBLE_EXECUTE",
                    ]
                ),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )

        if openai_base_url and (
            route_override in openai_compatible_aliases
            or (route_override in {"chatmock", "chatmock_openai_compatible"} and not chatmock_base_url)
            or (truthy(credentials.get("ALLOW_OPENAI_COMPATIBLE_EXECUTE")) and route_override in {"codex", "codex_cli", "cli"})
        ):
            return openai_compatible_config()
        if chatmock_base_url and route_override in {"chatmock", "chatmock_openai_compatible"}:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="chatmock_openai_compatible",
                model=credentials.get("CHATMOCK_GPT5_5_MODEL")
                or credentials.get("CHATMOCK_DEFAULT_MODEL")
                or "gpt-5.5",
                endpoint=chatmock_base_url,
                api_version=None,
                api_key=credentials.get("CHATMOCK_API_KEY") or "key",
                missing_credentials=[],
                credential_sources=credentials.source_map(
                    [
                        "CHATMOCK_BASE_URL",
                        "CHATMOCK_API_KEY",
                        "CHATMOCK_GPT5_5_MODEL",
                        "CHATMOCK_DEFAULT_MODEL",
                        "GPT5_5_CODEX_ROUTE",
                    ]
                ),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        if route_override in {"chatmock", "chatmock_openai_compatible"}:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="chatmock_openai_compatible",
                model=credentials.get("CHATMOCK_GPT5_5_MODEL")
                or credentials.get("CHATMOCK_DEFAULT_MODEL")
                or "gpt-5.5",
                endpoint=None,
                api_version=None,
                api_key=None,
                missing_credentials=["CHATMOCK_BASE_URL"],
                credential_sources=credentials.source_map(
                    ["CHATMOCK_GPT5_5_MODEL", "CHATMOCK_DEFAULT_MODEL", "GPT5_5_CODEX_ROUTE"]
                ),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        if route_override in {"codex", "codex_cli", "cli"}:
            codex_path = credentials.get("CODEX_CLI_PATH") or shutil.which("codex")
            codex_api_key = credentials.get("OPENAI_API_KEY") or credentials.get("CODEX_API_KEY")
            missing = [] if codex_path else ["CODEX_CLI_PATH or codex on PATH"]
            missing.extend(codex_cli_auth_missing(credentials, credential_purpose))
            sources = credentials.source_map(
                [
                    "CODEX_CLI_PATH",
                    "CODEX_API_KEY",
                    "CODEX_CLI_ASSUME_AUTH",
                    "CODEX_CLI_TRUST_AUTH_FILE",
                    "CODEX_CLI_GPT5_5_MODEL",
                    "CODEX_CLI_MODEL",
                    "GPT5_5_CODEX_ROUTE",
                    "OPENAI_API_KEY",
                ]
            )
            if codex_path and "CODEX_CLI_PATH" not in sources:
                sources["CODEX_CLI_PATH"] = "PATH"
            if truthy(credentials.get("CODEX_CLI_TRUST_AUTH_FILE")) and codex_cli_auth_file_available() and not codex_api_key:
                sources["CODEX_CLI_AUTH_FILE"] = "local_codex_auth_file"
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="codex_cli",
                model=credentials.get("CODEX_CLI_GPT5_5_MODEL")
                or credentials.get("CODEX_CLI_MODEL")
                or "gpt-5.5",
                endpoint=codex_path,
                api_version=None,
                api_key=codex_api_key,
                missing_credentials=missing,
                credential_sources=sources,
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        if route_override not in {
            "openai",
            "closed_api_openai",
            "azure",
            "azure_openai",
            "closed_api_azure_openai",
            "chatmock",
            "chatmock_openai_compatible",
        }:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="codex_cli",
                model=credentials.get("CODEX_CLI_GPT5_5_MODEL")
                or credentials.get("CODEX_CLI_MODEL")
                or "gpt-5.5",
                endpoint=credentials.get("CODEX_CLI_PATH"),
                api_version=None,
                api_key=None,
                missing_credentials=[f"Unsupported GPT5_5_CODEX_ROUTE={route_override}"],
                credential_sources=credentials.source_map(["GPT5_5_CODEX_ROUTE"]),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
        )
        openai_key = credentials.get("OPENAI_API_KEY")
        if route_override in {"openai", "closed_api_openai"} and openai_key:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="closed_api_openai",
                model=credentials.get("OPENAI_GPT5_5_MODEL") or "gpt-5.5",
                endpoint=None,
                api_version=None,
                api_key=openai_key,
                missing_credentials=[],
                credential_sources=credentials.source_map(["OPENAI_API_KEY", "OPENAI_GPT5_5_MODEL"]),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        if route_override in {"openai", "closed_api_openai"}:
            return build_engine_config(
                requested_engine=name,
                engine=engine,
                route="closed_api_openai",
                model=credentials.get("OPENAI_GPT5_5_MODEL") or "gpt-5.5",
                endpoint=None,
                api_version=None,
                api_key=None,
                missing_credentials=["OPENAI_API_KEY"],
                credential_sources=credentials.source_map(["OPENAI_API_KEY", "OPENAI_GPT5_5_MODEL"]),
                credential_files=credentials.files_used,
                credential_warnings=credentials.warnings,
            )
        required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
        deployment = credentials.get("AZURE_OPENAI_GPT5_5_DEPLOYMENT") or credentials.get(
            "AZURE_OPENAI_DEPLOYMENT"
        )
        missing = credentials.missing(required)
        if not deployment:
            missing.append("AZURE_OPENAI_GPT5_5_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT")
        return build_engine_config(
            requested_engine=name,
            engine=engine,
            route="closed_api_azure_openai",
            model=deployment or "gpt-5.5",
            endpoint=credentials.get("AZURE_OPENAI_ENDPOINT"),
            api_version=credentials.get("AZURE_OPENAI_API_VERSION") or "2024-02-01",
            api_key=credentials.get("AZURE_OPENAI_API_KEY"),
            missing_credentials=missing,
            credential_sources=credentials.source_map(
                required
                + [
                    "AZURE_OPENAI_API_VERSION",
                    "AZURE_OPENAI_GPT5_5_DEPLOYMENT",
                    "AZURE_OPENAI_DEPLOYMENT",
                ]
            ),
            credential_files=credentials.files_used,
            credential_warnings=credentials.warnings,
        )
    raise ValueError(f"Unsupported generator engine: {name}")


def stringify_prompt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def format_prompt_facts(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(stringify_prompt_value(item).strip() for item in value if stringify_prompt_value(item).strip())
    return stringify_prompt_value(value).strip()


FINQA_DSL_OPERATORS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "greater",
    "exp",
    "table_average",
    "table_max",
    "table_min",
    "table_sum",
}


def split_top_level_expressions(program: str) -> list[str]:
    expressions: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(program):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            item = program[start:index].strip()
            if item:
                expressions.append(item)
            start = index + 1
    tail = program[start:].strip()
    if tail:
        expressions.append(tail)
    return expressions


def const_token_to_python(token: str) -> str:
    value = token.removeprefix("const_")
    if value.startswith("m") and value[1:].isdigit():
        return f"(-{value[1:]})"
    if value.isdigit():
        return value
    raise ValueError(f"Unsupported FinQA constant: {token}")


def normalize_dsl_tokens(text: str) -> str:
    text = re.sub(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*%", r"(\1 / 100)", text)
    text = re.sub(r"#(\d+)", r"ref_\1", text)
    return re.sub(r"\bconst_(?:m?\d+)\b", lambda match: const_token_to_python(match.group(0)), text)


def ast_source(node: ast.AST, original: str) -> str:
    segment = ast.get_source_segment(original, node)
    if segment is not None:
        return segment.strip()
    return ast.unparse(node).strip()


def dsl_ast_to_python(node: ast.AST, original: str) -> str:
    if isinstance(node, ast.Expression):
        return dsl_ast_to_python(node.body, original)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        if re.fullmatch(r"ref_\d+", node.id):
            return f"step_{node.id.split('_', 1)[1]}"
        return node.id
    if isinstance(node, ast.UnaryOp):
        operand = dsl_ast_to_python(node.operand, original)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.UAdd):
            return f"(+{operand})"
    if isinstance(node, ast.BinOp):
        left = dsl_ast_to_python(node.left, original)
        right = dsl_ast_to_python(node.right, original)
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "**",
            ast.Mod: "%",
        }
        for op_type, op_text in op_map.items():
            if isinstance(node.op, op_type):
                return f"({left} {op_text} {right})"
    if isinstance(node, ast.List):
        return "[" + ", ".join(dsl_ast_to_python(item, original) for item in node.elts) + "]"
    if isinstance(node, ast.Tuple):
        return "(" + ", ".join(dsl_ast_to_python(item, original) for item in node.elts) + ")"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = node.func.id
        args = [dsl_ast_to_python(arg, original) for arg in node.args]
        if func == "add" and len(args) == 2:
            return f"({args[0]} + {args[1]})"
        if func == "subtract" and len(args) == 2:
            return f"({args[0]} - {args[1]})"
        if func == "multiply" and len(args) == 2:
            return f"({args[0]} * {args[1]})"
        if func == "divide" and len(args) == 2:
            return f"({args[0]} / {args[1]})"
        if func == "greater" and len(args) == 2:
            return f"({args[0]} > {args[1]})"
        if func == "exp" and len(args) == 2:
            return f"({args[0]} ** {args[1]})"
        if func in {"table_sum", "table_average", "table_max", "table_min"} and args:
            values = args[0] if len(args) == 1 and args[0].startswith("[") else "[" + ", ".join(args) + "]"
            if func == "table_sum":
                return f"sum({values})"
            if func == "table_average":
                return f"(sum({values}) / len({values}))"
            if func == "table_max":
                return f"max({values})"
            if func == "table_min":
                return f"min({values})"
    raise ValueError(f"Unsupported FinQA DSL node: {ast.dump(node, include_attributes=False)}")


def looks_like_finqa_dsl(program: str) -> bool:
    if "#" in program or re.search(r"\bconst_(?:m?\d+)\b", program):
        return True
    return any(re.search(rf"\b{re.escape(operator)}\s*\(", program) for operator in FINQA_DSL_OPERATORS)


def pythonize_finqa_program(program: str) -> str:
    source = stringify_prompt_value(program).strip()
    if not source:
        return ""
    expressions = split_top_level_expressions(source.replace("\n", " "))
    if not expressions or not looks_like_finqa_dsl(source):
        return source.rstrip()
    lines: list[str] = []
    for index, raw_expression in enumerate(expressions):
        expression = normalize_dsl_tokens(raw_expression)
        parsed = ast.parse(expression, mode="eval")
        python_expression = dsl_ast_to_python(parsed, expression)
        lines.append(f"step_{index} = {python_expression}")
    lines.append(f"ans = step_{len(expressions) - 1}")
    return "\n".join(lines)


def answer_program_for_prompt(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, bool):
        return f"ans = {answer}"
    if isinstance(answer, (int, float)):
        return f"ans = {answer!r}"
    raw_answer = stringify_prompt_value(answer).strip()
    if not raw_answer:
        return ""
    if re.fullmatch(r"-?\d+(?:\.\d+)?", raw_answer):
        return f"ans = {raw_answer}"
    return f"ans = {raw_answer!r}"


def format_example_program_for_prompt(program: Any, answer: Any = None) -> str:
    source = stringify_prompt_value(program).strip()
    if not source:
        return answer_program_for_prompt(answer)
    try:
        return pythonize_finqa_program(source)
    except Exception:
        if looks_like_finqa_dsl(source):
            fallback = answer_program_for_prompt(answer)
            if fallback:
                return fallback
        return source.rstrip()


GENERATOR_CODE_ONLY_INSTRUCTIONS = (
    "Generate only executable Python code.\n"
    "Assign the final result to a variable named ans.\n"
    "Define every variable before using it.\n"
    "Do not provide investment advice.\n"
    "Do not explain in natural language after the code.\n"
)


GENERATOR_CODE_ONLY_SYSTEM_PROMPT = (
    "Return only executable Python code that computes the FinQA answer. "
    "Assign the final result to a variable named ans. "
    "Define every variable before using it. "
    "Do not provide investment advice. "
    "Do not explain in natural language after the code. "
    "Do not use markdown fences."
)


def example_allows_dev_generator_prompt_guard(example: dict[str, Any]) -> bool:
    values = [
        example.get("source"),
        example.get("target_gold_csv"),
        example.get("retrieved_source"),
        example.get("flow_scope"),
    ]
    haystack = " ".join(stringify_prompt_value(value) for value in values).replace("\\", "/").lower()
    if "data/testing/finqa_10_rel_fact_instruction.csv" in haystack:
        return True
    if "finqa_dev" in haystack or "data/src/finqa/dev.json" in haystack:
        return True
    return False


def build_target_prompt(example: dict[str, Any], include_code: bool = False) -> str:
    intro = (
        "Read the following table and probable relevant facts, and then the python code below "
        "that answers the question, the answer can be a float/int or bool:\n"
        if include_code
        else "Read the following table and probable relevant facts, and then write code to answer "
        "a question, the answer can be a float/int or bool:\n"
    )
    prompt = intro
    table = example.get("table")
    if table in (None, ""):
        table = example.get("table_text", "")
    prompt += "Table: \n" + stringify_prompt_value(table).strip() + "\n"
    prompt += "Probable relevant facts:\n "
    retrieved = example.get("rel_fact") or example.get("retrieved") or example.get("text") or []
    prompt += format_prompt_facts(retrieved) + "\n\n"
    prompt += "Question: {}\n".format(example.get("question", ""))
    if not include_code and example_allows_dev_generator_prompt_guard(example):
        prompt += GENERATOR_CODE_ONLY_INSTRUCTIONS
    prompt += "#Python code below\n"
    if include_code:
        program = format_example_program_for_prompt(example.get("program"), example.get("answer"))
        if program:
            prompt += program.rstrip() + "\n"
    return prompt.rstrip() + "\n"


def selected_examples_from(example: dict[str, Any]) -> list[dict[str, Any]]:
    selected = example.get("selected_examples")
    if not isinstance(selected, list):
        return []
    return [item for item in selected if isinstance(item, dict)]


def prompt_contract_for_example(example: dict[str, Any]) -> dict[str, Any]:
    selected = selected_examples_from(example)
    prompt = stringify_prompt_value(example.get("prompt")).strip()
    prompt_scope = stringify_prompt_value(example.get("prompt_scope")).strip()
    selection = example.get("example_selection") if isinstance(example.get("example_selection"), dict) else {}
    formal_ready = bool(selection.get("formal_finder_ready"))
    if prompt and prompt_scope == "in_context_examples_only" and formal_ready:
        status = "formal_finder_promptpg_examples_ready"
    elif prompt and formal_ready:
        status = "formal_finder_promptpg_ready_legacy_full_prompt"
    elif prompt or selected:
        status = "selected_examples_present_nonformal_or_fallback"
    else:
        status = "smoke_fallback_no_in_context_examples"
    return {
        "status": status,
        "has_full_prompt": bool(prompt),
        "prompt_scope": prompt_scope or None,
        "selected_example_count": len(selected),
        "selection_status": selection.get("selection_status"),
        "formal_finder_ready": formal_ready,
    }


def prompt_contract_for_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    contracts = [prompt_contract_for_example(example) for example in examples]
    if not contracts:
        return {"status": "empty_input", "formal_finder_ready": False}
    return {
        "status": contracts[0]["status"],
        "formal_finder_ready": all(contract["formal_finder_ready"] for contract in contracts),
        "has_full_prompt_rows": sum(1 for contract in contracts if contract["has_full_prompt"]),
        "selected_example_rows": sum(1 for contract in contracts if contract["selected_example_count"] > 0),
        "selected_example_count_first_row": contracts[0]["selected_example_count"],
        "selection_status_first_row": contracts[0].get("selection_status"),
    }


def build_prompt(example: dict[str, Any]) -> str:
    prompt = stringify_prompt_value(example.get("prompt")).strip()
    prompt_scope = stringify_prompt_value(example.get("prompt_scope")).strip()
    if prompt and prompt_scope == "in_context_examples_only":
        return prompt.rstrip() + "\n\n" + build_target_prompt(example, include_code=False)

    selected = selected_examples_from(example)
    if selected:
        parts = [build_target_prompt(item, include_code=True).rstrip() for item in selected]
        parts.append(build_target_prompt(example, include_code=False).rstrip())
        return "\n\n".join(parts) + "\n"
    if prompt:
        return prompt.rstrip() + "\n"
    return build_target_prompt(example, include_code=False)


def extract_last_variable(ans: str) -> str:
    if re.search(r"[\[\(].*[\]\)]", ans):
        ans = re.sub(r"^ans\s*=\s*[\[\(]|[)\]]$", "", ans)
        parts = [part.strip() for part in ans.split(",")]
        return parts[-1]
    return "ans"


def extract_last_question(question: str) -> str:
    parts = [part.strip() for part in re.split(r"[?]", question) if part.strip()]
    return parts[-1] if parts else ""


def choice_field(value: Any, *names: str) -> Any:
    for name in names:
        item = getattr(value, name, None)
        if item not in (None, ""):
            return item
    extra = getattr(value, "model_extra", None)
    if isinstance(extra, dict):
        for name in names:
            item = extra.get(name)
            if item not in (None, ""):
                return item
    if isinstance(value, dict):
        for name in names:
            item = value.get(name)
            if item not in (None, ""):
                return item
    return None


def stringify_choice_content(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return str(value)


_LAST_GENERATION_METADATA: dict[str, Any] = {}


def last_generation_metadata() -> dict[str, Any]:
    """Return metadata for the most recent generation in this process."""
    return dict(_LAST_GENERATION_METADATA)


def response_generation_metadata(result: Any, config: EngineConfig) -> dict[str, Any]:
    usage = getattr(result, "usage", None)
    if hasattr(usage, "model_dump"):
        usage_payload = usage.model_dump()
    elif usage is None:
        usage_payload = None
    else:
        usage_payload = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return {
        "responseId": getattr(result, "id", None),
        "requestedModel": config.model,
        "responseModel": getattr(result, "model", None),
        "finishReasons": [
            getattr(choice, "finish_reason", None)
            for choice in getattr(result, "choices", [])
        ],
        "usage": usage_payload,
        "metadataSupport": "openai_compatible_response",
    }


def parse_choices(result: Any) -> list[str]:
    choices = getattr(result, "choices", [])
    parsed = []
    for choice in choices:
        message = getattr(choice, "message", None)
        content = choice_field(message, "content")
        if content in (None, ""):
            # vLLM reasoning parsers may put Qwen3 thinking output in an
            # OpenAI-extension field while leaving message.content empty.
            content = choice_field(message, "reasoning_content", "reasoning", "reasoning_details")
        if content in (None, ""):
            content = choice_field(choice, "text")
        parsed.append(stringify_choice_content(content))
    return parsed


def generation_profile_params(profile: str) -> tuple[float, float, int]:
    if profile == "greedy":
        return 0.0, 1.0, 1
    if profile == "self_consistency":
        return 0.5, 1.0, 30
    raise ValueError(f"Unsupported generation profile: {profile}")


def base_sampling_policy(profile: str) -> SamplingPolicy:
    temperature, top_p, n = generation_profile_params(profile)
    return SamplingPolicy(
        source=f"profile:{profile}",
        temperature=temperature,
        top_p=top_p,
        n=n,
    )


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def qwen3_6_enable_thinking() -> bool:
    return truthy(os.environ.get("QWEN3_6_ENABLE_THINKING", "1"))


def generator_response_format() -> dict[str, Any] | None:
    mode = (os.environ.get("GENERATOR_RESPONSE_FORMAT") or "").strip().lower()
    if not mode:
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    if mode != "json_schema":
        raise ValueError(
            "GENERATOR_RESPONSE_FORMAT must be empty, json_object, or json_schema"
        )
    raw_path = (os.environ.get("GENERATOR_RESPONSE_SCHEMA_PATH") or "").strip()
    if not raw_path:
        raise ValueError(
            "GENERATOR_RESPONSE_SCHEMA_PATH is required for json_schema"
        )
    schema_path = Path(raw_path)
    if not schema_path.is_absolute():
        schema_path = REPO_ROOT / schema_path
    try:
        document = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load response schema {schema_path}: {exc}") from exc
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("name"), str)
        or not isinstance(document.get("schema"), dict)
    ):
        raise ValueError(
            "response schema must contain string name and object schema"
        )
    return {"type": "json_schema", "json_schema": document}


def mistral4_sampling_policy(n: int) -> SamplingPolicy:
    effort = (os.environ.get("MISTRAL4_REASONING_EFFORT") or "none").strip().lower()
    if effort in {"", "none", "off", "0", "false"}:
        return SamplingPolicy(
            source="local:mistral4:stable_python",
            temperature=0.2,
            top_p=1.0,
            n=n,
            reasoning_effort_intent="none",
        )
    if effort not in {"low", "medium", "high"}:
        raise ValueError("MISTRAL4_REASONING_EFFORT must be one of: none, low, medium, high")
    return SamplingPolicy(
        source=f"local:mistral4:reasoning_{effort}",
        temperature=env_float("MISTRAL4_REASONING_TEMPERATURE", 0.7),
        top_p=env_float("MISTRAL4_REASONING_TOP_P", 1.0),
        n=n,
        reasoning_effort=effort,
        reasoning_effort_intent=effort,
    )


def local_sampling_policy_for_engine(engine: str, n: int) -> SamplingPolicy | None:
    if engine == "qwen3_6":
        return SamplingPolicy(
            source="local:qwen3_6:thinking_precise_coding",
            temperature=env_float("QWEN3_6_TEMPERATURE", 0.6),
            top_p=env_float("QWEN3_6_TOP_P", 0.95),
            n=n,
            top_k=env_int("QWEN3_6_TOP_K", 20),
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            enable_thinking=qwen3_6_enable_thinking(),
        )
    if engine == "deepseek_r1_qwen32b":
        return SamplingPolicy(
            source="local:deepseek_r1_qwen32b:recommended_code",
            temperature=0.6,
            top_p=0.95,
            n=n,
        )
    if engine == "llama3_3":
        return SamplingPolicy(
            source="local:llama3_3:official_default",
            temperature=0.6,
            top_p=0.9,
            n=n,
        )
    if engine == "llama4":
        return SamplingPolicy(
            source="local:llama4:short_context_code",
            temperature=0.6,
            top_p=0.9,
            n=n,
        )
    if engine == "qwythos9b":
        return SamplingPolicy(
            source="local:qwythos9b:recommended_code",
            temperature=0.6,
            top_p=0.95,
            n=n,
            top_k=20,
            repetition_penalty=1.05,
        )
    if engine == "mistral4":
        return mistral4_sampling_policy(n)
    return None


def sampling_policy_for_config(config: EngineConfig, profile: str) -> SamplingPolicy:
    base = base_sampling_policy(profile)
    if config.engine == "gpt5_5":
        effort = (os.environ.get("GPT5_5_REASONING_EFFORT") or "medium").strip().lower()
        if effort not in {"low", "medium", "high"}:
            raise ValueError("GPT5_5_REASONING_EFFORT must be low, medium, or high")
        return SamplingPolicy(
            source=f"gpt5_5:reasoning_{effort}",
            temperature=base.temperature,
            top_p=base.top_p,
            n=base.n,
            reasoning_effort=effort,
            reasoning_effort_intent=effort,
        )
    if config.route in {
        "local_vllm_openai_compatible",
        "local_llama_cpp_openai_compatible",
        "local_sglang_openai_compatible",
    }:
        return local_sampling_policy_for_engine(config.engine, base.n) or base
    return base


def chat_completion_kwargs_for_sampling(config: EngineConfig, policy: SamplingPolicy) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": policy.temperature,
        "top_p": policy.top_p,
        "n": policy.n,
    }
    if policy.presence_penalty is not None:
        kwargs["presence_penalty"] = policy.presence_penalty
    if policy.reasoning_effort is not None and not config.route.startswith("local_"):
        kwargs["reasoning_effort"] = policy.reasoning_effort

    extra_body: dict[str, Any] = {}
    if config.route in {
        "local_vllm_openai_compatible",
        "local_llama_cpp_openai_compatible",
        "local_sglang_openai_compatible",
    }:
        if policy.top_k is not None:
            extra_body["top_k"] = policy.top_k
        if policy.min_p is not None:
            extra_body["min_p"] = policy.min_p
        if policy.repetition_penalty is not None:
            extra_body["repetition_penalty"] = policy.repetition_penalty
        if policy.reasoning_effort is not None:
            if config.route != "local_llama_cpp_openai_compatible" or truthy(os.environ.get("LLAMA_CPP_ALLOW_REASONING_EFFORT")):
                extra_body["reasoning_effort"] = policy.reasoning_effort
        if policy.enable_thinking is not None and config.route == "local_vllm_openai_compatible":
            extra_body["chat_template_kwargs"] = {"enable_thinking": policy.enable_thinking}
    if extra_body:
        kwargs["extra_body"] = extra_body
    run_seed = os.environ.get("EXPERIMENT6_RUN_SEED")
    seed_routes = {
        "local_vllm_openai_compatible",
        "local_llama_cpp_openai_compatible",
        "local_sglang_openai_compatible",
        "chatmock_openai_compatible",
        "closed_api_openai_compatible",
        "closed_api_openai",
        "closed_api_azure_openai",
    }
    if run_seed not in (None, "") and config.route in seed_routes:
        kwargs["seed"] = int(run_seed)
    return kwargs


def effective_max_tokens_for_config(config: EngineConfig, requested_max_tokens: int) -> int:
    if config.engine == "qwythos9b":
        return int(os.environ.get("QWYTHOS_MAX_TOKENS", "8192") or "8192")
    return requested_max_tokens


def request_timeout_seconds_for_config(config: EngineConfig) -> float:
    explicit = os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS")
    if explicit:
        return float(explicit)
    if config.route in {
        "local_vllm_openai_compatible",
        "local_llama_cpp_openai_compatible",
        "local_sglang_openai_compatible",
    }:
        return float(os.environ.get("LOCAL_OPENAI_REQUEST_TIMEOUT_SECONDS", "600") or "600")
    return 120.0


def generator_batch_size_for_config(config: EngineConfig) -> int:
    raw = os.environ.get("GENERATOR_BATCH_SIZE", "1") or "1"
    try:
        requested = int(raw)
    except ValueError as exc:
        raise ValueError(f"GENERATOR_BATCH_SIZE must be an integer, got {raw!r}") from exc
    if requested < 1:
        raise ValueError(f"GENERATOR_BATCH_SIZE must be >= 1, got {requested}")
    if config.route != "local_vllm_openai_compatible":
        return 1
    return requested


TRANSIENT_GENERATION_ERROR_PATTERNS = (
    "upstream error",
    "internalservererror",
    "internal server error",
    "error code: 500",
    "status code: 500",
    "http 500",
    "bad gateway",
    "error code: 502",
    "status code: 502",
    "http 502",
    "service unavailable",
    "error code: 503",
    "status code: 503",
    "http 503",
    "gateway timeout",
    "error code: 504",
    "status code: 504",
    "http 504",
    "connection error",
    "connection reset",
    "read timeout",
    "timed out",
    "timeout",
)


def is_transient_generation_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    if any(pattern in message for pattern in API_AUTH_ERROR_PATTERNS):
        return False
    if any(pattern in message for pattern in API_QUOTA_ERROR_PATTERNS):
        return False
    return any(pattern in message for pattern in TRANSIENT_GENERATION_ERROR_PATTERNS)


def transient_retry_delays() -> list[float]:
    raw = os.environ.get("GENERATOR_TRANSIENT_RETRY_DELAYS", "10,20,40,60,60").strip()
    delays = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            delays.append(float(item))
    return delays


def chat_completion_create_with_retry(client: Any, *, config: EngineConfig, request_kwargs: dict[str, Any]) -> Any:
    delays = transient_retry_delays()
    attempts = int(os.environ.get("GENERATOR_TRANSIENT_MAX_ATTEMPTS", str(len(delays) + 1)) or str(len(delays) + 1))
    attempts = max(1, attempts)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or not is_transient_generation_error(exc):
                raise
            delay = delays[min(attempt - 1, len(delays) - 1)] if delays else 0.0
            print(
                json.dumps(
                    {
                        "time": utc_now(),
                        "stage": "generator_transient_retry",
                        "engine": config.engine,
                        "backend": ROUTE_BACKENDS.get(config.route, config.route),
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "delay_seconds": delay,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if delay > 0:
                sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("chat completion retry failed without exception")


def generate_text(
    config: EngineConfig,
    prompt: str,
    system_prompt: str,
    profile: str,
    max_tokens: int = 128,
) -> list[str]:
    global _LAST_GENERATION_METADATA
    _LAST_GENERATION_METADATA = {}
    if not config.available:
        raise RuntimeError(f"Missing credentials for {config.engine}: {config.missing_credentials}")

    sampling_policy = sampling_policy_for_config(config, profile)
    request_kwargs = chat_completion_kwargs_for_sampling(config, sampling_policy)
    response_format = generator_response_format()
    if response_format is not None:
        request_kwargs["response_format"] = response_format
    request_timeout = request_timeout_seconds_for_config(config)
    max_tokens = effective_max_tokens_for_config(config, max_tokens)
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "0") or "0")

    if config.route in {
        "local_vllm_openai_compatible",
        "local_llama_cpp_openai_compatible",
        "local_sglang_openai_compatible",
        "chatmock_openai_compatible",
        "closed_api_openai_compatible",
    }:
        from openai import OpenAI

        client = OpenAI(
            base_url=config.endpoint,
            api_key=config.api_key or "EMPTY",
            timeout=request_timeout,
            max_retries=max_retries,
        )
    elif config.route == "closed_api_azure_openai":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            timeout=request_timeout,
            max_retries=max_retries,
        )
    elif config.route == "closed_api_openai":
        from openai import OpenAI

        client = OpenAI(api_key=config.api_key, timeout=request_timeout, max_retries=max_retries)
    elif config.route == "codex_cli":
        outputs = generate_text_with_codex_cli(config, prompt, system_prompt, profile)
        _LAST_GENERATION_METADATA = {
            "responseId": None,
            "requestedModel": config.model,
            "responseModel": config.actual_model,
            "finishReasons": [None for _ in outputs],
            "usage": None,
            "metadataSupport": "codex_cli_not_exposed",
        }
        return outputs
    else:
        raise ValueError(f"Unsupported engine route: {config.route}")

    if config.engine == "deepseek_r1_qwen32b":
        messages = [{"role": "user", "content": system_prompt.strip() + "\n\n" + prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    result = chat_completion_create_with_retry(
        client,
        config=config,
        request_kwargs={
            "model": config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            **request_kwargs,
        },
    )
    _LAST_GENERATION_METADATA = response_generation_metadata(result, config)
    _LAST_GENERATION_METADATA["requestResponseFormat"] = (
        response_format["type"] if response_format is not None else None
    )
    return parse_choices(result)


def generator_system_prompt_for_example(example: dict[str, Any] | None = None) -> str:
    if example is not None and example_allows_dev_generator_prompt_guard(example):
        return GENERATOR_CODE_ONLY_SYSTEM_PROMPT
    return "Return only Python code that computes the answer."


def generate_codes(
    config: EngineConfig,
    prompt: str,
    profile: str,
    max_tokens: int = 128,
    example: dict[str, Any] | None = None,
) -> list[str]:
    return generate_text(
        config,
        prompt,
        generator_system_prompt_for_example(example),
        profile,
        max_tokens=max_tokens,
    )


def codex_cli_disabled_features() -> list[str]:
    configured = os.environ.get("CODEX_CLI_DISABLED_FEATURES", "image_generation")
    return [feature.strip() for feature in configured.split(",") if feature.strip()]


def build_codex_cli_command(
    config: EngineConfig,
    codex_path: str,
    output_path: Path,
) -> list[str]:
    command = [codex_path, "exec"]
    service_tier = os.environ.get("CODEX_CLI_SERVICE_TIER")
    if service_tier:
        command.extend(["-c", f"service_tier={json.dumps(service_tier)}"])
    for feature in codex_cli_disabled_features():
        command.extend(["--disable", feature])
    command.extend(
        [
            "--model",
            config.model,
            "--sandbox",
            "read-only",
            "--cd",
            str(REPO_ROOT),
            "--ephemeral",
            "--output-last-message",
            str(output_path),
            "-",
        ]
    )
    return command


def generate_text_with_codex_cli(config: EngineConfig, prompt: str, system_prompt: str, profile: str) -> list[str]:
    if profile != "greedy":
        raise ValueError("Codex CLI route currently supports only the greedy generator profile.")
    codex_path = config.endpoint or shutil.which("codex")
    if not codex_path:
        raise RuntimeError("Codex CLI executable is not available.")
    timeout = int(os.environ.get("CODEX_CLI_TIMEOUT_SECONDS", "180"))
    instructions = system_prompt.strip() + "\n\n" + prompt
    with tempfile.TemporaryDirectory(prefix="fqan_codex_cli_") as tmpdir:
        output_path = Path(tmpdir) / "last_message.txt"
        command = build_codex_cli_command(config, codex_path, output_path)
        env = os.environ.copy()
        if config.api_key:
            env["OPENAI_API_KEY"] = config.api_key
        proc = subprocess.run(
            command,
            input=instructions,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=env,
        )
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise RuntimeError(f"Codex CLI failed with exit code {proc.returncode}: {message}")
        if output_path.exists():
            content = output_path.read_text(encoding="utf-8").strip()
        else:
            content = proc.stdout.strip()
    return [content]


def generate_codes_with_codex_cli(
    config: EngineConfig,
    prompt: str,
    profile: str,
    example: dict[str, Any] | None = None,
) -> list[str]:
    return generate_text_with_codex_cli(
        config,
        prompt,
        generator_system_prompt_for_example(example),
        profile,
    )


def strip_reasoning_and_fences(code: str) -> str:
    text = (code or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    if text.lower().startswith("<think>"):
        return ""

    fenced = re.findall(r"```[ \t]*([^`\n]*)\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        python_blocks = [
            body.strip()
            for label, body in fenced
            if (label.strip().lower().split() or [""])[0] in {"python", "py"} and body.strip()
        ]
        fallback_blocks = [body.strip() for _, body in fenced if body.strip()]
        text = (python_blocks or fallback_blocks)[-1]
    text = text.replace("```python", "").replace("```", "").replace("&", "_")
    return text.strip()


def extract_python_lines_from_mixed_text(text: str) -> str:
    """Recover clear PoT lines when a local thinking model wraps code in prose."""
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", line).strip()
        line = line.strip("`")
        if re.match(r"^(?:step_\d+|ans|answer|result)\s*=", line):
            lines.append(line)
            continue
        if is_single_python_assignment_line(line):
            lines.append(line)
            continue
        if re.match(r"^print\s*\(.+\)\s*$", line):
            lines.append(line)
            continue
        if re.fullmatch(r"[-+*/()., 0-9]+", line) and re.search(r"\d", line) and re.search(r"[-+*/]", line):
            lines.append(line)
    return "\n".join(lines).strip()


def is_single_python_assignment_line(line: str) -> bool:
    try:
        tree = ast.parse(line, mode="exec")
    except SyntaxError:
        return False
    if len(tree.body) != 1:
        return False
    statement = tree.body[0]
    if isinstance(statement, ast.Assign):
        return bool(statement.targets) and all(isinstance(target, ast.Name) for target in statement.targets)
    return False


def last_assignment_target(tree: ast.Module) -> str | None:
    for statement in reversed(tree.body):
        if isinstance(statement, ast.Assign):
            for target in reversed(statement.targets):
                if isinstance(target, ast.Name):
                    return target.id
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            return statement.target.id
        if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            return statement.target.id
    return None


def module_assigns_ans(tree: ast.Module) -> bool:
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "ans" for target in statement.targets):
                return True
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.target.id == "ans":
            return True
    return False


def ensure_ans_assignment(code: str) -> str:
    normalized = normalize_dsl_tokens(code).strip()
    if not normalized:
        return ""
    if looks_like_finqa_dsl(normalized) and "=" not in normalized and "\n" not in normalized:
        try:
            return pythonize_finqa_program(normalized)
        except Exception:
            pass
    try:
        tree = ast.parse(normalized, mode="exec")
    except SyntaxError:
        if looks_like_finqa_dsl(normalized):
            try:
                return pythonize_finqa_program(normalized)
            except Exception:
                return normalized
        return normalized
    if module_assigns_ans(tree):
        return normalized
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        expr = tree.body[-1].value
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "print" and expr.args:
            printed_expr = ast_source(expr.args[0], normalized)
            return normalized + f"\nans = {printed_expr}"
        return normalized + f"\nans = {ast_source(expr, normalized)}"
    target = last_assignment_target(tree)
    if target:
        return normalized + f"\nans = {target}"
    return normalized


def make_execution_locals(code: str) -> dict[str, Any]:
    import sympy

    def add(a: Any, b: Any) -> Any:
        return a + b

    def subtract(a: Any, b: Any) -> Any:
        return a - b

    def multiply(a: Any, b: Any) -> Any:
        return a * b

    def divide(a: Any, b: Any) -> Any:
        return a / b

    def greater(a: Any, b: Any) -> bool:
        return a > b

    def exp(a: Any, b: Any) -> Any:
        return a**b

    def table_sum(values: Any, *rest: Any) -> Any:
        items = list(values) if not rest and isinstance(values, (list, tuple)) else [values, *rest]
        return sum(items)

    def table_average(values: Any, *rest: Any) -> Any:
        items = list(values) if not rest and isinstance(values, (list, tuple)) else [values, *rest]
        return sum(items) / len(items)

    printed: list[Any] = []

    def capture_print(*values: Any, **_: Any) -> None:
        if values:
            printed.append(values[-1])

    locals_: dict[str, Any] = {
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
        "greater": greater,
        "exp": exp,
        "table_sum": table_sum,
        "table_average": table_average,
        "table_max": max,
        "table_min": min,
        "sum": sum,
        "max": max,
        "min": min,
        "len": len,
        "abs": abs,
        "round": round,
        "float": float,
        "int": int,
        "pow": pow,
        "sympy": sympy,
        "sp": sympy,
        "print": capture_print,
        "_printed_values": printed,
    }
    for token in set(re.findall(r"\bconst_(?:m?\d+)\b", code)):
        locals_[token] = int(const_token_to_python(token).strip("()"))
    return locals_


GENERATED_CODE_EXECUTION_ENV = "FQAN_ALLOW_GENERATED_CODE_EXECUTION"


def execute_python_code(code: str, key: str = "ans") -> Any:
    if not truthy(os.environ.get(GENERATED_CODE_EXECUTION_ENV)):
        raise RuntimeError(
            "generated Python execution is disabled; set "
            f"{GENERATED_CODE_EXECUTION_ENV}=1 only in an isolated research environment"
        )
    locals_ = make_execution_locals(code)
    globals_ = {"__builtins__": {}}

    def execute() -> Any:
        try:
            exec(code, globals_, locals_)
        except Exception:
            return None
        if key in locals_:
            return locals_.get(key)
        printed = locals_.get("_printed_values") or []
        if printed:
            return printed[-1]
        return locals_.get("ans")

    try:
        return func_timeout.func_timeout(5, execute)
    except func_timeout.FunctionTimedOut:
        return None


FINDER_PERCENT_SCALE_PATTERNS = ("* 100 ", "* 100\n", "*100 ", "*100\n")


def example_allows_extended_percent_scale_detection(example: dict[str, Any]) -> bool:
    values = [
        example.get("source"),
        example.get("target_gold_csv"),
        example.get("retrieved_source"),
        example.get("flow_scope"),
    ]
    haystack = " ".join(stringify_prompt_value(value) for value in values).replace("\\", "/").lower()
    if "data/testing/finqa_10_rel_fact_instruction.csv" in haystack:
        return True
    if "finqa_dev" in haystack or "data/src/finqa/dev.json" in haystack:
        return True
    return False


def generated_code_has_percent_scale_multiply(code: str, example: dict[str, Any], raw_code: str | None = None) -> bool:
    del raw_code
    finder_inputs = [code]
    if any(pattern in candidate for candidate in finder_inputs for pattern in FINDER_PERCENT_SCALE_PATTERNS):
        return True
    if example_allows_extended_percent_scale_detection(example):
        return bool(re.search(r"\*\s*100(?:\.0+)?(?:\s|$)", code or ""))
    return False


def normalize_generated_code(code: str) -> str:
    stripped = strip_reasoning_and_fences(code)
    try:
        ast.parse(stripped, mode="exec")
    except SyntaxError:
        extracted = extract_python_lines_from_mixed_text(stripped)
        if extracted:
            stripped = extracted
    return ensure_ans_assignment(stripped)


def execute_codes(codes: list[str], example: dict[str, Any]) -> tuple[Any, list[str]]:
    counter: Counter[Any] = Counter()
    cleaned_codes = []
    for code in codes:
        raw_code = code
        code = normalize_generated_code(code)
        cleaned_codes.append(code)
        ans = execute_python_code(code, "ans") if code else None
        ans = floatify_ans(ans)
        if generated_code_has_percent_scale_multiply(code, example, raw_code):
            qn = extract_last_question(str(example.get("question", "")))
            if "percent" in qn or "percentage" in qn or "growth rate" in qn:
                try:
                    ans = ans / 100
                except Exception:
                    pass
        if ans is not None:
            counter.update([ans])
    prediction = counter.most_common(1)[0][0] if counter else None
    if isinstance(prediction, bool):
        prediction = "yes" if prediction else "no"
    elif isinstance(prediction, list):
        prediction = prediction[0] if prediction else None
    return prediction, cleaned_codes


class GenerationInterrupted(RuntimeError):
    def __init__(self, original_exception: Exception, example_index: int, completed_rows_before_failure: int):
        super().__init__(str(original_exception))
        self.original_exception = original_exception
        self.example_index = example_index
        self.completed_rows_before_failure = completed_rows_before_failure
        self.category = classify_generation_exception(original_exception)


def exception_search_text(exc: Exception) -> str:
    parts = [exc.__class__.__name__, str(exc)]
    for attr in ["code", "type", "status_code", "body", "response"]:
        value = getattr(exc, attr, None)
        if value is not None:
            parts.append(str(value))
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None) if response is not None else None
    if response_status is not None:
        parts.append(f"HTTP {response_status}")
    return " ".join(parts).lower()


def classify_generation_exception(exc: Exception) -> str:
    if isinstance(exc, ResumeOutputMismatch):
        return "resume_output_mismatch"
    if isinstance(exc, OutputLockConflict):
        return "output_lock_conflict"
    text = exception_search_text(exc)
    if any(pattern in text for pattern in API_AUTH_ERROR_PATTERNS):
        return "api_authentication_error"
    if any(pattern in text for pattern in API_QUOTA_ERROR_PATTERNS):
        return "api_quota_or_rate_limit"
    return "generator_runtime_error"


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as reader:
        return sum(1 for line in reader if line.strip())




class ResumeOutputMismatch(RuntimeError):
    """Raised before generation when an existing JSONL is not the input prefix."""


class OutputLockConflict(RuntimeError):
    """Raised when another process already owns the generator output."""


def normalize_resume_identity(key: str, value: Any) -> str:
    if key == "question":
        return " ".join(str(value).split()).casefold()
    if key == "source_csv_row":
        try:
            return str(int(value))
        except (TypeError, ValueError):
            pass
    return str(value).strip()


def resume_identity_mismatches(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    compared = 0
    for key in ("selection_key", "id", "source_csv_row", "question"):
        expected_value = expected.get(key)
        if expected_value in (None, ""):
            continue
        compared += 1
        actual_value = actual.get(key)
        if actual_value in (None, ""):
            mismatches.append(f"missing_{key}")
            continue
        if normalize_resume_identity(key, actual_value) != normalize_resume_identity(key, expected_value):
            mismatches.append(f"mismatched_{key}")
    if compared == 0:
        mismatches.append("input_row_has_no_stable_identity")
    return mismatches


def validate_resume_output_prefix(
    output_jsonl: Path,
    examples: list[dict[str, Any]],
) -> int:
    """Return the validated prefix length; reject corrupt, duplicate, or extra rows."""
    if not output_jsonl.exists():
        return 0
    mismatches: list[dict[str, Any]] = []
    output_index = 0
    with output_jsonl.open("r", encoding="utf-8") as reader:
        for physical_line, line in enumerate(reader, start=1):
            if not line.strip():
                mismatches.append({"line": physical_line, "reason": "blank_line"})
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                mismatches.append(
                    {"line": physical_line, "reason": "invalid_json", "detail": str(exc)}
                )
                output_index += 1
                continue
            if not isinstance(item, dict):
                mismatches.append({"line": physical_line, "reason": "row_not_object"})
                output_index += 1
                continue
            if output_index >= len(examples):
                mismatches.append({"line": physical_line, "reason": "extra_output_row"})
                output_index += 1
                continue
            identity_errors = resume_identity_mismatches(examples[output_index], item)
            if identity_errors:
                mismatches.append(
                    {
                        "line": physical_line,
                        "input_index": output_index,
                        "reason": "identity_mismatch",
                        "identity_errors": identity_errors,
                    }
                )
            output_index += 1
    if mismatches:
        preview = json.dumps(mismatches[:10], ensure_ascii=False, separators=(",", ":"))
        raise ResumeOutputMismatch(
            f"Existing output is not an exact prefix of the current input: {preview}. "
            "Repair into a fresh output path before resuming."
        )
    return output_index


@contextmanager
def exclusive_output_lock(output_jsonl: Path) -> Iterator[None]:
    lock_path = output_jsonl.with_name(f"{output_jsonl.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OutputLockConflict(
                f"Another generator process owns output lock: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def command_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_resume_command(args: argparse.Namespace) -> str:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        os.environ.get("CONDA_ENV", DEFAULT_CONDA_ENV),
        "python",
        "-B",
        "new_full_finqa_run.py",
        "--engine",
        args.engine,
        "--input-json",
        command_path(args.input_json),
        "--output-jsonl",
        command_path(args.output_jsonl),
        "--profile",
        args.profile,
        "--limit",
        str(args.limit),
        "--max-tokens",
        str(args.max_tokens),
        "--sleep-seconds",
        str(args.sleep_seconds),
        "--credential-purpose",
        "execute",
        "--execute",
        "--resume-output",
    ]
    if args.status_json is not None:
        command.extend(["--status-json", command_path(args.status_json)])
    return f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(command)}"


def build_resume_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "command": build_resume_command(args),
        "notes": [
            "Refresh quota, fix authentication, replace the API key, or restart the ChatMock server before rerunning.",
            "--resume-output skips existing output_jsonl rows; keep the same input order.",
            "interrupted_example_index is zero-based.",
        ],
    }


def local_vllm_runtime_public_dict() -> dict[str, Any]:
    names = [
        "VLLM_RUNTIME_PROFILE",
        "VLLM_TENSOR_PARALLEL_SIZE",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_GPU_MEMORY_UTILIZATION",
        "VLLM_DTYPE",
        "VLLM_KV_CACHE_DTYPE",
        "VLLM_QUANTIZATION",
        "VLLM_LOAD_FORMAT",
        "VLLM_CPU_OFFLOAD_GB",
        "VLLM_OFFLOAD_BACKEND",
        "VLLM_OFFLOAD_GROUP_SIZE",
        "VLLM_OFFLOAD_NUM_IN_GROUP",
        "VLLM_OFFLOAD_PREFETCH_STEP",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_MOE_BACKEND",
        "VLLM_LINEAR_BACKEND",
        "VLLM_ENABLE_EXPERT_PARALLEL",
        "VLLM_LANGUAGE_MODEL_ONLY",
        "VLLM_REASONING_PARSER",
        "VLLM_TOOL_CALL_PARSER",
        "VLLM_ENABLE_AUTO_TOOL_CHOICE",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_FLASH_ATTN_OPS_SHIM",
        "VLLM_SWAP_SPACE",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_KV_CACHE_MEMORY_BYTES",
        "VLLM_ENABLE_CHUNKED_PREFILL",
        "VLLM_CPU_OFFLOAD_PARAMS",
        "VLLM_OFFLOAD_PARAMS",
        "VLLM_SAFETENSORS_LOAD_STRATEGY",
        "VLLM_MAX_PARALLEL_LOADING_WORKERS",
        "VLLM_DISABLE_CUSTOM_ALL_REDUCE",
        "VLLM_ENFORCE_EAGER",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
    ]
    values = {name: os.environ.get(name) for name in names if os.environ.get(name) is not None}
    profile = values.get("VLLM_RUNTIME_PROFILE")
    return {
        "profile": profile,
        "values": values,
        "precision_policy": local_vllm_precision_policy(profile),
    }


def local_llama_cpp_runtime_public_dict() -> dict[str, Any]:
    names = [
        "MISTRAL_SMALL_RUNTIME_BACKEND",
        "LLAMA3_3_RUNTIME_BACKEND",
        "LLAMA_CPP_FORMAL_MODEL",
        "LLAMA_CPP_BASE_URL",
        "LLAMA_CPP_MODEL_PATH",
        "LLAMA_CPP_MODEL_ALIAS",
        "LLAMA_CPP_QUANT",
        "LLAMA_CPP_CTX_SIZE",
        "LLAMA_CPP_N_GPU_LAYERS",
        "LLAMA_CPP_TENSOR_SPLIT",
        "LLAMA_CPP_SPLIT_MODE",
        "LLAMA_CPP_PARALLEL",
        "LLAMA_CPP_BATCH_SIZE",
        "LLAMA_CPP_UBATCH_SIZE",
        "LLAMA_CPP_THREADS",
        "LLAMA_CPP_CACHE_TYPE_K",
        "LLAMA_CPP_CACHE_TYPE_V",
        "LLAMA_CPP_KV_OFFLOAD",
        "LLAMA_CPP_CPU_MOE",
        "LLAMA_CPP_N_CPU_MOE",
        "LLAMA_CPP_FIT_TARGET",
        "LLAMA_CPP_FIT_CTX",
        "LLAMA_CPP_OP_OFFLOAD",
        "LLAMA_CPP_FLASH_ATTN",
        "LLAMA_CPP_CACHE_RAM",
        "LLAMA_CPP_DEVICE",
        "LLAMA_CPP_MAIN_GPU",
        "LLAMA_CPP_NO_MMAP",
    ]
    values = {name: os.environ.get(name) for name in names if os.environ.get(name) is not None}
    quant = values.get("LLAMA_CPP_QUANT")
    return {
        "backend": "local_llama_cpp_openai_compatible",
        "profile": f"llama_cpp_{quant}" if quant else "llama_cpp",
        "values": values,
        "precision_policy": (
            "GGUF local feasibility route through llama.cpp OpenAI-compatible server; "
            "quantized inference-only backend, not a vLLM formal success."
        ),
    }



def local_sglang_runtime_public_dict() -> dict[str, Any]:
    names = [
        "MISTRAL_SMALL_RUNTIME_BACKEND",
        "LLAMA3_3_RUNTIME_BACKEND",
        "SGLANG_LLAMA_RUNTIME_BACKEND",
        "SGLANG_BASE_URL",
        "SGLANG_MODEL_PATH",
        "SGLANG_SERVED_MODEL_NAME",
        "SGLANG_FORMAL_MODEL",
        "SGLANG_RUNTIME_PROFILE",
        "SGLANG_QUANTIZATION",
        "SGLANG_LOAD_FORMAT",
        "SGLANG_TP",
        "SGLANG_CONTEXT_LENGTH",
        "SGLANG_MEM_FRACTION_STATIC",
        "SGLANG_CHUNKED_PREFILL_SIZE",
        "SGLANG_MAX_RUNNING_REQUESTS",
        "SGLANG_DTYPE",
        "SGLANG_KV_CACHE_DTYPE",
        "SGLANG_ATTENTION_BACKEND",
        "SGLANG_SAMPLING_BACKEND",
    ]
    values = {name: os.environ.get(name) for name in names if os.environ.get(name) is not None}
    return {
        "backend": "local_sglang_openai_compatible",
        "profile": values.get("SGLANG_RUNTIME_PROFILE") or "sglang",
        "values": values,
        "precision_policy": (
            "SGLang OpenAI-compatible target-answer route. It is inference-only; "
            "record success separately from vLLM or llama.cpp profiles."
        ),
    }

def local_vllm_precision_policy(profile: str | None) -> str | None:
    policies = {
        "qwen_fp8_tp2_precise_kv": "Qwen FP8 checkpoint for vLLM 0.22.1 with TP2, short context, language-model-only, qwen3 reasoning parser, and FlashInfer sampler disabled for CUDA 13 compatibility.",
        "qwen_fp8_low_memory": "Qwen FP8 checkpoint with FP8 KV cache for local memory pressure.",
        "qwen_fp8_tp1_cpu_offload": "Qwen FP8 TP1 fallback with FP8 KV cache and prefetch offload for local memory pressure.",
        "mistral_nvfp4_lmo_ep_offload4": "Mistral Small 4 official NVFP4 exact probe with text-only mode, TP2, expert parallel, 4GiB CPU offload, FP8 KV cache, and short context; feasibility route, not a full formal result.",
        "mistral_nvfp4_lmo_offload8": "Mistral Small 4 official NVFP4 exact probe with text-only mode, TP2, 8GiB CPU offload, lower GPU utilization, FP8 KV cache, and short context; feasibility route, not a full formal result.",
        "mistral_nvfp4_lmo_moe_emulation": "Mistral Small 4 official NVFP4 exact probe with text-only mode, TP2, 8GiB CPU offload, and emulated MoE backend to avoid Marlin scale-conversion OOM; feasibility route, not a full formal result.",
        "mistral_nvfp4_memreserve": "Mistral Small 4 official NVFP4 checkpoint with TP2, short context, FP8 KV cache, and prefetch offload; feasibility route, not a full-precision claim.",
        "mistral_nvfp4_prefetch": "Mistral Small 4 official NVFP4 checkpoint with TP2 and prefetch offload; quantized local route, not a full-precision claim.",
        "mistral_auto_tp2_short_context": "Legacy Mistral base-checkpoint probe; known impractical on dual RTX A4500.",
        "mistral_fp8_offload": "Mistral Small 4 FP8/offload local smoke fallback; not a full-precision formal result.",
        "llama3_3_bitsandbytes_offload": "Llama 3.3 official checkpoint bitsandbytes feasibility profile with TP2, short context, FP8 KV cache, eager safetensors loading, and CPU offload.",
        "llama3_3_awq_prefetch": "Llama 3.3 official checkpoint with TP2 and prefetch offload.",
        "llama3_3_fp8_no_cpu_offload": "Llama 3.3 official checkpoint FP8 feasibility profile without UVA CPU offload.",
        "llama3_3_prefetch_fp8": "Llama 3.3 official checkpoint FP8 feasibility profile with prefetch offload and no UVA offloader.",
        "llama3_3_fp8_offload": "Legacy online-FP8 plus UVA offload profile; incompatible with vLLM 0.19.1 meta-tensor loading.",
    }
    return policies.get(profile) if profile else None


def route_execution_status(config: EngineConfig) -> str:
    if config.available:
        return "ready"
    missing = set(config.missing_credentials)
    if (
        "VLLM_BASE_URL" in missing
        or "CHATMOCK_BASE_URL" in missing
        or "LLAMA_CPP_BASE_URL" in missing
        or "SGLANG_BASE_URL" in missing
    ):
        return "runtime_blocked"
    if "LLAMA_CPP_MODEL_PATH" in missing or "SGLANG_MODEL_PATH" in missing:
        return "runtime_blocked"
    if any("CODEX_CLI_PATH" in item or "codex on PATH" in item for item in missing):
        return "runtime_blocked"
    return "credential_blocked"


def load_examples(path: Path, limit: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if limit >= 0:
        return data[:limit]
    return data


def generated_text_nonempty(raw_generated: Any) -> bool:
    if isinstance(raw_generated, list):
        return any(str(item).strip() for item in raw_generated)
    return bool(str(raw_generated or "").strip())


def score_existing_output_rows(output_jsonl: Path, examples: list[dict[str, Any]], rows: int) -> dict[str, int]:
    correct = 0
    wrong = 0
    percentage_equivalent_correct = 0
    generated_nonempty_rows = 0
    executed_non_null_rows = 0
    if rows <= 0 or not output_jsonl.is_file():
        return {
            "rows": 0,
            "correct": 0,
            "wrong": 0,
            "percentage_equivalent_correct": 0,
            "generated_nonempty_rows": 0,
            "executed_non_null_rows": 0,
        }
    with output_jsonl.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= rows:
                break
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                wrong += 1
                continue
            generated_nonempty_rows += int(generated_text_nonempty(item.get("generated")))
            prediction = item.get("executed")
            executed_non_null_rows += int(prediction is not None)
            answer = item.get("answer")
            if answer is None and index < len(examples):
                answer = examples[index].get("answer")
            if prediction is not None and finqa_equal(prediction, answer, False):
                correct += 1
            else:
                wrong += 1
            if prediction is not None and finqa_equal(prediction, answer, True):
                percentage_equivalent_correct += 1
    return {
        "rows": correct + wrong,
        "correct": correct,
        "wrong": wrong,
        "percentage_equivalent_correct": percentage_equivalent_correct,
        "generated_nonempty_rows": generated_nonempty_rows,
        "executed_non_null_rows": executed_non_null_rows,
    }


def score_result_payload(
    *,
    scores: dict[str, int],
    output_jsonl: Path,
    total_output_rows: int,
    skipped_rows: int = 0,
    resume_existing_output_rows: int = 0,
    sampling_policy: dict[str, Any] | None = None,
    score_mode: str,
) -> dict[str, Any]:
    total = int(scores["rows"])
    generated_nonempty_rows = int(scores["generated_nonempty_rows"])
    executed_non_null_rows = int(scores["executed_non_null_rows"])
    percentage_equivalent_correct = int(scores["percentage_equivalent_correct"])
    extraction_failed = total > 0 and executed_non_null_rows == 0
    raw_execution_accuracy = int(scores["correct"]) / total if total else None
    percentage_equivalent_accuracy = percentage_equivalent_correct / total if total else None
    payload: dict[str, Any] = {
        "rows": total,
        "skipped_rows": skipped_rows,
        "resume_existing_output_rows": resume_existing_output_rows,
        "total_output_rows": total_output_rows,
        "generated_nonempty_rows": generated_nonempty_rows,
        "generated_nonempty_rate": generated_nonempty_rows / total if total else None,
        "executed_non_null_rows": executed_non_null_rows,
        "executed_non_null_rate": executed_non_null_rows / total if total else None,
        "correct": int(scores["correct"]),
        "wrong": int(scores["wrong"]),
        "percentage_equivalent_correct": percentage_equivalent_correct,
        "percentage_equivalent_accuracy": percentage_equivalent_accuracy,
        "percentage_equivalent_metric_scope": "diagnostic_only",
        "percentage_equivalent_metric_note": PERCENTAGE_EQUIVALENT_DIAGNOSTIC_NOTE,
        "raw_execution_accuracy": raw_execution_accuracy,
        "execution_accuracy": None if extraction_failed else raw_execution_accuracy,
        "failure_category": "execute_extraction_failed" if extraction_failed else None,
        "execution_status": "execute_extraction_failed" if extraction_failed else "scored",
        "output_jsonl": str(output_jsonl),
        "score_mode": score_mode,
        "score_policy": {
            "formal_execution_accuracy": "strict_finder",
            "formal_include_percentage": False,
            "diagnostic_percentage_equivalent_include_percentage": True,
        },
    }
    if sampling_policy is not None:
        payload["sampling_policy"] = sampling_policy
    return payload


def score_existing_output(input_json: Path, output_jsonl: Path, limit: int) -> dict[str, Any]:
    examples = load_examples(input_json, limit)
    existing_output_rows = count_jsonl_rows(output_jsonl) if output_jsonl.is_file() else 0
    rows_to_score = min(existing_output_rows, len(examples))
    scores = score_existing_output_rows(output_jsonl, examples, rows_to_score)
    return score_result_payload(
        scores=scores,
        output_jsonl=output_jsonl,
        total_output_rows=existing_output_rows,
        skipped_rows=rows_to_score,
        resume_existing_output_rows=existing_output_rows,
        score_mode="score_existing_output",
    )


def _run_generation_locked(
    config: EngineConfig,
    input_json: Path,
    output_jsonl: Path,
    profile: str,
    limit: int,
    sleep_seconds: float,
    max_tokens: int,
    resume_output: bool,
) -> dict[str, Any]:
    examples = load_examples(input_json, limit)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing_output_rows = validate_resume_output_prefix(output_jsonl, examples) if resume_output else 0
    skipped_rows = min(existing_output_rows, len(examples))
    existing_scores = score_existing_output_rows(output_jsonl, examples, skipped_rows) if resume_output else {
        "rows": 0,
        "correct": 0,
        "wrong": 0,
        "percentage_equivalent_correct": 0,
        "generated_nonempty_rows": 0,
        "executed_non_null_rows": 0,
    }
    correct = int(existing_scores["correct"])
    wrong = int(existing_scores["wrong"])
    percentage_equivalent_correct = int(existing_scores["percentage_equivalent_correct"])
    generated_nonempty_rows = int(existing_scores["generated_nonempty_rows"])
    executed_non_null_rows = int(existing_scores["executed_non_null_rows"])
    sampling_policy = sampling_policy_for_config(config, profile).to_public_dict()
    generator_batch_size = generator_batch_size_for_config(config)
    output_mode = "a" if resume_output else "w"

    def generate_one(example: dict[str, Any]) -> tuple[Any, list[str]]:
        prompt = build_prompt(example)
        codes = generate_codes(config, prompt, profile, max_tokens=max_tokens, example=example)
        return execute_codes(codes, example)

    with output_jsonl.open(output_mode, encoding="utf-8") as writer:
        for batch_start in range(skipped_rows, len(examples), generator_batch_size):
            batch_end = min(batch_start + generator_batch_size, len(examples))
            batch = [(index, examples[index]) for index in range(batch_start, batch_end)]
            if generator_batch_size == 1:
                example_index, example = batch[0]
                try:
                    results = [generate_one(example)]
                except Exception as exc:
                    raise GenerationInterrupted(exc, example_index, correct + wrong) from exc
            else:
                with ThreadPoolExecutor(max_workers=generator_batch_size) as executor:
                    futures = [executor.submit(generate_one, example) for _, example in batch]
                    results = []
                    for (example_index, _), future in zip(batch, futures):
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            for pending in futures:
                                pending.cancel()
                            raise GenerationInterrupted(exc, example_index, correct + wrong) from exc

            for (_, example), (prediction, cleaned_codes) in zip(batch, results):
                answer = example.get("answer")
                if generated_text_nonempty(cleaned_codes):
                    generated_nonempty_rows += 1
                if prediction is not None:
                    executed_non_null_rows += 1
                if prediction is not None and finqa_equal(prediction, answer, False):
                    correct += 1
                else:
                    wrong += 1
                if prediction is not None and finqa_equal(prediction, answer, True):
                    percentage_equivalent_correct += 1
                total = correct + wrong
                item = dict(example)
                item.update(
                    {
                        "generator_requested_engine": config.requested_engine,
                        "generator_engine": config.engine,
                        "generator_backend": ROUTE_BACKENDS.get(config.route, config.route),
                        "generator_formal_model": config.formal_model,
                        "generator_actual_model": config.actual_model,
                        "generator_runtime_profile": config.runtime_profile,
                        "generator_sampling_policy": sampling_policy,
                        "generator_batch_size": generator_batch_size,
                        "generated": cleaned_codes,
                        "executed": prediction,
                        "accuracy": correct / total if total else None,
                        "execution_diagnostics": {
                            "generated_nonempty": generated_text_nonempty(cleaned_codes),
                            "executed_non_null": prediction is not None,
                        },
                        "generator_prompt_contract": prompt_contract_for_example(example),
                    }
                )
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                writer.flush()
                if sleep_seconds:
                    sleep(sleep_seconds)
    scores = {
        "rows": correct + wrong,
        "correct": correct,
        "wrong": wrong,
        "percentage_equivalent_correct": percentage_equivalent_correct,
        "generated_nonempty_rows": generated_nonempty_rows,
        "executed_non_null_rows": executed_non_null_rows,
    }
    payload = score_result_payload(
        scores=scores,
        output_jsonl=output_jsonl,
        total_output_rows=skipped_rows + (int(scores["rows"]) - int(existing_scores["rows"])),
        skipped_rows=skipped_rows,
        resume_existing_output_rows=existing_output_rows,
        sampling_policy=sampling_policy,
        score_mode="generation",
    )
    payload["generator_batch_size"] = generator_batch_size
    return payload


def run_generation(
    config: EngineConfig,
    input_json: Path,
    output_jsonl: Path,
    profile: str,
    limit: int,
    sleep_seconds: float,
    max_tokens: int,
    resume_output: bool,
) -> dict[str, Any]:
    with exclusive_output_lock(output_jsonl):
        return _run_generation_locked(
            config=config,
            input_json=input_json,
            output_jsonl=output_jsonl,
            profile=profile,
            limit=limit,
            sleep_seconds=sleep_seconds,
            max_tokens=max_tokens,
            resume_output=resume_output,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate FinQA generator routes.")
    parser.add_argument("--engine", default="mistral4")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument(
        "--status-json",
        type=Path,
        default=None,
        help="Optional path for the validation/execution status payload.",
    )
    parser.add_argument("--profile", choices=["greedy", "self_consistency"], default="greedy")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--resume-output",
        action="store_true",
        help="Skip rows already present in output-jsonl before appending new generations.",
    )
    parser.add_argument(
        "--credential-purpose",
        choices=["auto", "test", "execute"],
        default="auto",
        help="GPT-4.1 credential policy: test allows OpenAI-compatible endpoints; execute requires Azure/OpenAI unless explicitly overridden.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument(
        "--score-existing-output",
        action="store_true",
        help="Recompute strict FINDER EA and diagnostic percentage-equivalent EA from output-jsonl without calling an LLM.",
    )
    return parser.parse_args()


def write_status_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    credential_purpose = args.credential_purpose
    if credential_purpose == "auto":
        credential_purpose = "execute" if args.execute else "test"
    config = resolve_engine(args.engine, credential_purpose=credential_purpose)
    exists = args.input_json.exists()
    payload: dict[str, Any] = {
        "requested_engine": args.engine,
        "normalized_engine": config.engine,
        "backend": ROUTE_BACKENDS.get(config.route, config.route),
        "formal_model": config.formal_model,
        "actual_model": config.actual_model,
        "runtime_profile": config.runtime_profile,
        "engine": config.to_public_dict(),
        "input_json": str(args.input_json),
        "input_exists": exists,
        "output_jsonl": str(args.output_jsonl),
        "profile": args.profile,
        "sampling_policy": sampling_policy_for_config(config, args.profile).to_public_dict(),
        "request_timeout_seconds": request_timeout_seconds_for_config(config),
        "limit": args.limit,
        "max_tokens": args.max_tokens,
        "resume_output": args.resume_output,
        "score_existing_output": args.score_existing_output,
        "credential_purpose": credential_purpose,
        "credential_purpose_requested": args.credential_purpose,
        "route_status": route_execution_status(config),
    }
    if config.route == "local_vllm_openai_compatible":
        payload["local_vllm_runtime"] = local_vllm_runtime_public_dict()
    if config.route == "local_llama_cpp_openai_compatible":
        payload["local_llama_cpp_runtime"] = local_llama_cpp_runtime_public_dict()
    if config.route == "local_sglang_openai_compatible":
        payload["local_sglang_runtime"] = local_sglang_runtime_public_dict()
    if exists:
        try:
            preview_examples = load_examples(args.input_json, 1)
            payload["prompt_contract"] = prompt_contract_for_examples(preview_examples)
        except Exception as exc:
            payload["prompt_contract"] = {
                "status": "unreadable_input",
                "formal_finder_ready": False,
                "error": str(exc),
            }
    if exists and args.show_prompt:
        examples = load_examples(args.input_json, 1)
        payload["prompt_preview"] = build_prompt(examples[0]) if examples else ""
    if args.score_existing_output:
        exit_code = 0
        if not exists:
            payload["error"] = {
                "type": "missing_input_json",
                "category": "missing_input",
                "message": f"Missing input JSON: {args.input_json}",
            }
            exit_code = 2
        elif not args.output_jsonl.is_file():
            payload["error"] = {
                "type": "missing_output_jsonl",
                "category": "missing_output",
                "message": f"Missing output JSONL: {args.output_jsonl}",
            }
            exit_code = 2
        else:
            payload["result"] = score_existing_output(
                input_json=args.input_json,
                output_jsonl=args.output_jsonl,
                limit=args.limit,
            )
            payload["note"] = "Score existing output only. No LLM request was made."
    elif args.execute:
        exit_code = 0
        if not exists:
            payload["error"] = {
                "type": "missing_input_json",
                "category": "missing_input",
                "message": f"Missing input JSON: {args.input_json}",
            }
            exit_code = 2
        elif not config.available:
            payload["error"] = {
                "type": "missing_credentials",
                "category": route_execution_status(config),
                "message": f"Missing credentials for {config.engine}: {config.missing_credentials}",
                "missing_credentials": config.missing_credentials,
            }
            exit_code = 2
        else:
            try:
                payload["result"] = run_generation(
                    config=config,
                    input_json=args.input_json,
                    output_jsonl=args.output_jsonl,
                    profile=args.profile,
                    limit=args.limit,
                    sleep_seconds=args.sleep_seconds,
                    max_tokens=args.max_tokens,
                    resume_output=args.resume_output,
                )
            except GenerationInterrupted as exc:
                original = exc.original_exception
                error: dict[str, Any] = {
                    "type": original.__class__.__name__,
                    "message": str(original),
                    "category": exc.category,
                    "interrupted_example_index": exc.example_index,
                    "completed_rows_before_failure": exc.completed_rows_before_failure,
                }
                if exc.category in {"api_quota_or_rate_limit", "api_authentication_error"}:
                    error["resume"] = build_resume_payload(args)
                payload["error"] = error
                exit_code = 1
            except Exception as exc:
                category = classify_generation_exception(exc)
                payload["error"] = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "category": category,
                }
                if category in {"api_quota_or_rate_limit", "api_authentication_error"}:
                    payload["error"]["resume"] = build_resume_payload(args)
                exit_code = 1
    else:
        exit_code = 0
        payload["note"] = "Validation only. Add --execute to call the configured LLM engine."
    payload["exit_code"] = exit_code
    write_status_json(args.status_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
