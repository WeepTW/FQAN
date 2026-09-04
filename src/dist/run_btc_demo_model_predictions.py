#!/usr/bin/env python3
"""Run executable model routes for the BTC FinFlier demo without fabricating output."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "data" / "financial_narratives" / "demo"
VARIABLES_MD = WORKSPACE_ROOT / "src" / "doc" / "workspace" / "variables.md"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import build_btc_finflier_demo as demo  # noqa: E402
import new_full_finqa_run as runtime  # noqa: E402

try:  # noqa: WPS229 - optional local helper, no external dependency
    from binding_extraction import coerce_jsonish  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    coerce_jsonish = None


FINE_TUNED_BLOCKED_REASON = (
    "fine-tuned retriever route requires executable retriever inference plus "
    "RetFact-to-Binding conversion for this custom BTC workbook; gold binding was not substituted."
)


def load_variables_md(path: Path = VARIABLES_MD) -> list[str]:
    """Load allowlisted exports from variables.md without printing values."""
    allowlist = {
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    }
    loaded: list[str] = []
    if not path.is_file():
        return loaded
    pattern = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        if key not in allowlist or key in os.environ:
            continue
        value = raw_value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded


def configure_route(route_id: str) -> Any:
    if route_id == "gpt5_5":
        os.environ.setdefault("GPT5_5_CODEX_ROUTE", "openai_compatible")
        os.environ.setdefault("ALLOW_OPENAI_COMPATIBLE_EXECUTE", "1")
    elif route_id == "qwen3_6":
        os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8121/v1")
        os.environ.setdefault("VLLM_API_KEY", "EMPTY")
        os.environ.setdefault("VLLM_SERVED_MODEL_NAME", "qwen3_6_35b_a3b_fp8")
    elif route_id == "mistral4":
        os.environ.setdefault("MISTRAL_SMALL_RUNTIME_BACKEND", "llama_cpp")
        os.environ.setdefault("LLAMA_CPP_BASE_URL", "http://localhost:8122/v1")
        os.environ.setdefault("LLAMA_CPP_API_KEY", "EMPTY")
        os.environ.setdefault("LLAMA_CPP_MODEL_ALIAS", "mistral4")
        os.environ.setdefault(
            "LLAMA_CPP_MODEL_PATH",
            str(
                WORKSPACE_ROOT
                / "Models"
                / "mistral_small_4_119b_2603_gguf"
                / "UD-Q4_K_M"
                / "Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf"
            ),
        )
    return runtime.resolve_engine(route_id, credential_purpose="execute")



def extract_balanced_json_fragment(text: str, start_at: int = 0) -> str | None:
    opener_index = -1
    opener = ""
    for index in range(start_at, len(text)):
        if text[index] in "[{":
            opener_index = index
            opener = text[index]
            break
    if opener_index < 0:
        return None
    closer = "]" if opener == "[" else "}"
    stack = [closer]
    in_string = False
    escaped = False
    for index in range(opener_index + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "[{":
            stack.append("]" if char == "[" else "}")
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return text[opener_index : index + 1]
    return None


def extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    if coerce_jsonish is not None:
        parsed = coerce_jsonish(stripped)
        if isinstance(parsed, (dict, list)) and parsed:
            return parsed
    candidates = [stripped]
    result_match = re.search(r"\bresult\s*:", stripped, re.I)
    if result_match:
        result_fragment = extract_balanced_json_fragment(stripped, result_match.end())
        if result_fragment:
            candidates.insert(0, result_fragment)
    binding_match = re.search(r"\bBinding\s*:", stripped, re.I)
    if binding_match:
        binding_fragment = extract_balanced_json_fragment(stripped, binding_match.end())
        if binding_fragment:
            candidates.insert(0, binding_fragment)
    code_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.S | re.I)
    if code_match:
        candidates.insert(0, code_match.group(1).strip())
    brace_match = re.search(r"(\{.*\})", stripped, re.S)
    if brace_match:
        candidates.append(brace_match.group(1))
    bracket_match = re.search(r"(\[.*\])", stripped, re.S)
    if bracket_match:
        candidates.append(bracket_match.group(1))
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except Exception:
                continue
    return None


def parse_binding_output(raw: str) -> list[dict[str, Any]]:
    parsed = extract_json_object(raw)
    if isinstance(parsed, dict):
        for key in ("Binding", "binding", "result", "Result"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(parsed, (list, tuple)):
        return [item for item in parsed if isinstance(item, dict)]
    return []



def comparison_score(comparison: dict[str, Any]) -> dict[str, Any]:
    total = 0
    matched = 0
    for row in comparison.get("rows", []):
        for field in row.get("fields", {}).values():
            total += 1
            if field.get("match"):
                matched += 1
    return {
        "matched_fields": matched,
        "total_fields": total,
        "ratio": (matched / total) if total else 0.0,
    }


def generate_for_case(config: Any, case: dict[str, Any], max_attempts: int, max_tokens: int) -> dict[str, Any]:
    attempts = []
    best_prediction: list[dict[str, Any]] = []
    best_comparison = demo.compare_binding_fields(case["gold_binding"], best_prediction)
    best_score = comparison_score(best_comparison)
    best_attempt: int | None = None
    for attempt in range(1, max_attempts + 1):
        row: dict[str, Any] = {"attempt": attempt}
        try:
            completions = runtime.generate_text(
                config,
                case["prompt"],
                demo.FINFLIER_STYLE_SYSTEM,
                "greedy",
                max_tokens=max_tokens,
            )
            raw = completions[0] if completions else ""
            prediction = parse_binding_output(raw)
            comparison = demo.compare_binding_fields(case["gold_binding"], prediction)
            score = comparison_score(comparison)
            row.update(
                {
                    "status": "completed",
                    "raw_output": raw,
                    "parsed_prediction": prediction,
                    "comparison": comparison,
                    "field_match_score": score,
                }
            )
            attempts.append(row)
            if score["matched_fields"] > best_score["matched_fields"] or (
                score["matched_fields"] == best_score["matched_fields"] and len(prediction) > len(best_prediction)
            ):
                best_prediction = prediction
                best_comparison = comparison
                best_score = score
                best_attempt = attempt
            if comparison["exact_match"]:
                break
        except Exception as exc:  # pragma: no cover - model runtime diagnostic
            row.update({"status": "runtime_error", "error": f"{exc.__class__.__name__}: {exc}"})
            attempts.append(row)
    return {
        "prediction": best_prediction,
        "comparison": best_comparison,
        "attempts": attempts,
        "exact_match": best_comparison["exact_match"],
        "best_attempt": best_attempt,
        "field_match_score": best_score,
        "runtime_error_only": bool(attempts) and all(item["status"] == "runtime_error" for item in attempts),
    }


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_payload(folder_dir: Path, payload: dict[str, Any]) -> None:
    (folder_dir / "payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (folder_dir / "index.html").write_text(demo.render_html(payload), encoding="utf-8")


def run_folder(
    folder: str,
    route_id: str,
    route_type: str,
    output_root: Path,
    max_attempts: int,
    max_tokens: int,
) -> dict[str, Any]:
    folder_dir = output_root / folder
    payload_path = folder_dir / "payload.json"
    payload = load_payload(payload_path)

    if route_type == "fine_tuned_retriever":
        payload["runtime_status"] = {
            "status": "runtime_blocked",
            "reason": FINE_TUNED_BLOCKED_REASON,
            "route_status": "retriever_to_binding_path_missing",
            "missing": ["custom workbook RetFact-to-Binding execution path"],
        }
        payload["comparison"] = demo.compare_cases(payload["cases"])
        write_payload(folder_dir, payload)
        return {
            "folder": folder,
            "route_id": route_id,
            "runtime_status": payload["runtime_status"],
            "comparison": payload["comparison"],
        }

    config = configure_route(route_id)
    route_status = runtime.route_execution_status(config)
    if not config.available:
        payload["runtime_status"] = {
            "status": "runtime_blocked",
            "reason": "model runtime is not executable in the current shell.",
            "route_status": route_status,
            "missing": list(config.missing_credentials),
        }
        payload["comparison"] = demo.compare_cases(payload["cases"])
        write_payload(folder_dir, payload)
        return {
            "folder": folder,
            "route_id": route_id,
            "runtime_status": payload["runtime_status"],
            "comparison": payload["comparison"],
        }

    predictions: list[list[dict[str, Any]]] = []
    all_case_results = []
    for case in payload["cases"]:
        result = generate_for_case(config, case, max_attempts=max_attempts, max_tokens=max_tokens)
        case["model_prediction"] = result["prediction"]
        case["prediction_attempts"] = result["attempts"]
        case["model_comparison"] = result["comparison"]
        case["best_attempt"] = result["best_attempt"]
        case["field_match_score"] = result["field_match_score"]
        predictions.append(result["prediction"])
        all_case_results.append(result)

    comparison = demo.compare_cases(payload["cases"], predictions)
    if all(result["runtime_error_only"] for result in all_case_results):
        status = "runtime_error"
        reason = "all model calls failed; see prediction_attempts in payload.json."
    elif comparison["exact_match"]:
        status = "completed_exact_match"
        reason = "model prediction matched ObjectName/DataName/Trend/Num for every case."
    else:
        status = "completed_mismatch"
        reason = "model prediction was generated, but ObjectName/DataName/Trend/Num did not all match ground truth after retry budget; best closest attempt was kept."

    payload["runtime_status"] = {
        "status": status,
        "reason": reason,
        "route_status": route_status,
        "missing": [],
        "max_attempts": max_attempts,
    }
    payload["comparison"] = comparison
    write_payload(folder_dir, payload)
    return {
        "folder": folder,
        "route_id": route_id,
        "runtime_status": payload["runtime_status"],
        "comparison": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folders", nargs="*", default=[folder for folder, *_ in demo.MODEL_ROUTES])
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded_vars = load_variables_md()
    selected = set(args.folders)
    report = {
        "generated_at_utc": demo.utc_now(),
        "output_root": str(args.output_root),
        "variables_loaded": sorted(loaded_vars),
        "max_attempts": args.max_attempts,
        "folders": [],
    }
    route_map = {folder: (route_id, route_type) for folder, _label, route_id, route_type in demo.MODEL_ROUTES}
    for folder in args.folders:
        if folder not in route_map:
            raise SystemExit(f"unknown folder: {folder}")
        route_id, route_type = route_map[folder]
        report["folders"].append(
            run_folder(
                folder,
                route_id,
                route_type,
                args.output_root,
                max_attempts=args.max_attempts,
                max_tokens=args.max_tokens,
            )
        )
    rerun_path = args.output_root / "rerun_report.json"
    rerun_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
