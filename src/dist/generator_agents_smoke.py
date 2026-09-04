#!/usr/bin/env python3
"""Validate configured generator agents and optionally run one inference."""

from __future__ import annotations

import argparse
import os
import json
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import new_full_finqa_run as generator_runtime  # noqa: E402
from new_full_finqa_run import (  # noqa: E402
    build_prompt,
    generate_codes,
    load_examples,
    resolve_engine,
    route_execution_status,
)

DEFAULT_ENGINES = [
    "qwen3_6",
    "mistral4",
    "llama4",
    "gpt4_1",
    "gpt5_3_codexS",
    "gpt5_5",

]

SMOKE_DEFAULT_ENDPOINT = os.environ.get("CHATMOCK_BASE_URL") or os.environ.get("VLLM_BASE_URL") or "http://localhost:8000/v1"

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def engine_status(engine: str, prompt: str, execute: bool, max_tokens: int) -> dict[str, Any]:
    config = resolve_engine(engine)
    route_status = route_execution_status(config)
    payload: dict[str, Any] = {
        "engine": engine,
        "config": config.to_public_dict(),
        "execute_requested": execute,
        "route_status": route_status,
        "status": "configured" if config.available else route_status,
    }
    if not execute:
        return payload
    if not config.available:
        payload["status"] = route_status
        return payload
    try:
        codes = generate_codes(config, prompt, "greedy", max_tokens=max_tokens)
    except Exception as exc:
        category = generator_runtime.classify_generation_exception(exc)
        payload["status"] = "runtime_blocked"
        payload["route_status"] = "runtime_blocked"
        payload["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "category": category,
            "traceback_tail": traceback.format_exc()[-2000:],
        }
        return payload
    payload["status"] = "passed_smoke"
    payload["route_status"] = "passed_smoke"
    payload["generated_preview"] = codes[0][:500] if codes else ""
    return payload


def run_resume_self_test() -> dict[str, Any]:
    class QuotaError(Exception):
        status_code = 429
        code = "insufficient_quota"

    class PlainRuntimeError(Exception):
        pass

    assert generator_runtime.classify_generation_exception(
        QuotaError("insufficient_quota: credit balance exhausted")
    ) == "api_quota_or_rate_limit"
    assert generator_runtime.classify_generation_exception(
        PlainRuntimeError("model returned invalid code")
    ) == "generator_runtime_error"

    examples = [
        {"prompt": "", "table": "", "retrieved": [], "question": "q1", "answer": 1},
        {"prompt": "", "table": "", "retrieved": [], "question": "q2", "answer": 1},
        {"prompt": "", "table": "", "retrieved": [], "question": "q3", "answer": 1},
    ]
    config = generator_runtime.EngineConfig(
        requested_engine="gpt5_5",
        engine="gpt5_5",
        route="chatmock_openai_compatible",
        model="gpt-5",
        actual_model="gpt-5.5",
        endpoint=SMOKE_DEFAULT_ENDPOINT,
        formal_model="gpt-5.5",
        runtime_profile="fallback_smoke",
        api_version=None,
        api_key="key",
        missing_credentials=[],
        credential_sources={},
        credential_files=[],
        credential_warnings=[],
    )

    with tempfile.TemporaryDirectory(prefix="generator_resume_smoke_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_json = tmp_path / "input.json"
        output_jsonl = tmp_path / "generated.jsonl"
        status_json = tmp_path / "status.json"
        input_json.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
        output_jsonl.write_text('{"done": 1}\n{"done": 2}\n', encoding="utf-8")

        original_generate_codes = generator_runtime.generate_codes

        def fail_with_quota(*_args: Any, **_kwargs: Any) -> list[str]:
            raise QuotaError("insufficient_quota: credit balance exhausted")

        generator_runtime.generate_codes = fail_with_quota
        try:
            try:
                generator_runtime.run_generation(
                    config=config,
                    input_json=input_json,
                    output_jsonl=output_jsonl,
                    profile="greedy",
                    limit=-1,
                    sleep_seconds=0,
                    max_tokens=16,
                    resume_output=True,
                )
            except generator_runtime.GenerationInterrupted as exc:
                assert exc.category == "api_quota_or_rate_limit"
                assert exc.example_index == 2
                assert exc.completed_rows_before_failure == 2
            else:
                raise AssertionError("quota interruption was not raised")
        finally:
            generator_runtime.generate_codes = original_generate_codes

        resume_args = argparse.Namespace(
            engine="gpt5_5",
            input_json=input_json,
            output_jsonl=output_jsonl,
            status_json=status_json,
            profile="greedy",
            limit=-1,
            max_tokens=16,
            sleep_seconds=0,
        )
        resume_payload = generator_runtime.build_resume_payload(resume_args)
        assert "--resume-output" in resume_payload["command"]
        assert "OPENAI_API_KEY" not in resume_payload["command"]
        assert "CHATMOCK_API_KEY" not in resume_payload["command"]

    original_collect_credentials = generator_runtime.collect_credentials

    def fake_credentials(values: dict[str, str]) -> generator_runtime.CredentialStore:
        return generator_runtime.CredentialStore(
            values=values,
            sources={name: "self_test" for name in values},
            files_used=[],
            warnings=[],
        )

    try:
        generator_runtime.collect_credentials = lambda: fake_credentials({"GPT5_3_CODEX_ROUTE": "api_key"})
        missing_api_config = generator_runtime.resolve_engine("gpt5_3_codexS", credential_purpose="execute")
        assert missing_api_config.route == "closed_api_openai"
        assert missing_api_config.missing_credentials == ["OPENAI_API_KEY or CODEX_API_KEY"]

        generator_runtime.collect_credentials = lambda: fake_credentials(
            {
                "GPT5_3_CODEX_ROUTE": "api_key",
                "OPENAI_API_KEY": "example_key",
                "OPENAI_GPT5_3_CODEX_MODEL": "gpt-test",
            }
        )
        api_config = generator_runtime.resolve_engine("gpt5_3_codexS", credential_purpose="execute")
        assert api_config.route == "closed_api_openai"
        assert api_config.available
        assert api_config.model == "gpt-test"
    finally:
        generator_runtime.collect_credentials = original_collect_credentials

    return {
        "quota_category": "api_quota_or_rate_limit",
        "nonquota_category": "generator_runtime_error",
        "resume_output_skipped_rows": 2,
        "resume_command_contains_resume_output": True,
        "gpt5_3_codexS_api_key_route": "closed_api_openai",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check generator LLM agents.")
    parser.add_argument("--input-json", type=Path, default=REPO_ROOT / "Experiment/finqa_flan_o/retriever/outputs/best_matched_with_retrieved_facts_and_questions.json")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "Experiment/generator_agents_smoke/status.json")
    parser.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--resume-self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_self_test:
        payload = {"time": utc_now(), "resume_self_test": run_resume_self_test()}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output_json": str(args.output_json), "resume_self_test": "passed"}, indent=2))
        return
    examples = load_examples(args.input_json, 1) if args.input_json.exists() else []
    prompt = build_prompt(examples[0]) if examples else "Return Python code that sets ans = 1."
    results = [engine_status(engine, prompt, args.execute, args.max_tokens) for engine in args.engines]
    counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        route_status = str(result.get("route_status") or status)
        counts[status] = counts.get(status, 0) + 1
        route_counts[route_status] = route_counts.get(route_status, 0) + 1
    payload = {
        "time": utc_now(),
        "input_json": str(args.input_json),
        "input_exists": args.input_json.exists(),
        "execute": args.execute,
        "max_tokens": args.max_tokens,
        "status_counts": counts,
        "route_status_counts": route_counts,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "status_counts": counts, "route_status_counts": route_counts}, indent=2))
    if args.execute and counts.get("runtime_blocked"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
