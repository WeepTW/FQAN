#!/usr/bin/env python3
"""Run exported Experiment 6 GPT-4.1 tasks in a GPT-4.1-capable environment.

This script is intentionally serial: it sends one request at a time and appends a
checkpoint row after each completed task.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_task_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for row in read_jsonl(path):
        if row.get("status") == "completed" and row.get("prediction"):
            done.add(str(row.get("task_id")))
    return done


def build_client(args: argparse.Namespace) -> tuple[Any, str, str]:
    from openai import AzureOpenAI, OpenAI

    if os.environ.get("OPENAI_BASE_URL") and os.environ.get("OPENAI_API_KEY"):
        model = args.model or os.environ.get("OPENAI_MODEL") or os.environ.get("AZURE_OPENAI_GPT4_1_DEPLOYMENT") or "gpt-4.1"
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
            timeout=args.request_timeout_seconds,
            max_retries=0,
        )
        return client, model, "openai_compatible"
    if os.environ.get("OPENAI_API_KEY"):
        model = args.model or os.environ.get("OPENAI_MODEL") or "gpt-4.1"
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=args.request_timeout_seconds,
            max_retries=0,
        )
        return client, model, "openai"
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"):
        deployment = (
            args.model
            or os.environ.get("AZURE_OPENAI_GPT4_1_DEPLOYMENT")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or "gpt-4.1"
        )
        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            timeout=args.request_timeout_seconds,
            max_retries=0,
        )
        return client, deployment, "azure_openai"
    raise RuntimeError(
        "No GPT-4.1 route found. Set OPENAI_API_KEY, or OPENAI_BASE_URL+OPENAI_API_KEY, "
        "or AZURE_OPENAI_ENDPOINT+AZURE_OPENAI_API_KEY."
    )


def generate(client: Any, model: str, task: dict[str, Any], args: argparse.Namespace) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["prompt"]},
        ],
        temperature=0,
        max_tokens=int(task.get("max_tokens") or args.max_tokens),
    )
    return response.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout-seconds", type=int, default=900)
    parser.add_argument("--request-spacing-seconds", type=int, default=20)
    parser.add_argument("--retry-max", type=int, default=3)
    parser.add_argument("--retry-wait-seconds", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.tasks_jsonl)
    done = completed_task_ids(args.results_jsonl)
    client, model, route = build_client(args)
    print(
        json.dumps(
            {
                "time": utc_now(),
                "status": "started",
                "route": route,
                "model": model,
                "task_count": len(tasks),
                "already_completed": len(done),
                "results_jsonl": str(args.results_jsonl),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for task_index, task in enumerate(tasks, start=1):
        task_id = str(task["task_id"])
        if task_id in done:
            continue
        last_error: dict[str, Any] | None = None
        for attempt in range(1, max(0, args.retry_max) + 2):
            try:
                prediction = generate(client, model, task, args)
                append_jsonl(
                    args.results_jsonl,
                    {
                        "time": utc_now(),
                        "task_id": task_id,
                        "status": "completed",
                        "prediction": prediction,
                        "route": route,
                        "model": model,
                        "experiment_id": task["experiment_id"],
                        "run": task["run"],
                        "row_index": task["row_index"],
                        "case_id": task["case_id"],
                    },
                )
                done.add(task_id)
                print(
                    json.dumps(
                        {"time": utc_now(), "status": "completed", "task": task_index, "task_id": task_id},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                break
            except Exception as exc:  # noqa: BLE001 - external route evidence
                last_error = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback_tail": traceback.format_exc()[-2000:],
                    "attempt": attempt,
                }
                append_jsonl(
                    args.results_jsonl,
                    {
                        "time": utc_now(),
                        "task_id": task_id,
                        "status": "runtime_blocked",
                        "error": last_error,
                        "route": route,
                        "model": model,
                        "experiment_id": task["experiment_id"],
                        "run": task["run"],
                        "row_index": task["row_index"],
                        "case_id": task["case_id"],
                    },
                )
                if attempt > max(0, args.retry_max):
                    raise
                print(
                    json.dumps(
                        {
                            "time": utc_now(),
                            "status": "retry_wait",
                            "task_id": task_id,
                            "attempt": attempt,
                            "wait_seconds": args.retry_wait_seconds,
                            "error": last_error["message"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                time.sleep(max(0, args.retry_wait_seconds))
        if args.request_spacing_seconds > 0:
            time.sleep(args.request_spacing_seconds)

    print(
        json.dumps(
            {"time": utc_now(), "status": "completed", "completed": len(done), "task_count": len(tasks)},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
