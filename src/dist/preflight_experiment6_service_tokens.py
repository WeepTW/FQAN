#!/usr/bin/env python3
"""Token-budget preflight against a running llama.cpp server for Experiment 6 v2."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment6_narrative2_generation as runner


def request_tokens(base_url: str, api_key: str, prompt: str, timeout: float) -> int:
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[:-3]
    request = urllib.request.Request(
        endpoint + "/tokenize",
        data=json.dumps({"content": prompt, "add_special": True}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: Any = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        raise runner.ProtocolError(f"tokenize endpoint failed: {error}") from error
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list):
        raise runner.ProtocolError(f"tokenize response has no tokens[]: {payload!r}")
    return len(tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=runner.REPO_ROOT / "config" / "experiment6_narrative2_generation.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--source-id", default="mistral4")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    config = runner.load_config(args.config.resolve())
    rows, input_report = runner.read_input_rows(config, 0)
    cases = [
        case for case in runner.expand_matrix(config)
        if case.source_id == args.source_id
    ]
    if not cases:
        raise runner.ProtocolError(f"no cases for source_id={args.source_id}")
    direct = config["directBinding"]
    max_input = int(direct["maxInputTokens"])
    max_new = int(direct["maxNewTokens"])
    context = int(direct["localContextTokens"])
    measurements: list[dict[str, Any]] = []
    for case in cases:
        for row in rows:
            prompt = row.direct_prompts[case.prompt_mode]
            token_count = request_tokens(args.base_url, args.api_key, prompt, args.timeout)
            measurement = {
                "outputId": case.output_id,
                "source": row.source,
                "promptMode": case.prompt_mode,
                "promptSha256": runner.sha256_text(prompt),
                "tokens": token_count,
                "promptPlusCompletion": token_count + max_new,
            }
            measurements.append(measurement)
            if token_count > max_input:
                raise runner.ProtocolError(
                    f"{case.output_id}/{row.source} has {token_count}>{max_input} input tokens"
                )
            if token_count + max_new > context:
                raise runner.ProtocolError(
                    f"{case.output_id}/{row.source} prompt+completion "
                    f"{token_count + max_new}>{context}"
                )
    report = {
        "time": runner.utc_now(),
        "protocol": config["protocol"],
        "status": "passed",
        "sourceId": args.source_id,
        "baseUrl": args.base_url,
        "rows": input_report["fullWorkbookRows"],
        "cases": len(cases),
        "measurements": len(measurements),
        "maxInputAllowed": max_input,
        "maxNewTokens": max_new,
        "contextWindow": context,
        "maxObserved": max(item["tokens"] for item in measurements),
        "minObserved": min(item["tokens"] for item in measurements),
        "maxPromptPlusCompletion": max(item["promptPlusCompletion"] for item in measurements),
        "truncationAllowed": False,
        "records": measurements,
    }
    runner.write_json(args.output.resolve(), report)
    print(json.dumps({key: report[key] for key in (
        "status", "sourceId", "rows", "cases", "measurements",
        "maxObserved", "maxPromptPlusCompletion", "contextWindow",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
