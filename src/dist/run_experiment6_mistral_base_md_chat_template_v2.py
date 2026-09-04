#!/usr/bin/env python3
"""Isolated Experiment 6 runner for Mistral base many/dynamic chat-template cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from experiment6_paths import PATHS
import run_experiment6_narrative2_generation as core


INFERENCE_SCRIPT = PATHS.resolve(
    "repo", ".external/FINDER/Retriever Codes/Mistral/mistral_direct_binding_chat_inference.py"
)
ORIGINAL_RUN_RETRIEVER_CASE = core.legacy.run_retriever_case
ORIGINAL_TOKEN_PREFLIGHT = core.token_preflight
OOM_PATTERN = re.compile(r"(?:CUDA\s+out\s+of\s+memory|torch\.OutOfMemoryError)", re.I)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_config_path(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    return args.config.resolve()


def validate_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = config.get("mistralDirectChatTemplate")
    if not isinstance(contract, Mapping):
        raise RuntimeError("config is missing mistralDirectChatTemplate")
    expected_template = str(contract.get("chatTemplateSha256") or "")
    if len(expected_template) != 64:
        raise RuntimeError("chatTemplateSha256 is missing or invalid")
    for name, path in (
        ("wrapper", Path(__file__).resolve()),
        ("inference", INFERENCE_SCRIPT.resolve()),
    ):
        spec = contract.get(name)
        if not isinstance(spec, Mapping):
            raise RuntimeError(f"chat-template contract missing {name}")
        actual = file_sha256(path)
        if actual != spec.get("sha256"):
            raise RuntimeError(
                f"{name} SHA-256 mismatch: {actual} != {spec.get('sha256')}"
            )
    return contract


def build_command(
    *,
    csv_path: Path,
    raw_output: Path,
    prompt_mode: str,
    args: argparse.Namespace,
    batch_size: int,
    contract: Mapping[str, Any],
) -> list[str]:
    return [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--input-csv",
        str(csv_path),
        "--output-txt",
        str(raw_output),
        "--prompt-mode",
        prompt_mode,
        "--max-infer-samples",
        str(args.limit if args.limit > 0 else -1),
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(args.max_tokens),
        "--max-input-length",
        str(args.max_input_tokens),
        "--context-window",
        str(contract["contextWindow"]),
        "--structured-output",
        "off",
        "--chat-template-sha256",
        str(contract["chatTemplateSha256"]),
        "--load-in-4bit",
        "true",
        "--sort-by-length",
        "true",
        "--no-adapter",
    ]


def install_token_preflight(contract: Mapping[str, Any]) -> None:
    def token_preflight(
        rows: Sequence[Any],
        cases: Sequence[Any],
        config: Mapping[str, Any],
        base_route_mode: str = "historical",
    ) -> dict[str, Any]:
        report = ORIGINAL_TOKEN_PREFLIGHT(rows, cases, config, base_route_mode)
        selected = [
            case
            for case in cases
            if case.source_id == "mistral_v0_3"
            and case.route == "direct-binding"
            and case.prompt_mode in {"many-shot", "dynamic-shot"}
        ]
        if not selected:
            return report
        tokenizer = AutoTokenizer.from_pretrained(
            str(contract["baseModel"]),
            local_files_only=True,
            trust_remote_code=True,
        )
        template = str(tokenizer.chat_template or "")
        template_sha = core.sha256_text(template)
        if template_sha != contract["chatTemplateSha256"]:
            raise core.ProtocolError(
                "native Mistral chat-template SHA-256 mismatch during preflight"
            )
        if len(template.encode("utf-8")) != int(contract["chatTemplateBytes"]):
            raise core.ProtocolError("native Mistral chat-template byte count mismatch")
        measurements: list[dict[str, Any]] = []
        maximum = int(config["directBinding"]["maxInputTokens"])
        completion = int(config["directBinding"]["maxNewTokens"])
        context = int(contract["contextWindow"])
        for case in selected:
            for row in rows:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row.direct_prompts[case.prompt_mode]}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                item = {
                    "outputId": case.output_id,
                    "source": row.source,
                    "tokens": len(ids),
                    "inputIdsSha256": core.sha256_text(
                        json.dumps(ids, separators=(",", ":"))
                    ),
                }
                measurements.append(item)
                if len(ids) > maximum or len(ids) + completion > context:
                    raise core.ProtocolError(
                        "chat-templated Mistral prompt exceeds the frozen token gate: "
                        f"{item}"
                    )
        report["mistralNativeChatTemplate"] = {
            "policy": contract["policy"],
            "chatTemplateBytes": len(template.encode("utf-8")),
            "chatTemplateSha256": template_sha,
            "maxInputAllowed": maximum,
            "maxNewTokens": completion,
            "contextWindow": context,
            "minObserved": min(item["tokens"] for item in measurements),
            "maxObserved": max(item["tokens"] for item in measurements),
            "measurements": len(measurements),
            "truncationAllowed": False,
            "inputIdentitySha256": core.sha256_text(
                json.dumps(measurements, sort_keys=True, separators=(",", ":"))
            ),
        }
        return report

    core.token_preflight = token_preflight


def install_runner(contract: Mapping[str, Any]) -> None:
    def run_retriever_case(
        case: Any,
        csv_path: Path,
        prompt_mode: str,
        args: argparse.Namespace,
        *,
        use_adapter: bool = True,
        raw_suffix: str = "",
        family_override: str | None = None,
        adapter_dir_override: Path | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        if family_override != "mistral" or use_adapter:
            return ORIGINAL_RUN_RETRIEVER_CASE(
                case,
                csv_path,
                prompt_mode,
                args,
                use_adapter=use_adapter,
                raw_suffix=raw_suffix,
                family_override=family_override,
                adapter_dir_override=adapter_dir_override,
            )
        if adapter_dir_override is not None:
            raise RuntimeError("Mistral chat-template direct route may not load an adapter")
        if prompt_mode not in {"many-shot", "dynamic-shot"}:
            raise RuntimeError(f"unsupported Mistral chat-template prompt mode: {prompt_mode}")

        raw_dir = args.pred_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_stem = f"{case.experiment_id}{raw_suffix}.chat_template"
        raw_output = raw_dir / f"{raw_stem}.jsonl"
        attempted_batches: list[int] = []
        batch_size = int(args.batch_size)
        final_log: Path | None = None
        while True:
            attempted_batches.append(batch_size)
            log_path = raw_dir / f"{raw_stem}.batch_{batch_size}.log"
            completed = core.legacy.run_command(
                build_command(
                    csv_path=csv_path,
                    raw_output=raw_output,
                    prompt_mode=prompt_mode,
                    args=args,
                    batch_size=batch_size,
                    contract=contract,
                ),
                log_path,
                args.case_timeout_seconds,
                args.cuda_visible_devices,
            )
            final_log = log_path
            if completed.returncode == 0:
                break
            combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
            if not OOM_PATTERN.search(combined) or batch_size <= 1:
                raise RuntimeError(
                    f"Mistral chat-template inference failed; see {log_path}"
                )
            batch_size = max(1, batch_size // 2)

        checkpoint = Path(str(raw_output) + ".checkpoint.jsonl")
        predictions = core.legacy.read_raw_prediction_lines(raw_output)
        return predictions, {
            "family": "mistral",
            "actual_engine": case.source_id,
            "prediction_contract": "binding-json-native-chat-template-v1",
            "adapter_dir": None,
            "use_adapter": False,
            "raw_output": str(raw_output),
            "log": str(final_log),
            "row_checkpoint": str(checkpoint),
            "row_checkpoint_rows": sum(
                1
                for line in checkpoint.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            "cuda_visible_devices": args.cuda_visible_devices,
            "execution_device": (
                "cpu" if args.cuda_visible_devices == "cpu" else "cuda"
            ),
            "structured_output": "off",
            "max_input_tokens": int(args.max_input_tokens),
            "max_new_tokens": int(args.max_tokens),
            "context_window": int(contract["contextWindow"]),
            "run_seed": int(os.environ["EXPERIMENT6_RUN_SEED"]),
            "chat_template_applied": True,
            "chat_template_sha256": str(contract["chatTemplateSha256"]),
            "batch_size_initial": int(args.batch_size),
            "batch_size_effective": batch_size,
            "batch_size_attempts": attempted_batches,
        }

    core.legacy.run_retriever_case = run_retriever_case


def main() -> None:
    config_path = selected_config_path(sys.argv[1:])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = validate_contract(config)
    install_token_preflight(contract)
    install_runner(contract)
    core.main()


if __name__ == "__main__":
    main()
