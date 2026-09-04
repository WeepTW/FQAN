#!/usr/bin/env python3
"""Export canonical FinQA retriever train-loss and dev-match JSON data.

This helper is report-only: it reads existing trainer_state and matched JSON
artifacts, then writes a compact thesis-facing JSON document plus an optional
indexed docs/log trace. It never runs model training, inference, or matching.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from experiment6_paths import PATHS


REPO_ROOT = PATHS.repo
WORKSPACE_ROOT = PATHS.workspace

PROMPT_TYPES = {
    "o": "original",
    "z": "zero-shot",
    "m": "many-shot",
    "d": "dynamic-shot",
}

CANONICAL_EXPERIMENTS = [
    ("finqa_flan_o", "flan_t5_large", "FLAN-T5-Large", "o"),
    ("finqa_flan_z", "flan_t5_large", "FLAN-T5-Large", "z"),
    ("finqa_flan_m", "flan_t5_large", "FLAN-T5-Large", "m"),
    ("finqa_flan_d", "flan_t5_large", "FLAN-T5-Large", "d"),
    ("finqa_mistral_o", "mistral_v0_3", "Mistral-7B-Instruct-v0.3", "o"),
    ("finqa_mistral_z", "mistral_v0_3", "Mistral-7B-Instruct-v0.3", "z"),
    ("finqa_mistral_m", "mistral_v0_3", "Mistral-7B-Instruct-v0.3", "m"),
    ("finqa_mistral_d", "mistral_v0_3", "Mistral-7B-Instruct-v0.3", "d"),
    ("finqa_t5gemma2_o", "t5gemma_2_1b_1b", "t5gemma-2-1b-1b", "o"),
    ("finqa_t5gemma2_z", "t5gemma_2_1b_1b", "t5gemma-2-1b-1b", "z"),
    ("finqa_t5gemma2_m", "t5gemma_2_1b_1b", "t5gemma-2-1b-1b", "m"),
    ("finqa_t5gemma2_d", "t5gemma_2_1b_1b", "t5gemma-2-1b-1b", "d"),
]

PRIMARY_CONTEXT_MATCH_TYPES = frozenset({"prediction_fragment"})
AUXILIARY_RETFACT_MATCH_TYPES = frozenset({"retfact_vs_rel_fact_label"})
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")
DICT_RE = re.compile(r"\{.*\}")
TRAIN_MESSAGE_RE = re.compile(r"Training (?P<route>.+?) retriever (?P<fields>.+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument(
        "--source-map-log",
        type=Path,
        default=PATHS.log / "20260615T024537Z_experiment7_fqan_corrected_current_status.json",
        help="Experiment 7 status JSON containing processed_input_paths for finqa_dev matches.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PATHS.asset / "finqa_retriever_loss_by_epoch.json",
    )
    parser.add_argument("--log-dir", type=Path, default=PATHS.log)
    parser.add_argument(
        "--skip-log",
        action="store_true",
        help="Write only --output-json; useful for /tmp dry-run validation.",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else base / path


def display_path(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_key_value_fields(text: str) -> dict[str, str]:
    if text.startswith("with "):
        text = text[5:]
    fields: dict[str, str] = {}
    for part in text.split(", "):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def checkpoint_number(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.parent.name)
    return int(match.group(1)) if match else -1


def find_latest_trainer_state(repo_root: Path, expt_id: str) -> Path:
    train_root = repo_root / "Experiment" / expt_id / "retriever" / "train"
    candidates = sorted(train_root.glob("checkpoint-*/trainer_state.json"), key=checkpoint_number)
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json found under {train_root}")
    return candidates[-1]


def parse_train_log(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path_exists": path.is_file(),
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "command": None,
        "training_message": None,
        "training_config": {},
        "final_train_metrics": None,
    }
    if not path.is_file():
        return result

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("started_at="):
            result["started_at"] = line.split("=", 1)[1]
        elif line.startswith("finished_at="):
            result["finished_at"] = line.split("=", 1)[1]
        elif line.startswith("exit_code="):
            result["exit_code"] = as_int(line.split("=", 1)[1])
        elif line.startswith("command="):
            result["command"] = line.split("=", 1)[1]
        elif "Training " in line and " retriever " in line:
            message = line[line.find("Training ") :]
            match = TRAIN_MESSAGE_RE.search(message)
            if match:
                result["training_message"] = message
                result["training_config"] = parse_key_value_fields(match.group("fields"))
        elif "'train_loss'" in line:
            match = DICT_RE.search(line)
            if not match:
                continue
            try:
                metrics = ast.literal_eval(match.group(0))
            except (SyntaxError, ValueError):
                continue
            if isinstance(metrics, dict) and "train_loss" in metrics:
                result["final_train_metrics"] = {
                    "epoch": as_float(metrics.get("epoch")),
                    "train_loss": as_float(metrics.get("train_loss")),
                    "train_runtime": as_float(metrics.get("train_runtime")),
                    "train_samples_per_second": as_float(metrics.get("train_samples_per_second")),
                    "train_steps_per_second": as_float(metrics.get("train_steps_per_second")),
                    "source": "retriever/train.log final train_loss record",
                }

    return result


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def epoch_bucket(epoch: float) -> int:
    if epoch <= 0:
        return 0
    return int(math.ceil(epoch))


def parse_train_state(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    log_history = payload.get("log_history", [])
    if not isinstance(log_history, list):
        raise ValueError(f"log_history must be a list: {path}")

    raw_points: list[dict[str, Any]] = []
    eval_points: list[dict[str, Any]] = []
    final_train_metrics: dict[str, Any] | None = None

    for index, item in enumerate(log_history):
        if not isinstance(item, dict):
            continue
        step = as_int(item.get("step"))
        epoch = as_float(item.get("epoch"))
        if "loss" in item:
            loss = as_float(item.get("loss"))
            if step is None or epoch is None or loss is None:
                continue
            raw_points.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "epoch_fraction": epoch,
                    "loss": loss,
                    "learning_rate": as_float(item.get("learning_rate")),
                    "grad_norm": as_float(item.get("grad_norm")),
                    "source_index": index,
                    "source": "trainer_state.log_history.loss",
                }
            )
        if "eval_loss" in item:
            eval_loss = as_float(item.get("eval_loss"))
            if step is None or epoch is None or eval_loss is None:
                continue
            eval_points.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "epoch_fraction": epoch,
                    "eval_loss": eval_loss,
                    "eval_runtime": as_float(item.get("eval_runtime")),
                    "eval_samples_per_second": as_float(item.get("eval_samples_per_second")),
                    "eval_steps_per_second": as_float(item.get("eval_steps_per_second")),
                    "source_index": index,
                    "source": "trainer_state.log_history.eval_loss",
                }
            )
        if "train_loss" in item:
            final_train_metrics = {
                "epoch": epoch,
                "train_loss": as_float(item.get("train_loss")),
                "train_runtime": as_float(item.get("train_runtime")),
                "train_samples_per_second": as_float(item.get("train_samples_per_second")),
                "train_steps_per_second": as_float(item.get("train_steps_per_second")),
            }

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in raw_points:
        grouped[epoch_bucket(float(point["epoch"]))].append(point)

    epoch_summary: list[dict[str, Any]] = []
    for bucket in sorted(grouped):
        points = sorted(grouped[bucket], key=lambda point: (point["step"], point["source_index"]))
        losses = [float(point["loss"]) for point in points]
        epoch_summary.append(
            {
                "epoch_interval": bucket,
                "epoch_range": {
                    "start_exclusive": bucket - 1,
                    "end_inclusive": bucket,
                },
                "is_exact_epoch_end": False,
                "count": len(points),
                "mean": sum(losses) / len(losses),
                "min": min(losses),
                "max": max(losses),
                "last": losses[-1],
                "first_step": points[0]["step"],
                "last_step": points[-1]["step"],
                "first_epoch_fraction": points[0]["epoch_fraction"],
                "last_epoch_fraction": points[-1]["epoch_fraction"],
            }
        )

    return {
        "loss_source_policy": {
            "curve_loss": "latest retriever/train/checkpoint-*/trainer_state.json log_history entries containing loss",
            "final_train_loss": "retriever/train.log final train_loss record",
            "epoch_axis": "Use raw_points[*].epoch_fraction for plotting; epoch_summary groups logging points by HF epoch interval and is not exact epoch-end loss.",
        },
        "latest_checkpoint": path.parent.name,
        "global_step": as_int(payload.get("global_step")),
        "completed_epoch": as_float(payload.get("epoch")),
        "best_metric": as_float(payload.get("best_metric")),
        "best_model_checkpoint": payload.get("best_model_checkpoint"),
        "raw_points": raw_points,
        "logging_points": raw_points,
        "epoch_summary": epoch_summary,
        "eval_points": eval_points,
        "trainer_state_final_train_metrics": final_train_metrics,
        "final_train_metrics": None,
    }


def score_summary(matched: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        match
        for item in matched
        for match in item.get("retrieved_with_scores", [])
        if isinstance(match, dict)
    ]
    primary_context_matches = [
        match
        for match in matches
        if str(match.get("matched_by", "unknown")) in PRIMARY_CONTEXT_MATCH_TYPES
    ]
    question_matches = [
        match
        for match in matches
        if str(match.get("matched_by", "unknown")) == "question"
    ]
    relfact_diagnostic_matches = [
        match
        for match in matches
        if str(match.get("matched_by", "unknown")) in AUXILIARY_RETFACT_MATCH_TYPES
    ]

    def values(group: list[dict[str, Any]], key: str) -> list[float]:
        result: list[float] = []
        for match in group:
            value = as_float(match.get(key))
            if value is not None:
                result.append(value)
        return result

    by_matched_by: dict[str, dict[str, Any]] = {}
    for matched_by in sorted({str(match.get("matched_by", "unknown")) for match in matches}):
        group = [match for match in matches if str(match.get("matched_by", "unknown")) == matched_by]
        by_matched_by[matched_by] = {
            "cosine_similarity_summary": numeric_summary(values(group, "cosine_similarity")),
            "legacy_dot_score_summary": numeric_summary(values(group, "legacy_dot_score")),
        }

    cosine_scores = values(matches, "cosine_similarity")
    legacy_scores = values(matches, "legacy_dot_score")
    return {
        "score_selection": "finder_context_sentence_matching",
        "matched_sentences": len(matches),
        "primary_metric": "primary_prediction_fragment_legacy_dot_score_summary.mean",
        "primary_prediction_fragment_cosine_summary": numeric_summary(
            values(primary_context_matches, "cosine_similarity")
        ),
        "primary_prediction_fragment_legacy_dot_score_summary": numeric_summary(
            values(primary_context_matches, "legacy_dot_score")
        ),
        "auxiliary_question_cosine_summary": numeric_summary(
            values(question_matches, "cosine_similarity")
        ),
        "auxiliary_question_legacy_dot_score_summary": numeric_summary(
            values(question_matches, "legacy_dot_score")
        ),
        "auxiliary_retfact_label_cosine_summary": numeric_summary(
            values(relfact_diagnostic_matches, "cosine_similarity")
        ),
        "auxiliary_retfact_label_legacy_dot_score_summary": numeric_summary(
            values(relfact_diagnostic_matches, "legacy_dot_score")
        ),
        "cosine_similarity_summary": numeric_summary(cosine_scores),
        "legacy_dot_score_summary": numeric_summary(legacy_scores),
        "by_matched_by": by_matched_by,
        "score_type": "legacy_dot_score",
        "mean": sum(legacy_scores) / len(legacy_scores) if legacy_scores else None,
        "min": min(legacy_scores) if legacy_scores else None,
        "max": max(legacy_scores) if legacy_scores else None,
    }


def parse_match(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Matched artifact must be a JSON list: {path}")
    return {
        "dataset": "finqa_dev",
        "rows": len(payload),
        "score_summary": score_summary(payload),
    }


def load_source_map(path: Path) -> dict[str, str]:
    payload = load_json(path)
    source_map = payload.get("processed_input_paths", {})
    if not isinstance(source_map, dict):
        raise ValueError(f"processed_input_paths must be a JSON object: {path}")
    return {str(key): str(value) for key, value in source_map.items()}


def build_payload(args: argparse.Namespace, now: str) -> dict[str, Any]:
    repo_root = resolve_path(args.repo_root, WORKSPACE_ROOT)
    workspace_root = resolve_path(args.workspace_root, WORKSPACE_ROOT)
    source_map_log = resolve_path(args.source_map_log, workspace_root)
    source_map = load_source_map(source_map_log)

    experiments: list[dict[str, Any]] = []
    for expt_id, model_key, model_label, prompt_key in CANONICAL_EXPERIMENTS:
        prompt_type = PROMPT_TYPES[prompt_key]
        trainer_state = find_latest_trainer_state(repo_root, expt_id)
        train = parse_train_state(trainer_state)
        train_log = repo_root / "Experiment" / expt_id / "retriever" / "train.log"
        train_log_info = parse_train_log(train_log)
        train["train_log_summary"] = {
            key: value
            for key, value in train_log_info.items()
            if key not in {"command"}
        }
        train["final_train_metrics"] = train_log_info.get("final_train_metrics")
        train["loss_contract"] = {
            "prompt_mode_reported": train_log_info.get("training_config", {}).get("prompt_mode"),
            "json_targets": as_int(train_log_info.get("training_config", {}).get("json_targets")),
            "retfact_only_loss": as_int(train_log_info.get("training_config", {}).get("retfact_only_loss")),
            "label_max_length": as_int(train_log_info.get("training_config", {}).get("label_max_length")),
            "base_model": train_log_info.get("training_config", {}).get("base_model"),
        }
        source_map_key = f"{expt_id}:finqa_dev"
        if source_map_key not in source_map:
            raise KeyError(f"Missing dev match source for {source_map_key} in {source_map_log}")
        dev_match_path = resolve_path(Path(source_map[source_map_key]), workspace_root)
        if not dev_match_path.is_file():
            raise FileNotFoundError(f"Missing dev matched JSON for {source_map_key}: {dev_match_path}")
        dev_match = parse_match(dev_match_path)

        experiments.append(
            {
                "expt_id": expt_id,
                "model": {
                    "key": model_key,
                    "label": model_label,
                },
                "prompt_type": {
                    "key": prompt_key,
                    "name": prompt_type,
                },
                "train": train,
                "dev_match": dev_match,
                "sources": {
                    "trainer_state": display_path(trainer_state, workspace_root),
                    "trainer_checkpoint": display_path(trainer_state.parent, workspace_root),
                    "train_log": display_path(train_log, workspace_root) if train_log.is_file() else None,
                    "dev_matched_json": display_path(dev_match_path, workspace_root),
                    "source_map_log": display_path(source_map_log, workspace_root),
                    "source_map_key": source_map_key,
                },
            }
        )

    return {
        "schema_version": 2,
        "created_at_utc": now,
        "source_policy": {
            "scope": "canonical_finqa_retriever_experiments_only",
            "canonical_expt_ids": [item[0] for item in CANONICAL_EXPERIMENTS],
            "excluded_patterns": ["*_new", "old_*", "*few*", "*smoke*", "*probe*"],
            "train_curve_loss_source": "latest retriever/train/checkpoint-*/trainer_state.json log_history[*].loss per canonical experiment",
            "train_final_loss_source": "retriever/train.log final train_loss record per canonical experiment",
            "epoch_plot_policy": "Plot raw_points[*].loss against raw_points[*].epoch_fraction; epoch_summary is interval-grouped logging data, not exact epoch-end loss.",
            "dev_match_source": "processed_input_paths[*:finqa_dev] from the formal Experiment 7 source-map log",
            "source_map_log": display_path(source_map_log, workspace_root),
            "no_model_execution": True,
        },
        "experiments": experiments,
    }


def validate_payload(payload: dict[str, Any], workspace_root: Path) -> None:
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != len(CANONICAL_EXPERIMENTS):
        raise ValueError(f"Expected {len(CANONICAL_EXPERIMENTS)} experiments, got {len(experiments or [])}")

    forbidden = ("_new", "old_", "few", "smoke", "probe")
    for item in experiments:
        expt_id = str(item.get("expt_id", ""))
        if any(token in expt_id for token in forbidden):
            raise ValueError(f"Non-canonical experiment leaked into export: {expt_id}")
        train = item.get("train", {})
        raw_points = train.get("raw_points", [])
        if not raw_points:
            raise ValueError(f"Missing train.raw_points for {expt_id}")
        final_train_metrics = train.get("final_train_metrics")
        if not isinstance(final_train_metrics, dict) or final_train_metrics.get("train_loss") is None:
            raise ValueError(f"Missing train.final_train_metrics.train_loss for {expt_id}")
        rows = item.get("dev_match", {}).get("rows", 0)
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError(f"Missing dev_match rows for {expt_id}")
        sources = item.get("sources", {})
        for key in ["trainer_state", "trainer_checkpoint", "train_log", "dev_matched_json", "source_map_log"]:
            value = sources.get(key)
            if value is None:
                continue
            source_path = Path(value)
            if not source_path.is_absolute():
                source_path = workspace_root / source_path
            if not source_path.exists():
                raise FileNotFoundError(f"Referenced source path is missing for {expt_id}: {value}")


def write_run_log(
    payload: dict[str, Any],
    output_json: Path,
    log_dir: Path,
    workspace_root: Path,
    repo_root: Path,
    now: str,
) -> Path:
    safe_now = now.replace("-", "").replace(":", "")
    log_path = log_dir / f"{safe_now}_finqa_retriever_loss_export.json"
    log_payload = {
        "time": now,
        "kind": "finqa_retriever_loss_export",
        "status": "completed",
        "repo": str(repo_root),
        "output_json": display_path(output_json, workspace_root),
        "experiments_count": len(payload.get("experiments", [])),
        "source_policy": payload.get("source_policy", {}),
        "summary": "Exported canonical FinQA retriever train loss and dev-match FINDER legacy dot-score summaries with cosine diagnostics; no model execution.",
    }
    write_json(log_path, log_payload)

    index_path = log_dir / "index.json"
    if index_path.is_file():
        index = load_json(index_path)
        if not isinstance(index, dict):
            index = {"entries": []}
    else:
        index = {"entries": []}
    entries = index.setdefault("entries", [])
    if not isinstance(entries, list):
        entries = []
    rel_log = display_path(log_path, workspace_root)
    entry = {
        "time": now,
        "path": rel_log,
        "repo": str(repo_root),
        "kind": "finqa_retriever_loss_export",
        "status": "completed",
        "summary": "Canonical FinQA retriever train loss and dev-match FINDER legacy dot-score JSON exported to docs/asset.",
        "tags": ["finqa", "retriever", "loss", "legacy_dot_score", "cosine_similarity", "doc_export"],
    }
    index["entries"] = [item for item in entries if not isinstance(item, dict) or item.get("path") != rel_log] + [entry]
    write_json(index_path, index)
    return log_path


def main() -> None:
    args = parse_args()
    repo_root = resolve_path(args.repo_root, WORKSPACE_ROOT)
    workspace_root = resolve_path(args.workspace_root, WORKSPACE_ROOT)
    output_json = resolve_path(args.output_json, workspace_root)
    log_dir = resolve_path(args.log_dir, workspace_root)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = build_payload(args, now)
    validate_payload(payload, workspace_root)
    write_json(output_json, payload)

    log_path = None
    if not args.skip_log:
        log_path = write_run_log(payload, output_json, log_dir, workspace_root, repo_root, now)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_json": str(output_json),
                "log_path": str(log_path) if log_path else None,
                "experiments": len(payload["experiments"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
