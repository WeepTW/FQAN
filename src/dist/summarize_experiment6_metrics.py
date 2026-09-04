#!/usr/bin/env python3
"""Summarize Experiment 6 binding metrics across experiment directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FINETUNED_RETRIEVER_SOURCE_IDS = {
    "finqa_flan_z",
    "finqa_flan_m",
    "finqa_flan_d",
    "finqa_mistral_z",
    "finqa_mistral_m",
    "finqa_mistral_d",
    "finqa_t5gemma2_z",
    "finqa_t5gemma2_m",
    "finqa_t5gemma2_d",
}
BASE_RETRIEVER_SOURCE_IDS = {
    "flan_t5_large",
    "mistral_v0_3",
    "t5gemma_2_1b_1b",
}
LOCAL_GENERATOR_SOURCE_IDS = {
    "mistral4",
    "qwen3_6",
    "llama3_3",
    "gpt4_1",
    "gpt5_3_codexS",
    "gpt5_5",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(payload: dict[str, Any], key: str) -> Any:
    return payload.get(key)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def runtime_stages(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for item in metadata.get("runtime") or []:
        nested = item.get("stages") if isinstance(item, dict) else None
        if isinstance(nested, list) and nested:
            stages.extend(stage for stage in nested if isinstance(stage, dict))
        elif isinstance(item, dict):
            stages.append(item)
    return stages


def route_audit(metadata: dict[str, Any], status_payload: dict[str, Any]) -> dict[str, str]:
    source_id = metadata.get("source_id") or status_payload.get("source_id") or ""
    stages = runtime_stages(metadata)
    notes: list[str] = []
    actual_route = metadata.get("generation_mode") or ""
    if stages:
        first = stages[0]
        config = first.get("config") if isinstance(first.get("config"), dict) else {}
        if first.get("prediction_contract"):
            actual_route = str(first.get("prediction_contract"))
        elif config.get("route"):
            actual_route = str(config.get("route"))
        elif first.get("family"):
            actual_route = f"legacy_{first.get('family')}_retriever"

    has_retfact_stage = any(
        stage.get("prediction_contract") == "retfact_retriever"
        or stage.get("family") in {"flan", "mistral", "t5gemma2"}
        for stage in stages
    )
    has_conversion = any(stage.get("stage") == "retfact_to_binding_conversion" for stage in stages)

    if source_id in FINETUNED_RETRIEVER_SOURCE_IDS:
        has_adapter = any(stage.get("use_adapter") is True and stage.get("adapter_dir") for stage in stages)
        if not has_adapter:
            notes.append("fine_tuned_source_without_adapter")

    if source_id in FINETUNED_RETRIEVER_SOURCE_IDS | BASE_RETRIEVER_SOURCE_IDS:
        if has_retfact_stage and not has_conversion:
            notes.append("retfact_without_binding_conversion")

    if source_id in LOCAL_GENERATOR_SOURCE_IDS:
        bad_retriever_stage = any(
            stage.get("family") in {"flan", "mistral", "t5gemma2"}
            or stage.get("prediction_contract") == "retfact_retriever"
            for stage in stages
        )
        if bad_retriever_stage:
            notes.append("generator_source_routed_as_retriever")

    if notes:
        return {
            "status_override": "invalid_superseded_wrong_route",
            "route_audit": ";".join(notes),
            "actual_route": actual_route,
        }
    return {"status_override": "", "route_audit": "ok", "actual_route": actual_route}


def case_rows(experiment_dir: Path, script_name: str) -> list[dict[str, Any]]:
    binding_eval = experiment_dir / "binding_eval"
    if not binding_eval.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for case_dir in sorted(item for item in binding_eval.iterdir() if item.is_dir()):
        metrics_path = case_dir / "metrics.json"
        status_path = case_dir / "status.json"
        if not metrics_path.is_file():
            continue
        metrics = load_json(metrics_path)
        status_payload = load_json(status_path) if status_path.is_file() else {}
        metadata = status_payload.get("preflight", {}).get("prediction_metadata") or {}
        audit = route_audit(metadata, status_payload)
        aggregate = metrics.get("run_aggregate") or {}
        average = aggregate.get("average") or {}
        top3 = (aggregate.get("top_3_best_runs") or {})
        top3_average = top3.get("average") or {}
        best = (top3.get("runs") or [{}])[0]
        best_metrics = best.get("metrics") or best.get("overall") or {}
        by_field = metrics.get("by_field") or {}
        status = status_payload.get("status") or ("runtime_blocked" if metadata.get("runtime_blocked") else "completed")
        blocker = " | ".join(str(item) for item in status_payload.get("blockers", []))
        if audit["status_override"]:
            status = audit["status_override"]
            blocker = " | ".join(item for item in [blocker, audit["route_audit"]] if item)
        rows.append(
            {
                "script": script_name,
                "experiment": experiment_dir.name,
                "case": case_dir.name,
                "model": metadata.get("source_id") or status_payload.get("source_id") or "",
                "route": metadata.get("narrative_route") or status_payload.get("narrative_route") or "",
                "actual_route": audit["actual_route"],
                "route_audit": audit["route_audit"],
                "status": status,
                "blocker": blocker,
                "overall_precision": metrics.get("precision"),
                "overall_recall": metrics.get("recall"),
                "overall_f1": metrics.get("f1"),
                "average_precision": average.get("precision"),
                "average_recall": average.get("recall"),
                "average_f1": average.get("f1"),
                "top3_precision": top3_average.get("precision"),
                "top3_recall": top3_average.get("recall"),
                "top3_f1": top3_average.get("f1"),
                "top1_run": best.get("run") or best.get("run_id"),
                "top1_precision": best_metrics.get("precision"),
                "top1_recall": best_metrics.get("recall"),
                "top1_f1": best_metrics.get("f1"),
                "ObjectName_f1": (by_field.get("ObjectName") or {}).get("f1"),
                "Trend_f1": (by_field.get("Trend") or {}).get("f1"),
                "Num_f1": (by_field.get("Num") or {}).get("f1"),
            }
        )
    return rows


def parse_experiment_arg(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        script, path = raw.split("=", 1)
    else:
        path = raw
        script = ""
    exp_path = Path(path)
    if not exp_path.is_absolute():
        exp_path = REPO_ROOT / "Experiment" / exp_path
    return script, exp_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", default=[], help="script=experiment_dir or experiment_dir")
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for raw in args.experiment:
        script, exp_path = parse_experiment_arg(raw)
        rows.extend(case_rows(exp_path, script))

    fieldnames = [
        "script",
        "experiment",
        "case",
        "model",
        "route",
        "actual_route",
        "route_audit",
        "status",
        "blocker",
        "overall_precision",
        "overall_recall",
        "overall_f1",
        "average_precision",
        "average_recall",
        "average_f1",
        "top3_precision",
        "top3_recall",
        "top3_f1",
        "top1_run",
        "top1_precision",
        "top1_recall",
        "top1_f1",
        "ObjectName_f1",
        "Trend_f1",
        "Num_f1",
    ]
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(args.output_tsv)


if __name__ == "__main__":
    main()
