#!/usr/bin/env python3
"""Materialize BTC model payloads into FinFlier narrative2_ui-style folders."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_btc_finflier_demo as demo


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
FINFLIER_ROOT = REPO_ROOT / "FinFlier"
DEFAULT_OUTPUT_ROOT = FINFLIER_ROOT
TEMPLATE_DIR = FINFLIER_ROOT / "narrative2_ui"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def position_list(binding: dict[str, Any]) -> list[dict[str, list[int]]]:
    positions = binding.get("Position") or binding.get("positions") or []
    normalized: list[dict[str, list[int]]] = []
    if not isinstance(positions, list):
        return normalized
    for item in positions:
        if not isinstance(item, dict):
            continue
        begin = item.get("Begin") or item.get("begin")
        end = item.get("End") or item.get("end")
        if isinstance(begin, list) and isinstance(end, list) and len(begin) >= 2 and len(end) >= 2:
            try:
                normalized.append({"begin": [int(begin[0]), int(begin[1])], "end": [int(end[0]), int(end[1])]})
            except (TypeError, ValueError):
                continue
    return normalized


def object_names(binding: dict[str, Any]) -> list[str]:
    value = binding.get("ObjectName") or binding.get("object_names")
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, "", "None"):
        return []
    return [str(value)]


def data_name_value(binding: dict[str, Any]) -> Any:
    value = binding.get("DataName") or binding.get("data_name")
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return clean_scalar(value)


def nums_value(binding: dict[str, Any]) -> Any:
    value = binding.get("Num") if "Num" in binding else binding.get("nums")
    return clean_scalar(value)


def normalize_binding(binding: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    data_name = data_name_value(binding)
    first_data_name = str(data_name).split(",", 1)[0].strip() if data_name not in (None, "") else None
    column_index = columns.index(first_data_name) if first_data_name in columns else None
    return {
        "object_names": object_names(binding),
        "data_name": data_name,
        "data_column_index": column_index,
        "positions": position_list(binding),
        "trend": clean_scalar(binding.get("Trend") or binding.get("trend")),
        "nums": nums_value(binding),
        "evidence_text": clean_scalar(binding.get("Text") or binding.get("evidence_text")),
        "raw": binding,
    }


def attempts_summary(case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for attempt in case.get("prediction_attempts", []):
        row = {
            "attempt": attempt.get("attempt"),
            "status": attempt.get("status"),
            "field_match_score": attempt.get("field_match_score"),
        }
        if "error" in attempt:
            row["error"] = attempt.get("error")
        rows.append(row)
    return rows


def clean_model_reason(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^\s*(?:reason|Reason)\s*:\s*", "", text)
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        try:
            text = json.loads(text)
        except Exception:
            text = text[1:-1]
    return str(text).strip()


def extract_reason_from_raw(raw: Any) -> str:
    if raw in (None, ""):
        return ""
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("Reason", "reason"):
            reason = clean_model_reason(parsed.get(key))
            if reason:
                return reason
    match = re.search(r"\breason\s*:\s*", text, flags=re.I)
    if not match:
        return ""
    rest = text[match.end() :].strip()
    if not rest:
        return ""
    if rest[0] == '"':
        try:
            parsed_reason, _ = json.JSONDecoder().raw_decode(rest)
            return clean_model_reason(parsed_reason)
        except Exception:
            pass
    if rest[0] == "'":
        end = rest.find("'", 1)
        if end > 0:
            return clean_model_reason(rest[1:end])
    return clean_model_reason(rest.splitlines()[0].rstrip(",} "))


def best_attempt_row(case: dict[str, Any]) -> dict[str, Any]:
    attempts = [item for item in case.get("prediction_attempts", []) if isinstance(item, dict)]
    best_attempt = case.get("best_attempt")
    if best_attempt is not None:
        for attempt in attempts:
            if attempt.get("attempt") == best_attempt:
                return attempt
    completed = [item for item in attempts if item.get("status") == "completed"]
    return completed[0] if completed else (attempts[0] if attempts else {})


def binding_reason_lines(bindings: list[Any]) -> list[str]:
    lines: list[str] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        for key in ("Reason", "reason", "Text", "result"):
            line = clean_model_reason(binding.get(key))
            if line:
                lines.append(line)
                break
    return lines


def reason_text(payload: dict[str, Any], case: dict[str, Any]) -> str:
    attempt = best_attempt_row(case)
    reason = extract_reason_from_raw(attempt.get("raw_output"))
    if reason:
        return reason
    lines = binding_reason_lines(case.get("model_prediction") or [])
    if lines:
        return "\n".join(lines)
    lines = binding_reason_lines(attempt.get("parsed_prediction") or [])
    return "\n".join(lines)


def to_dataset(payload: dict[str, Any], folder: str) -> dict[str, Any]:
    records = []
    for index, case in enumerate(payload.get("cases", []), start=1):
        columns = case.get("schema", {}).get("columns", [])
        predictions = case.get("model_prediction") or []
        gold = case.get("gold_binding") or []
        records.append(
            {
                "number": index,
                "source": case.get("file") or case.get("case_id") or f"case_{index}",
                "chart_type": case.get("chart_type"),
                "overlay_flags": {},
                "semantics": {},
                "schema": {
                    "columns": columns,
                    "category_column": case.get("schema", {}).get("x_column") or (columns[0] if columns else None),
                    "metric_columns": case.get("schema", {}).get("metric_columns", []),
                },
                "table": case.get("table", []),
                "text": case.get("text"),
                "reason": reason_text(payload, case),
                "result": predictions,
                "gold_result": gold,
                "bindings": [normalize_binding(item, columns) for item in predictions if isinstance(item, dict)],
                "gold_bindings": [normalize_binding(item, columns) for item in gold if isinstance(item, dict)],
                "comparison": case.get("model_comparison"),
                "prompt": case.get("prompt"),
                "attempts_summary": attempts_summary(case),
            }
        )
    return {
        "dataset": f"BTC-USD_2024-12-24_{folder}",
        "source_workbook": payload.get("source_xlsx"),
        "generated_at_utc": utc_now(),
        "model_label": payload.get("model_label"),
        "route_id": payload.get("route_id"),
        "route_type": payload.get("route_type"),
        "runtime_status": payload.get("runtime_status"),
        "comparison": payload.get("comparison"),
        "record_count": len(records),
        "records": records,
    }


def out_path(folder: str, suffix: str) -> str:
    return f"FinFlier/{folder}/{suffix}"


def workflow_lines(payload: dict[str, Any]) -> list[str]:
    route_type = payload.get("route_type")
    lines = [
        "1. Read `data` and `text` cells from `data/src/BTC-USD_2024-12-24.xlsx`.",
        "2. Build a FinFlier-style prompt with `dist/build_btc_finflier_demo.py`.",
    ]
    if route_type == "fine_tuned_retriever":
        lines.extend(
            [
                "3. Run the fine-tuned retriever through `dist/run_experiment6_binding_generation.py --mode mixed`.",
                "4. Convert RetFact output to Binding with the configured binding converter.",
                "5. Import the best of five attempts with `dist/import_btc_experiment6_bindings.py`.",
            ]
        )
    else:
        lines.extend(
            [
                "3. Generate Binding candidates directly with `dist/run_btc_demo_model_predictions.py`.",
                "4. Keep the best of five attempts by ObjectName/DataName/Trend/Num comparison.",
            ]
        )
    lines.extend(
        [
            "6. Infer UI row/column overlays from model output with `dist/repair_btc_binding_positions.py`.",
            "7. Materialize this folder with `dist/materialize_btc_finflier_ui_folders.py`.",
        ]
    )
    return lines


def readme_text(folder: str, title: str, payload: dict[str, Any]) -> str:
    status = payload.get("runtime_status") or {}
    metadata = status.get("experiment6_metadata")
    lines = [
        f"# {title}",
        "",
        "## Usage",
        "Open `index.html` directly in a browser, or run `run_windows.bat` on Windows.",
        "",
        "## Model",
        f"- Folder: `{folder}`",
        f"- Model name: `{payload.get('model_label')}`",
        f"- Route id: `{payload.get('route_id')}`",
        f"- Route type: `{payload.get('route_type')}`",
        f"- Runtime status: `{status.get('status')}`",
        f"- Max attempts: `{status.get('max_attempts')}`",
        "",
        "## Generation Flow",
        *workflow_lines(payload),
        "",
        "## Paths",
        f"- Workspace: `{WORKSPACE_ROOT}`",
        f"- FinFlier root: `{FINFLIER_ROOT}`",
        f"- Source workbook: `{payload.get('source_xlsx')}`",
        "- Custom BTC prompt CSV: `Experiment/btc_finflier_custom/btc_20241224_rel_fact_instruction.csv`",
        "- Experiment 6 predictions: `Experiment/btc_finflier_custom/binding_eval_predictions/`",
        f"- Folder payload: `{out_path(folder, 'payload.json')}`",
        f"- UI data JSON: `{out_path(folder, 'data/narrative2.json')}`",
        "- Combined report: `FinFlier/btc_A_E_rerun_report.json`",
        "- Windows zip: `FinFlier/btc_A_E_finflier_ui.zip`",
    ]
    if metadata:
        lines.append(f"- Experiment 6 metadata: `{metadata}`")
    lines.extend(
        [
            "",
            "## Notes",
            "Gold bindings are retained only for comparison metadata. They are not substituted as model predictions.",
            "`reason` in `data/narrative2.json` preserves model-produced text after removing wrapper prefixes such as `reason:`.",
            "",
        ]
    )
    return "\n".join(lines)


def copy_static_assets(out_dir: Path, title: str, folder: str, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["app.css", "app.js"]:
        shutil.copy2(TEMPLATE_DIR / name, out_dir / name)
    index = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    index = index.replace("FinFlier Narrative2 UI", title)
    index = index.replace("Narrative2 Data Binding", title)
    index = index.replace("Narrative2", "BTC")
    (out_dir / "index.html").write_text(index, encoding="utf-8")
    readme = readme_text(folder, title, payload)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "README_WINDOWS.md").write_text(readme, encoding="utf-8")
    (out_dir / "run_windows.bat").write_text("@echo off\r\nstart \"\" \"%~dp0index.html\"\r\n", encoding="utf-8")


def materialize_folder(root: Path, folder: str) -> dict[str, Any]:
    out_dir = root / folder
    payload_path = out_dir / "payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    title = f"BTC FinFlier Binding - {folder}"
    copy_static_assets(out_dir, title, folder, payload)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = to_dataset(payload, folder)
    text = json.dumps(dataset, ensure_ascii=False, indent=2)
    (data_dir / "narrative2.json").write_text(text + "\n", encoding="utf-8")
    (data_dir / "narrative2.js").write_text("window.NARRATIVE2_DATA = " + text + ";\n", encoding="utf-8")
    return {
        "folder": folder,
        "index_html": str(out_dir / "index.html"),
        "payload_json": str(payload_path),
        "data_json": str(data_dir / "narrative2.json"),
        "runtime_status": payload.get("runtime_status"),
        "exact_match": (payload.get("comparison") or {}).get("exact_match"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folders", nargs="*", default=[folder for folder, *_ in demo.MODEL_ROUTES])
    args = parser.parse_args()
    report = {
        "generated_at_utc": utc_now(),
        "output_root": str(args.output_root),
        "template": str(TEMPLATE_DIR),
        "folders": [materialize_folder(args.output_root, folder) for folder in args.folders],
    }
    report_path = args.output_root / "btc_A_E_ui_materialize_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
