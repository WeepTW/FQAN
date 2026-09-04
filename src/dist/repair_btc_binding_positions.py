#!/usr/bin/env python3
"""Repair model Binding Position fields to row-column cells using DataName and Num."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import build_btc_finflier_demo as demo


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINFLIER_ROOT = REPO_ROOT / "FinFlier"


def as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def data_names(binding: dict[str, Any]) -> list[str]:
    value = binding.get("DataName")
    names: list[str] = []
    for item in as_list(value):
        names.extend(part.strip() for part in str(item).split(",") if part.strip())
    return names


def numeric(value: Any) -> float | None:
    try:
        if value in (None, "", "None"):
            return None
        return float(str(value).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def num_values(binding: dict[str, Any]) -> list[float]:
    values = []
    for item in as_list(binding.get("Num")):
        number = numeric(item)
        if number is not None:
            values.append(number)
    return values


def close_enough(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-6, abs(a) * 1e-4, abs(b) * 1e-4)


def find_value_position(rows: list[dict[str, Any]], columns: list[str], candidate_cols: list[str], value: float, used: set[tuple[int, int]]) -> tuple[int, int] | None:
    search_cols = [col for col in candidate_cols if col in columns]
    search_cols.extend(col for col in columns if col not in search_cols)
    best: tuple[float, int, int] | None = None
    for col in search_cols:
        col_index = columns.index(col)
        for row_index, row in enumerate(rows):
            cell = numeric(row.get(col))
            if cell is None:
                continue
            diff = abs(cell - value)
            if not close_enough(cell, value):
                continue
            penalty = 0 if col in candidate_cols else 1
            key = (penalty, diff, row_index, col_index)
            if best is None or key < (best[0], best[1], best[2], best[3]):
                best = (float(penalty), diff, row_index, col_index)
    if best is None:
        return None
    pos = (best[2], best[3])
    if pos in used:
        return pos
    used.add(pos)
    return pos


def valid_row_col_position(binding: dict[str, Any], rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, list[int]]]:
    positions = binding.get("Position") or []
    repaired: list[dict[str, list[int]]] = []
    if isinstance(positions, list):
        for item in positions:
            if not isinstance(item, dict):
                continue
            begin = item.get("Begin")
            end = item.get("End")
            if isinstance(begin, list) and isinstance(end, list) and len(begin) >= 2 and len(end) >= 2:
                try:
                    b = [int(begin[0]), int(begin[1])]
                    e = [int(end[0]), int(end[1])]
                except (TypeError, ValueError):
                    continue
                if 0 <= b[0] < len(rows) and 0 <= e[0] < len(rows) and 0 <= b[1] < len(columns) and 0 <= e[1] < len(columns):
                    repaired.append({"Begin": b, "End": e})
    return repaired


def repair_binding(binding: dict[str, Any], rows: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
    item = dict(binding)
    existing = valid_row_col_position(item, rows, columns)
    names = data_names(item)
    values = num_values(item)
    used: set[tuple[int, int]] = set()
    inferred = []
    for value in values:
        pos = find_value_position(rows, columns, names, value, used)
        if pos is not None:
            inferred.append({"Begin": [pos[0], pos[1]], "End": [pos[0], pos[1]]})
    if inferred:
        item["Position"] = inferred
        item["PositionRepair"] = {"status": "repaired_from_DataName_Num", "original": binding.get("Position")}
    elif existing:
        item["Position"] = existing
        item["PositionRepair"] = {"status": "kept_valid_row_col"}
    else:
        item["Position"] = []
        item["PositionRepair"] = {"status": "unresolved_no_row_col_match", "original": binding.get("Position")}
    item.setdefault("result", item.get("Text") or "")
    return item


def repair_folder(root: Path, folder: str) -> dict[str, Any]:
    payload_path = root / folder / "payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    counts = []
    predictions_for_compare = []
    for case in payload.get("cases", []):
        columns = case.get("schema", {}).get("columns", [])
        rows = case.get("table", [])
        repaired = [repair_binding(binding, rows, columns) for binding in case.get("model_prediction") or []]
        case["model_prediction"] = repaired
        case["result"] = repaired
        case["gold_result"] = case.get("gold_binding") or []
        case["model_comparison"] = demo.compare_binding_fields(case.get("gold_binding") or [], repaired)
        predictions_for_compare.append(repaired)
        counts.append({
            "case_id": case.get("case_id"),
            "bindings": len(repaired),
            "positions": sum(len(binding.get("Position") or []) for binding in repaired),
            "unresolved": sum(1 for binding in repaired if (binding.get("PositionRepair") or {}).get("status") == "unresolved_no_row_col_match"),
        })
    payload["comparison"] = demo.compare_cases(payload.get("cases", []), predictions_for_compare)
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / folder / "index.html").write_text(demo.render_html(payload), encoding="utf-8")
    return {"folder": folder, "position_summary": counts, "exact_match": payload["comparison"].get("exact_match")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finflier-root", type=Path, default=DEFAULT_FINFLIER_ROOT)
    parser.add_argument("--folders", nargs="*", default=["A", "B", "C", "D", "E"])
    args = parser.parse_args()
    print(json.dumps([repair_folder(args.finflier_root, folder) for folder in args.folders], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
