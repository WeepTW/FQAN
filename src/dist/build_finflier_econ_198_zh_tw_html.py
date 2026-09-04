#!/usr/bin/env python3
"""Materialize Econ_198 as a standalone Traditional Chinese FinFlier page."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SOURCE_ID = "Econ_198"
SOURCE_WORKBOOK = WORKSPACE_ROOT / "data" / "src" / "narratives" / "narrative2.xlsx"
BASE_BUILDER_PATH = Path(__file__).resolve().parent / "build_finflier_record_html.py"

DOCUMENT_TITLE = "FinFlier · Econ_198（繁體中文）"
FONT_CSS = (
    "\nbody { font-family: 'Noto Sans TC', 'Microsoft JhengHei', "
    "'PingFang TC', sans-serif; }\n"
)

TEXT_REPLACEMENTS = {
    "FinFlier static interface": "FinFlier 靜態介面",
    "Narrative2 Data Binding": "Narrative2 資料綁定",
    "Record": "紀錄",
    "Chart": "圖表",
    "Data chart": "資料圖表",
    "Financial data chart": "財務資料圖表",
    "Narrative text": "敘事文字",
    "Data binding": "資料綁定",
    "Matched evidence": "對應證據",
    "Reason": "理由",
    "narrative2.js was not loaded.": "未載入 narrative2.js。",
    'return "None";': 'return "無";',
    '["Rows", record.table.length]': '["資料列", record.table.length]',
    '["Columns", record.schema.columns.length]': '["欄位", record.schema.columns.length]',
    '["Bindings", record.bindings.length]': '["綁定", record.bindings.length]',
    '["Active data", active?.data_name || "None"]': '["目前資料欄", active?.data_name || "無"]',
    '"ObjectName=None"': '"ObjectName=無"',
    "No binding result.": "沒有資料綁定結果。",
    "No chartable data.": "沒有可繪製的資料。",
}

RECORD_TRANSLATION = {
    "chart_type": "單一長條圖",
    "text": "舊金山的疫前職位復甦百分比為 46%。",
    "reason": "",
    "object_names": ["舊金山"],
    "evidence_text": "舊金山的疫前職位復甦百分比為 46%。",
}


def load_base_builder() -> Any:
    spec = importlib.util.spec_from_file_location("finflier_record_html", BASE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load FinFlier builder: {BASE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replace_required(document: str, source: str, target: str) -> str:
    if source not in document:
        raise ValueError(f"Expected FinFlier template text not found: {source!r}")
    return document.replace(source, target)


def localize_payload(payload: dict[str, Any]) -> None:
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("Expected exactly one selected FinFlier record")
    record = records[0]
    if record.get("source") != SOURCE_ID:
        raise ValueError(f"Expected source {SOURCE_ID!r}; found {record.get('source')!r}")

    record["chart_type"] = RECORD_TRANSLATION["chart_type"]
    record["text"] = RECORD_TRANSLATION["text"]
    record["reason"] = RECORD_TRANSLATION["reason"]

    bindings = record.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 1:
        raise ValueError("Econ_198 must contain exactly one binding")
    binding = bindings[0]
    binding["object_names"] = list(RECORD_TRANSLATION["object_names"])
    binding["evidence_text"] = RECORD_TRANSLATION["evidence_text"]


def localize_html(document: str) -> str:
    document = replace_required(document, '<html lang="en">', '<html lang="zh-Hant-TW">')
    document = replace_required(
        document,
        f"<title>FinFlier · {SOURCE_ID}</title>",
        f"<title>{DOCUMENT_TITLE}</title>",
    )
    for source, target in TEXT_REPLACEMENTS.items():
        document = replace_required(document, source, target)
    return replace_required(document, "</style>", FONT_CSS + "</style>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--log-index", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_index and not args.run_log:
        raise ValueError("--log-index requires --run-log")
    if not SOURCE_WORKBOOK.is_file():
        raise FileNotFoundError(f"Source workbook not found: {SOURCE_WORKBOOK}")

    builder = load_base_builder()
    payload = builder.select_record(SOURCE_WORKBOOK, SOURCE_ID)
    localize_payload(payload)
    rendered_html = localize_html(builder.inline_finflier_html(payload, SOURCE_ID))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "index.html"
    manifest_path = args.output_dir / "manifest.json"
    if html_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    html_path.write_text(rendered_html, encoding="utf-8")

    record = payload["records"][0]
    manifest = {
        "time": builder.utc_now(),
        "kind": "finflier_econ_198_traditional_chinese_html",
        "status": "completed",
        "source": {
            "workbook": builder.workspace_path(SOURCE_WORKBOOK),
            "sha256": builder.sha256_file(SOURCE_WORKBOOK),
            "source_id": SOURCE_ID,
            "record_number": record["number"],
            "table_rows": len(record["table"]),
        },
        "localization": {
            "locale": "zh-Hant-TW",
            "reason": "",
            "table_content": "original English retained",
            "binding_trend": record["bindings"][0]["trend"],
            "binding_num": record["bindings"][0]["nums"],
        },
        "outputs": {
            "html": builder.workspace_path(html_path),
            "html_sha256": builder.sha256_file(html_path),
            "manifest": builder.workspace_path(manifest_path),
        },
    }
    builder.write_json(manifest_path, manifest)

    if args.run_log:
        args.run_log.parent.mkdir(parents=True, exist_ok=True)
        log = {
            "time": manifest["time"],
            "repo": str(REPO_ROOT),
            "kind": manifest["kind"],
            "status": "completed",
            "summary": (
                "Materialized a standalone Traditional Chinese FinFlier page from "
                "narrative2.xlsx Econ_198 raw table data; reason is blank and "
                "Binding Trend/Num remain original values."
            ),
            "path": builder.workspace_path(args.run_log),
            "artifact": manifest["outputs"]["html"],
            "manifest": manifest["outputs"]["manifest"],
            "tags": ["finflier", "narrative2", SOURCE_ID, "zh-Hant-TW", "static-html"],
        }
        builder.write_json(args.run_log, log)
        if args.log_index:
            builder.append_log_index(args.log_index, {
                "time": log["time"],
                "repo": log["repo"],
                "kind": log["kind"],
                "status": log["status"],
                "summary": log["summary"],
                "path": log["path"],
                "tags": log["tags"],
            })

    print(f"Wrote {html_path}")
    print(f"Wrote {manifest_path}")
    if args.run_log:
        print(f"Wrote {args.run_log}")


if __name__ == "__main__":
    main()
