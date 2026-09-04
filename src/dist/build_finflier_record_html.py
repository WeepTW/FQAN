#!/usr/bin/env python3
"""Materialize one Narrative2 record as a standalone FinFlier HTML page."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
FINFLIER_UI_ROOT = REPO_ROOT / "FinFlier" / "narrative2_ui"
PAYLOAD_BUILDER_PATH = FINFLIER_UI_ROOT / "scripts" / "build_narrative2_ui_data.py"
DEFAULT_SOURCE_WORKBOOK = WORKSPACE_ROOT / "data" / "src" / "narratives" / "narrative2.xlsx"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path.resolve())


def load_payload_builder() -> Any:
    spec = importlib.util.spec_from_file_location("finflier_payload_builder", PAYLOAD_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load FinFlier payload builder: {PAYLOAD_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_record(source_workbook: Path, source_id: str) -> dict[str, Any]:
    builder = load_payload_builder()
    payload = builder.build_payload(source_workbook)
    selected = [record for record in payload["records"] if record.get("source") == source_id]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one record for {source_id!r}; found {len(selected)}")
    return {
        "dataset": payload["dataset"],
        "source_workbook": str(source_workbook.resolve()),
        "generated_at_utc": utc_now(),
        "record_count": 1,
        "records": selected,
    }


def inline_finflier_html(payload: dict[str, Any], source_id: str) -> str:
    index_template = (FINFLIER_UI_ROOT / "index.html").read_text(encoding="utf-8")
    style = (FINFLIER_UI_ROOT / "app.css").read_text(encoding="utf-8")
    app_script = (FINFLIER_UI_ROOT / "app.js").read_text(encoding="utf-8")
    title = f"FinFlier · {source_id}"
    payload_script = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")

    rendered = index_template.replace(
        "<title>FinFlier Narrative2 UI</title>",
        f"<title>{html.escape(title)}</title>",
    )
    rendered = rendered.replace(
        '<link rel="stylesheet" href="./app.css">',
        f"<style>\n{style}\n</style>",
    )
    external_scripts = '    <script src="./data/narrative2.js"></script>\n    <script src="./app.js"></script>'
    inline_scripts = (
        f"    <script>window.NARRATIVE2_DATA = {payload_script};</script>\n"
        f"    <script>\n{app_script}\n</script>"
    )
    if external_scripts not in rendered:
        raise RuntimeError("FinFlier index template does not contain the expected data and app scripts")
    return rendered.replace(external_scripts, inline_scripts)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_log_index(index_path: Path, entry: dict[str, Any]) -> None:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Log index has no entries list: {index_path}")
    if any(existing.get("path") == entry["path"] for existing in entries if isinstance(existing, dict)):
        raise ValueError(f"Log index already contains {entry['path']}")
    entries.append(entry)
    write_json(index_path, index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-workbook", type=Path, default=DEFAULT_SOURCE_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--log-index", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_index and not args.run_log:
        raise ValueError("--log-index requires --run-log")
    if not args.source_workbook.is_file():
        raise FileNotFoundError(f"Source workbook not found: {args.source_workbook}")

    payload = select_record(args.source_workbook, args.source_id)
    record = payload["records"][0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "index.html"
    manifest_path = args.output_dir / "manifest.json"
    html_path.write_text(inline_finflier_html(payload, args.source_id), encoding="utf-8")

    manifest = {
        "time": utc_now(),
        "kind": "finflier_standalone_record_html",
        "status": "completed",
        "source": {
            "workbook": workspace_path(args.source_workbook),
            "sha256": sha256_file(args.source_workbook),
            "source_id": args.source_id,
            "record_number": record["number"],
            "chart_type": record["chart_type"],
            "table_rows": len(record["table"]),
        },
        "finflier_ui": {
            "template": workspace_path(FINFLIER_UI_ROOT / "index.html"),
            "template_sha256": sha256_file(FINFLIER_UI_ROOT / "index.html"),
            "style_sha256": sha256_file(FINFLIER_UI_ROOT / "app.css"),
            "script_sha256": sha256_file(FINFLIER_UI_ROOT / "app.js"),
            "payload_builder": workspace_path(PAYLOAD_BUILDER_PATH),
        },
        "outputs": {
            "html": workspace_path(html_path),
            "html_sha256": sha256_file(html_path),
            "manifest": workspace_path(manifest_path),
        },
    }
    write_json(manifest_path, manifest)

    if args.run_log:
        args.run_log.parent.mkdir(parents=True, exist_ok=True)
        log = {
            "time": manifest["time"],
            "repo": str(REPO_ROOT),
            "kind": "finflier_econ_311_static_html",
            "status": "completed",
            "summary": "Materialized a standalone FinFlier page from narrative2.xlsx Econ_311 raw table data.",
            "path": workspace_path(args.run_log),
            "artifact": manifest["outputs"]["html"],
            "manifest": manifest["outputs"]["manifest"],
            "source_workbook_sha256": manifest["source"]["sha256"],
            "tags": ["finflier", "narrative2", args.source_id, "static-html"],
        }
        write_json(args.run_log, log)
        if args.log_index:
            append_log_index(args.log_index, {
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
