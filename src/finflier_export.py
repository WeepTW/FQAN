"""Export validated retriever artifacts into a FinFlier-style interface payload.

This FQAN adapter emits a portable payload without bundling the FinFlier source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in matched artifact: {path}")
    if limit >= 0:
        rows = rows[:limit]
    return rows


def validate_rows(rows: list[dict[str, Any]], allow_schema_failure: bool) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows):
        if row.get("retriever_schema_failure") and not allow_schema_failure:
            errors.append(f"row {index} has retriever_schema_failure")
        retrieved = row.get("retrieved")
        if not isinstance(retrieved, list) or not retrieved:
            errors.append(f"row {index} has no retrieved RetFact text")
        scored = row.get("retrieved_with_scores")
        if not isinstance(scored, list) or not any(
            item.get("matched_by") == "prediction_fragment" for item in scored if isinstance(item, dict)
        ):
            errors.append(f"row {index} has no primary RetFact match score")
    return errors


def row_to_interface(row: dict[str, Any], index: int) -> dict[str, Any]:
    retrieved = [str(item) for item in row.get("retrieved", [])]
    primary_scores = [
        score
        for score in row.get("retrieved_with_scores", [])
        if isinstance(score, dict) and score.get("matched_by") == "prediction_fragment"
    ]
    return {
        "CaseId": row.get("id") or row.get("uid") or index,
        "Question": row.get("question", ""),
        "OriginText": row.get("text", ""),
        "TableText": row.get("table_text", ""),
        "RetFact": retrieved,
        "Binding": row.get("Binding", []),
        "ConversationInfo": [
            {
                "Position": [],
                "Text": fact,
                "OverTag": 1,
                "Type": "RetFact",
            }
            for fact in retrieved
        ],
        "GraphicalOverlay": [],
        "RetrieverScores": primary_scores,
    }


def write_index_html(path: Path, payload_file: str) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FQAN FinFlier Interface Export</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; line-height: 1.45; }}
    pre {{ white-space: pre-wrap; border: 1px solid #ddd; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>FQAN FinFlier Interface Export</h1>
  <p>Payload: <code>{payload_file}</code></p>
  <pre id="payload">Loading...</pre>
  <script>
    fetch("./{payload_file}")
      .then((response) => response.json())
      .then((payload) => {{
        document.getElementById("payload").textContent = JSON.stringify(payload, null, 2);
      }});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export matched RetFact artifacts to FinFlier-style UI payload.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--allow-schema-failure", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_json, args.limit)
    errors = validate_rows(rows, args.allow_schema_failure)
    payload = {
        "input_json": str(args.input_json),
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "errors": errors,
        "status": "ready" if not errors else "blocked",
    }
    if args.execute and not errors:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        interface_rows = [row_to_interface(row, index) for index, row in enumerate(rows)]
        output_json = args.output_dir / "output.json"
        manifest_json = args.output_dir / "manifest.json"
        output_json.write_text(json.dumps(interface_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_index_html(args.output_dir / "index.html", output_json.name)
        manifest = {
            **payload,
            "output_json": str(output_json),
            "index_html": str(args.output_dir / "index.html"),
            "kind": "finflier_interface_export",
        }
        manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payload.update(
            {
                "status": "completed",
                "output_json": str(output_json),
                "index_html": str(args.output_dir / "index.html"),
                "manifest_json": str(manifest_json),
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
