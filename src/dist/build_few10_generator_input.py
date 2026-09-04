"""Build generator input JSON for CSV-backed generator smoke paths.

Formal retriever-conditioned generator runs should consume matched retriever
artifacts. CSV Rel_Fact fallback is kept only for the few10 Qwen smoke route
and requires an explicit flag so it is not confused with a formal data flow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


FORMAL_CSV_SOURCE_MODES = {"finqa10_formal_smoke", "finqa_train_formal", "narratives5_formal_smoke"}
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT") or REPO_ROOT.parent)


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def selection_key(index: int, question: str) -> str:
    return f"row:{index}|question:{normalize_question(question)}"


def parse_answer(value: str) -> Any:
    text = str(value).strip()
    if text.lower() in {"yes", "no"}:
        return text.lower()
    normalized = text.replace(",", "").replace("$", "").replace("%", "").strip()
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", normalized):
        number = float(normalized)
        return int(number) if number.is_integer() else number
    return text


def split_retrieved(value: str) -> list[str]:
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    return parts if parts else [str(value).strip()]


def csv_id_prefix(source_mode: str) -> str:
    if source_mode == "finqa10_formal_smoke":
        return "finqa10_formal"
    if source_mode == "finqa_train_formal":
        return "finqa_train_formal"
    if source_mode == "narratives5_formal_smoke":
        return "narratives5_formal"
    return "finqa_few10"


def flow_scope_for_csv_source(source_mode: str) -> str:
    if source_mode in FORMAL_CSV_SOURCE_MODES:
        return "formal_csv_relfact_generator"
    return "isolated_generator_smoke"


def csv_stable_row(index: int, row: dict[str, str]) -> str | int:
    raw_index = str(row.get("") or "").strip()
    if raw_index:
        return int(raw_index) if raw_index.isdigit() else raw_index
    return index


def row_to_csv_example(index: int, row: dict[str, str], source_csv: Path, source_mode: str) -> dict[str, Any]:
    question = row.get("Question", "")
    if not question and row.get("Narrative_Text"):
        question = "What data-text bindings are described in the narrative?"
    source_csv_row = csv_stable_row(index, row)
    stable_id = row.get("id") or str(source_csv_row) or f"{csv_id_prefix(source_mode)}_{index:06d}"
    return {
        "id": stable_id,
        "source_id": stable_id,
        "selection_key": selection_key(source_csv_row, question),
        "normalized_question": normalize_question(question),
        "question": question,
        "answer": parse_answer(row.get("GT_Answer", "")),
        "program": row.get("GT_Program", ""),
        "text": row.get("Sentence") or " ".join(
            part for part in [row.get("Pre_Text", ""), row.get("Post_Text", "")] if part
        ),
        "table": row.get("Tables", ""),
        "table_text": row.get("Table_Text", ""),
        "retrieved": split_retrieved(row.get("Rel_Fact") or row.get("Binding_Result", "")),
        "binding_result": row.get("Binding_Result", ""),
        "binding_reason": row.get("Binding_Reason", ""),
        "prompt_mode": row.get("Prompt_Mode", ""),
        "generator_model": row.get("Generator_Model", ""),
        "source_csv_row": source_csv_row,
        "source_csv_position": index,
        "source": str(source_csv),
        "retrieved_source": source_mode,
        "flow_scope": flow_scope_for_csv_source(source_mode),
    }


def row_to_smoke_example(index: int, row: dict[str, str], source_csv: Path) -> dict[str, Any]:
    return row_to_csv_example(index, row, source_csv, "csv_rel_fact_smoke_only")


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def infer_finqa_split_from_path(path: Path) -> str | None:
    value = str(path).replace("\\", "/").lower()
    if "finqa_dev" in value:
        return "dev"
    if "finqa_test" in value:
        return "test"
    if "finqa_train" in value:
        return "train"
    return None


def default_finqa_gold_csv(split: str) -> Path | None:
    candidates = [
        WORKSPACE_ROOT / "data" / "src" / "FINDER" / f"finqa_{split}_rel_fact_instruction.csv",
        WORKSPACE_ROOT / "data" / "finqa_original" / f"finqa_{split}_rel_fact_instruction.csv",
        WORKSPACE_ROOT / "data" / "finqa" / f"finqa_{split}_rel_fact_instruction.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def default_finqa_gold_json(split: str) -> Path | None:
    candidates = [
        WORKSPACE_ROOT / "data" / "src" / "FinQA" / f"{split}.json",
        WORKSPACE_ROOT / "src" / "code" / "apollo" / "dataset" / "FinQA" / f"{split}.json",
    ]
    if split == "test":
        candidates.extend(
            [
                WORKSPACE_ROOT / "src" / "code" / "Data" / "Data_Target_Module" / "Finqa" / "finqa_test.json",
                WORKSPACE_ROOT / "src" / "code" / "apollo" / "dataset" / "FinQA" / "test.json",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def stringify_table(table: Any) -> str:
    if isinstance(table, str):
        return table
    if isinstance(table, list):
        lines = []
        for row in table:
            if isinstance(row, list):
                lines.append(" | ".join(str(cell) for cell in row))
            else:
                lines.append(str(row))
        return "\n".join(lines)
    return str(table) if table is not None else ""


def json_gold_to_example(index: int, row: dict[str, Any], source_json: Path, split: str) -> dict[str, Any]:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    question = row.get("question") or qa.get("question") or ""
    answer = row.get("answer")
    if is_blank(answer):
        answer = qa.get("answer")
    if is_blank(answer):
        answer = qa.get("exe_ans")
    program = row.get("program") or qa.get("program") or qa.get("program_re") or ""
    stable_id = row.get("id") or f"finqa_{split}_gold_{index:06d}"
    text_parts = []
    for key in ("pre_text", "post_text"):
        value = row.get(key)
        if isinstance(value, list):
            text_parts.extend(str(item) for item in value)
        elif value:
            text_parts.append(str(value))
    return {
        "id": stable_id,
        "source_id": stable_id,
        "selection_key": selection_key(index, question),
        "normalized_question": normalize_question(question),
        "question": question,
        "answer": parse_answer(answer),
        "program": program,
        "text": " ".join(part for part in text_parts if part),
        "table": stringify_table(row.get("table")),
        "table_text": stringify_table(row.get("table")),
        "source_csv_row": index,
        "source_csv_position": index,
        "source": str(source_json),
        "retrieved_source": f"finqa_{split}_formal_gold_json",
        "flow_scope": "formal_json_gold_recovery",
    }


def load_finqa_gold_rows_from_json(split: str) -> tuple[list[dict[str, Any]], Path | None]:
    json_path = default_finqa_gold_json(split)
    if json_path is None:
        return [], None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return [], None
    return [json_gold_to_example(index, row, json_path, split) for index, row in enumerate(payload)], json_path


def load_finqa_gold_rows_for_matched(source_json: Path) -> tuple[list[dict[str, Any]], Path | None]:
    split = infer_finqa_split_from_path(source_json)
    if split is None:
        return [], None
    json_rows, json_path = load_finqa_gold_rows_from_json(split)
    if json_rows:
        return json_rows, json_path
    csv_path = default_finqa_gold_csv(split)
    if csv_path is None:
        return [], None
    csv.field_size_limit(sys.maxsize)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row_to_csv_example(index, row, csv_path, f"finqa_{split}_formal_gold")
            for index, row in enumerate(csv.DictReader(handle))
        ]
    return rows, csv_path


def gold_lookup_by_question(gold_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in gold_rows:
        question = normalize_question(str(row.get("question", "")))
        if question and question not in lookup:
            lookup[question] = row
    return lookup


def matching_gold_row(
    index: int,
    item: dict[str, Any],
    gold_rows: list[dict[str, Any]],
    by_question: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    question = normalize_question(str(item.get("question", "")))
    if index < len(gold_rows):
        indexed = gold_rows[index]
        if not question or normalize_question(str(indexed.get("question", ""))) == question:
            return indexed
    return by_question.get(question)


def retrieved_from_matched(row: dict[str, Any]) -> list[str]:
    retrieved = row.get("retrieved")
    if isinstance(retrieved, list):
        return [str(item).strip() for item in retrieved if str(item).strip()]
    if isinstance(retrieved, str) and retrieved.strip():
        return split_retrieved(retrieved)
    scored = row.get("retrieved_with_scores")
    if isinstance(scored, list):
        texts = []
        for item in scored:
            if isinstance(item, dict) and str(item.get("text", "")).strip():
                texts.append(str(item["text"]).strip())
        return texts
    return []


def matched_to_example(
    index: int,
    row: dict[str, Any],
    source_json: Path,
    gold_row: dict[str, Any] | None = None,
    gold_csv: Path | None = None,
) -> dict[str, Any]:
    item = dict(row)
    if gold_row is not None:
        for key in (
            "id",
            "source_id",
            "selection_key",
            "normalized_question",
            "answer",
            "program",
            "table",
            "table_text",
            "text",
            "source_csv_row",
            "source_csv_position",
        ):
            if is_blank(item.get(key)) and not is_blank(gold_row.get(key)):
                item[key] = gold_row[key]
        if gold_csv is not None:
            item["target_gold_csv"] = str(gold_csv)
            item["target_gold_recovered_from_csv"] = True
    item["retrieved"] = retrieved_from_matched(row)
    item["source_matched_json_row"] = index
    item["source"] = str(source_json)
    item["retrieved_source"] = "matched_retriever_artifact"
    item["flow_scope"] = "formal_retriever_conditioned_generator"
    return item


def load_matched_examples(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Matched generator input must be a JSON list.")
    rows = payload if limit < 0 else payload[:limit]
    gold_rows, gold_csv = load_finqa_gold_rows_for_matched(path)
    by_question = gold_lookup_by_question(gold_rows)
    examples = []
    for index, row in enumerate(rows):
        gold_row = matching_gold_row(index, row, gold_rows, by_question)
        examples.append(matched_to_example(index, row, path, gold_row, gold_csv))
    return examples


def load_csv_examples(path: Path, limit: int, source_mode: str) -> list[dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit >= 0:
        rows = rows[:limit]
    return [row_to_csv_example(index, row, path, source_mode) for index, row in enumerate(rows)]


def load_smoke_examples(path: Path, limit: int) -> list[dict[str, Any]]:
    return load_csv_examples(path, limit, "csv_rel_fact_smoke_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FinQA generator input JSON.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-json",
        type=Path,
        help="Formal matched retriever artifact to pass through as generator input.",
    )
    source.add_argument(
        "--input-csv",
        type=Path,
        help="Few10 CSV source for isolated smoke only; requires --allow-relfact-smoke.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--allow-relfact-smoke",
        action="store_true",
        help="Allow CSV Rel_Fact to seed retrieved facts for isolated smoke tests only.",
    )
    parser.add_argument(
        "--formal-csv-source",
        choices=sorted(FORMAL_CSV_SOURCE_MODES),
        default="",
        help="Treat --input-csv as a formal CSV-backed Experiment 7 source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_json is not None:
        examples = load_matched_examples(args.input_json, args.limit)
        source_mode = "matched_retriever_artifact"
    else:
        if args.formal_csv_source:
            examples = load_csv_examples(args.input_csv, args.limit, args.formal_csv_source)
            source_mode = args.formal_csv_source
        elif not args.allow_relfact_smoke:
            raise SystemExit(
                "--input-csv reads Rel_Fact directly and is smoke-only; "
                "use --allow-relfact-smoke or provide --input-json with a matched retriever artifact."
            )
        else:
            examples = load_smoke_examples(args.input_csv, args.limit)
            source_mode = "csv_rel_fact_smoke_only"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(examples), "output_json": str(args.output_json), "source_mode": source_mode}, indent=2))


if __name__ == "__main__":
    main()
