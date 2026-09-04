#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path


csv.field_size_limit(10**9)

REQUIRED_COLUMNS = (
    "Sentence",
    "input",
    "Question",
    "Pre_Text",
    "Post_Text",
    "Tables",
    "Table_Text",
)


def normalize(text: str | None) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def normalize_optional_text(text: str | None) -> str:
    normalized = normalize(text)
    if normalized.lower() in {"none", "nan", "null"}:
        return ""
    return normalized


DATA_PLACEHOLDERS = (
    "{as the row[`data`] in `1_full_used_data.csv`}",
    "{FinFlier-like `data`}",
)
JSON_QUOTE_ARTIFACTS = ('""RetFact""', '""Binding""', '""Reason""')


def sanitize_cell(value: object) -> str:
    if value is None:
        return ""
    return normalize(str(value)).replace('"', "'")


def coerce_cell_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = normalize(value)
    if re.fullmatch(r"[-+]?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped):
        try:
            return float(stripped)
        except ValueError:
            return stripped
    return stripped.replace('"', "'")


def format_data_value(value: object) -> str:
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if value is None:
        return "None"
    return str(value)


def build_context(row: dict[str, str]) -> str:
    context_parts = [
        normalize_optional_text(row.get("Pre_Text")),
        normalize_optional_text(row.get("Table_Text")),
        normalize_optional_text(row.get("Post_Text")),
    ]
    context = " ".join(part for part in context_parts if part)
    return (context or normalize_optional_text(row.get("Sentence"))).rstrip("; ")


def parse_table(raw_table: str) -> object:
    try:
        return ast.literal_eval(str(raw_table or "[]"))
    except (SyntaxError, ValueError):
        return str(raw_table or "")


def contains_data_label(text: str) -> bool:
    return bool(re.search(r"(?im)^\s*(?:data|tabular\s+data)\s*:", text))


def contains_json_quote_artifact(text: str) -> bool:
    return any(artifact in text for artifact in JSON_QUOTE_ARTIFACTS)


def build_data_records_text(row: dict[str, str]) -> str:
    table = parse_table(row.get("Tables", "[]"))
    if isinstance(table, list) and table:
        headers = [
            sanitize_cell(header) or ("Position" if index == 0 else f"Column {index}")
            for index, header in enumerate(table[0])
        ]
        records = []
        for table_row in table[1:]:
            if not isinstance(table_row, list):
                continue
            pairs = []
            for index, header in enumerate(headers):
                value = coerce_cell_value(table_row[index]) if index < len(table_row) else ""
                pairs.append(f"{format_data_value(header)}:{format_data_value(value)}")
            records.append("{" + ",".join(pairs) + "}")
        return "[" + ",".join(records) + "]"
    table_text = normalize(row.get("Table_Text"))
    return table_text.replace('"', "'") if table_text else ""


def build_context_block(row: dict[str, str], context: str) -> str:
    question = normalize(row.get("Question"))
    data_text = build_data_records_text(row)
    context_line = f"{context}; question:{question}"
    if not data_text:
        return context_line
    return f"{data_text}\n{context_line}"


def extract_instruction_template(prompt_path: Path) -> str:
    prompt_text = prompt_path.read_text(encoding="utf-8-sig")
    match = re.search(r"```[ \t]*instruction[ \t]*\n(.*?)\n```", prompt_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        match = re.search(r"```[^\n]*\n(.*?)\n```", prompt_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No fenced instruction block found in {prompt_path}")

    lines = []
    for line in match.group(1).replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if line.strip() == "`prompt-type`":
            continue
        lines.append(line.rstrip())
    template = "\n".join(lines).strip()
    if "`Sentences`" not in template and "{Sentences}" not in template:
        raise ValueError("Instruction template does not contain a Sentences placeholder")
    return template + "\n"


def build_input(template: str, row: dict[str, str], context: str) -> str:
    question = normalize(row.get("Question"))
    rendered = template
    for placeholder in DATA_PLACEHOLDERS:
        rendered = rendered.replace(placeholder, build_data_records_text(row))
    return (
        rendered.replace("`Sentences`", context)
        .replace("{Sentences}", context)
        .replace("{Question}", question)
        .replace("{examples_by_prompt_type}", "")
    ).strip() + "\n"


def split_loose_records(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return [], []
    fieldnames = lines[0].rstrip("\r\n").split(",")
    records: list[str] = []
    current: list[str] = []
    for line in lines[1:]:
        if re.match(r"^\d+,", line) and current:
            records.append("".join(current).rstrip("\r\n"))
            current = [line]
        else:
            current.append(line)
    if current:
        records.append("".join(current).rstrip("\r\n"))
    return fieldnames, records


def parse_loose_record(record: str) -> list[str]:
    fields: list[str] = []
    buffer: list[str] = []
    in_quote = False
    at_field_start = True
    bracket_depth = 0
    index = 0
    while index < len(record):
        char = record[index]
        next_char = record[index + 1] if index + 1 < len(record) else ""

        if in_quote:
            if char == '"':
                if next_char == '"':
                    buffer.append('"')
                    index += 2
                    at_field_start = False
                    continue
                if bracket_depth == 0 and next_char in {",", "\n", "\r", ""}:
                    in_quote = False
                    index += 1
                    at_field_start = False
                    continue
                buffer.append(char)
                index += 1
                at_field_start = False
                continue
            if char in "[{":
                bracket_depth += 1
            elif char in "]}" and bracket_depth > 0:
                bracket_depth -= 1
            buffer.append(char)
            index += 1
            at_field_start = False
            continue

        if char == ",":
            fields.append("".join(buffer))
            buffer = []
            at_field_start = True
            index += 1
            continue
        if char == '"' and at_field_start:
            in_quote = True
            index += 1
            continue
        buffer.append(char)
        at_field_start = False
        index += 1

    fields.append("".join(buffer))
    return fields


def load_loose_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames, records = split_loose_records(csv_path.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, str]] = []
    for record_number, record in enumerate(records, start=1):
        values = parse_loose_record(record)
        if len(values) != len(fieldnames):
            raise ValueError(
                f"Could not repair row {record_number} in {csv_path}: expected {len(fieldnames)} fields, found {len(values)}"
            )
        rows.append(dict(zip(fieldnames, values)))
    return fieldnames, rows


def load_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    if any(None in row for row in rows):
        fieldnames, rows = load_loose_rows(csv_path)
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in repaired {csv_path}: {missing}")
    return fieldnames, rows


def rebuild_rows(rows: list[dict[str, str]], template: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    rebuilt_rows: list[dict[str, str]] = []
    changed_sentence = 0
    changed_input = 0
    for row in rows:
        rebuilt = dict(row)
        context = build_context(row)
        input_text = build_input(template, row, context)
        if rebuilt.get("Sentence") != context:
            changed_sentence += 1
        if rebuilt.get("input") != input_text:
            changed_input += 1
        rebuilt["Sentence"] = context
        rebuilt["input"] = input_text
        rebuilt_rows.append(rebuilt)
    return rebuilt_rows, {"Sentence": changed_sentence, "input": changed_input}


def validate_rows(fieldnames: list[str], rows: list[dict[str, str]], template: str) -> None:
    if "Setence" in fieldnames:
        raise AssertionError("Unexpected misspelled Setence column found")
    for row_number, row in enumerate(rows, start=1):
        expected_sentence = build_context(row)
        if row["Sentence"] != expected_sentence:
            raise AssertionError(f"Row {row_number} Sentence is not generated from source columns")
        input_text = row["input"]
        context_block = build_context_block(row, expected_sentence)
        for required_fragment in ("# Retriever + Data Binding", "## Context", "## Output Format"):
            if required_fragment not in input_text:
                raise AssertionError(f"Row {row_number} input is missing {required_fragment!r}")
        for placeholder in (
            "`Sentences`",
            "`prompt-type`",
            "{FinFlier-like `data`}",
            "{as the row[`data`] in `1_full_used_data.csv`}",
            "{Sentences}",
            "{Question}",
            "{examples_by_prompt_type}",
        ):
            if placeholder in input_text:
                raise AssertionError(f"Row {row_number} input still contains placeholder {placeholder}")
        if contains_data_label(input_text):
            raise AssertionError(f"Row {row_number} input contains a data label in the Context block")
        if contains_json_quote_artifact(input_text):
            raise AssertionError(f"Row {row_number} input contains parsed JSON quote artifacts")
        if contains_data_label(context_block):
            raise AssertionError(f"Row {row_number} Context block contains a data label")
        if '""""' in context_block:
            raise AssertionError(f"Row {row_number} Context block contains CSV quote artifacts")
        if re.search(r"\Squestion:", input_text):
            raise AssertionError(f"Row {row_number} input contains malformed question spacing")
        if "question:" in row["Sentence"].lower():
            raise AssertionError(f"Row {row_number} Sentence contains known spacing defect")
        expected_input = build_input(template, row, expected_sentence)
        if input_text != expected_input:
            raise AssertionError(f"Row {row_number} input is not generated from the prompt template")


def write_rows_atomic(output_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(output_path)


def build_preview(rows: list[dict[str, str]], preview_rows: int) -> list[dict[str, object]]:
    preview = []
    for index, row in enumerate(rows[:preview_rows]):
        context_block = build_context_block(row, row["Sentence"])
        preview.append(
            {
                "row": index,
                "sentence_len": len(row["Sentence"]),
                "sentence_preview": row["Sentence"][:220],
                "context_block_preview": context_block[:400],
                "input_len": len(row["input"]),
                "input_preview": row["input"][:220],
                "checks": {
                    "contains_data_label": contains_data_label(context_block),
                    "contains_quote_artifact": '""""' in context_block,
                    "uses_data_records": context_block.startswith("[{"),
                },
            }
        )
    return preview


def print_preview_text(rows: list[dict[str, str]], preview_rows: int) -> None:
    for index, row in enumerate(rows[:preview_rows]):
        print(f"=== Row {index} Sentence ===")
        print(row["Sentence"])
        print(f"=== Row {index} Context Block ===")
        print(build_context_block(row, row["Sentence"]))
        print(f"=== Row {index} Input ===")
        print(row["input"].rstrip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild full_example.csv Sentence and input columns from source fields.")
    parser.add_argument("--prompt", type=Path, default=Path("new_prompt.txt"))
    parser.add_argument("--source", type=Path, default=Path("data/src/old_full_example.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/src/full_example.csv"))
    parser.add_argument("--csv", type=Path, default=None, help="Backward-compatible in-place CSV path.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-rows", type=int, default=3)
    parser.add_argument("--preview-text", action="store_true")
    args = parser.parse_args()

    source_path = args.csv if args.csv is not None else args.source
    output_path = args.csv if args.csv is not None else args.output

    fieldnames, source_rows = load_rows(source_path)
    template = extract_instruction_template(args.prompt)
    rebuilt_rows, changed = rebuild_rows(source_rows, template)
    validate_rows(fieldnames, rebuilt_rows, template)

    if args.preview_text:
        print_preview_text(rebuilt_rows, max(args.preview_rows, 0))
        return

    if not args.dry_run:
        write_rows_atomic(output_path, fieldnames, rebuilt_rows)

    report = {
        "source": str(source_path),
        "output": str(output_path),
        "prompt": str(args.prompt),
        "dry_run": args.dry_run,
        "rows": len(rebuilt_rows),
        "fieldnames": fieldnames,
        "changed": changed,
        "preview": build_preview(rebuilt_rows, max(args.preview_rows, 0)),
        "status": "validated",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
