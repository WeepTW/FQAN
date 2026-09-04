#!/usr/bin/env python3
"""Build Experiment 7 generator inputs with selected in-context examples.

Formal FINDER target-answer computation needs retrieved facts plus dynamic
in-context examples before the target question.  This entrypoint keeps that
contract explicit:

* the independent selection stage may write a stable selection cache;
* formal final answer computation reads that cache and backfills examples from
  the target prompt-type train CSV;
* deterministic similarity remains available only for isolated smoke routes.

A formal Experiment 7 final computation must run with formal_finder_ready=true
and must not silently fall back when the selection cache is absent or invalid.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from build_few10_generator_input import (
    FORMAL_CSV_SOURCE_MODES,
    load_csv_examples,
    load_matched_examples,
    load_smoke_examples,
    normalize_question,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
PROMPT_TYPE_TRAIN_CSV = {
    "raw": WORKSPACE_ROOT / "data" / "src" / "FINDER" / "finqa_train_rel_fact_instruction.csv",
    "original": WORKSPACE_ROOT / "data" / "finqa_original" / "finqa_train_rel_fact_instruction.csv",
    "zero-shot": WORKSPACE_ROOT / "data" / "finqa_zero_shot" / "finqa_train_rel_fact_instruction.csv",
    "many-shot": WORKSPACE_ROOT / "data" / "finqa_many_shot" / "finqa_train_rel_fact_instruction.csv",
    "dynamic-shot": WORKSPACE_ROOT / "data" / "finqa_dynamic_shot" / "finqa_train_rel_fact_instruction.csv",
}
DEFAULT_CANDIDATE_JSON = (
    REPO_ROOT / ".external/FINDER/In_Context_Selection" / "Data_Prompt_Dynamic" / "finqa_train promptpg_samples.json"
)
DEFAULT_MATCHED_JSON = (
    REPO_ROOT
    / "Experiment"
    / "finqa_flan_o"
    / "retriever"
    / "outputs"
    / "best_matched_with_retrieved_facts_and_questions.json"
)
DEFAULT_PROMPTPG_OUTPUT_DIR = REPO_ROOT / "Experiment" / "promptpg_policy_outputs"
DEFAULT_PROMPTPG_CKPT_ROOT = REPO_ROOT / "checkpoints"
FORMAL_SOURCE_MODES = {"matched_retriever_artifact"} | FORMAL_CSV_SOURCE_MODES
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}


def env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Experiment 7 selected examples for generator input.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-json", type=Path, default=env_path("EXAMPLE_SELECTION_INPUT_JSON"))
    source.add_argument("--input-csv", type=Path, default=env_path("EXAMPLE_SELECTION_INPUT_CSV"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=env_path("EXAMPLE_SELECTION_OUTPUT_JSON"),
        help="Defaults to Experiment/${EXPT_ID}/example_selection/generator_input.json.",
    )
    parser.add_argument("--limit", type=int, default=env_int("LIMIT", 1))
    parser.add_argument("--shot-number", type=int, default=env_int("EXAMPLE_SELECTION_SHOT_NUMBER", 4))
    parser.add_argument(
        "--selection-mode",
        choices=["auto", "policy", "similarity", "cache", "none"],
        default=os.environ.get("EXAMPLE_SELECTION_MODE", "auto"),
        help="cache is the formal final-computation path; auto/similarity are smoke-only unless policy output is present.",
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        default=env_path("EXAMPLE_SELECTION_CANDIDATE_JSON") or DEFAULT_CANDIDATE_JSON,
        help="FinQA training candidate examples used for similarity fallback.",
    )
    parser.add_argument(
        "--policy-output",
        type=Path,
        default=env_path("EXAMPLE_SELECTION_POLICY_OUTPUT"),
        help="Optional PromptPG/policy selection output as JSON or CSV.",
    )
    parser.add_argument(
        "--promptpg-ckpt",
        type=Path,
        default=env_path("PROMPTPG_CKPT"),
        help="Optional learned PromptPG checkpoint. Used to build --policy-output when policy output is absent.",
    )
    parser.add_argument(
        "--promptpg-output-dir",
        type=Path,
        default=env_path("PROMPTPG_OUTPUT_DIR") or DEFAULT_PROMPTPG_OUTPUT_DIR,
        help="Directory for generated PromptPG policy JSON/CSV artifacts.",
    )
    parser.add_argument(
        "--promptpg-generated-policy-output",
        type=Path,
        default=env_path("PROMPTPG_GENERATED_POLICY_OUTPUT"),
        help="Exact PromptPG policy JSON path to create when extracting from --promptpg-ckpt.",
    )
    parser.add_argument(
        "--require-policy",
        action="store_true",
        default=os.environ.get("EXAMPLE_SELECTION_REQUIRE_POLICY", "0") == "1",
        help="Fail unless --policy-output exists and yields selected examples.",
    )
    parser.add_argument(
        "--allow-relfact-smoke",
        action="store_true",
        default=os.environ.get("ALLOW_RELFACT_SMOKE", "0") == "1",
        help="Allow CSV Rel_Fact fallback for isolated smoke only.",
    )
    parser.add_argument(
        "--formal-csv-source",
        choices=sorted(FORMAL_CSV_SOURCE_MODES),
        default=os.environ.get("EXAMPLE_SELECTION_FORMAL_CSV_SOURCE", "").strip(),
        help="Treat --input-csv as a formal CSV-backed Experiment 7 source.",
    )
    parser.add_argument(
        "--selection-cache-json",
        type=Path,
        default=env_path("EXAMPLE_SELECTION_CACHE_JSON"),
        help="Selection cache to write in selection stage or read in cache-only final computation.",
    )
    parser.add_argument(
        "--materialized-selection-jsonl",
        type=Path,
        default=env_path("EXAMPLE_SELECTION_MATERIALIZED_JSONL"),
        help="Pre-materialized target selected examples derived from a formal selection cache.",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        default=os.environ.get("EXAMPLE_SELECTION_REQUIRE_CACHE", "0") == "1",
        help="Fail with blocked_example_selection_cache unless every target row is resolved from cache.",
    )
    parser.add_argument(
        "--formal-finder-ready",
        action="store_true",
        default=os.environ.get("FORMAL_FINDER_READY", "0") == "1",
        help="Mark cache-backed final computation as formal only when cache resolution succeeds without fallback.",
    )
    parser.add_argument(
        "--allow-legacy-selection-binding",
        action="store_true",
        default=os.environ.get("ALLOW_LEGACY_SELECTION_BINDING", "0") == "1",
        help="Allow row-number/cache-id aliases for legacy smoke only. Formal runs keep this disabled.",
    )
    parser.add_argument(
        "--target-prompt-type",
        choices=sorted(PROMPT_TYPE_TRAIN_CSV),
        default=os.environ.get("TARGET_PROMPT_TYPE", "").strip() or None,
        help="Prompt type whose train CSV should backfill selected ids in cache mode.",
    )
    parser.add_argument(
        "--prompt-type-train-csv",
        type=Path,
        default=env_path("PROMPT_TYPE_TRAIN_CSV"),
        help="Explicit prompt-type train CSV used to backfill selected ids in cache mode.",
    )
    parser.add_argument(
        "--promptpg-test-file",
        type=Path,
        default=env_path("PROMPTPG_TEST_FILE"),
        help="PromptPG test JSON to use when extracting examples from --promptpg-ckpt.",
    )
    parser.add_argument(
        "--promptpg-train-file",
        type=Path,
        default=env_path("PROMPTPG_TRAIN_FILE"),
        help="Optional PromptPG train candidate JSON override for examples_extraction.py.",
    )
    return parser.parse_args()


def default_output() -> Path:
    expt_id = os.environ.get("EXPT_ID", "experiment_7_example_selection")
    return Path("Experiment") / expt_id / "example_selection" / "generator_input.json"


def default_input_json() -> Path | None:
    override = env_path("EXAMPLE_SELECTION_DEFAULT_INPUT_JSON")
    if override and override.exists():
        return override
    if DEFAULT_MATCHED_JSON.exists():
        return DEFAULT_MATCHED_JSON
    return None


def resolve_promptpg_ckpt(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return DEFAULT_PROMPTPG_CKPT_ROOT / path


def selection_key_for_row(row: dict[str, Any]) -> str:
    existing = stringify(row.get("selection_key")).strip()
    if existing:
        return existing
    source_row = row.get("source_csv_row")
    question = stringify(row.get("question"))
    if source_row is not None and question.strip():
        return f"row:{source_row}|question:{normalize_question(question)}"
    row_id = stringify(row.get("id") or row.get("source_id")).strip()
    if row_id:
        return row_id
    return normalize_question(question)


def question_key(question: Any) -> str:
    return f"question:{normalize_question(stringify(question))}"


def promptpg_problem_from_row(row: dict[str, Any]) -> dict[str, Any]:
    key = selection_key_for_row(row)
    rel_fact = row.get("rel_fact") or row.get("retrieved") or row.get("text") or ""
    if isinstance(rel_fact, list):
        rel_fact = "\n".join(stringify(item) for item in rel_fact)
    return {
        "id": key,
        "source_id": row.get("id") or row.get("source_id") or key,
        "source_csv_row": row.get("source_csv_row"),
        "selection_key": key,
        "question": stringify(row.get("question")),
        "answer": row.get("answer"),
        "program": stringify(row.get("program")),
        "table": stringify(row.get("table") or row.get("table_text")),
        "table_text": stringify(row.get("table_text") or row.get("table")),
        "rel_fact": stringify(rel_fact),
        "text": stringify(rel_fact),
        "retrieved": [stringify(rel_fact)] if stringify(rel_fact).strip() else [],
    }


def write_promptpg_test_file(path: Path, rows: list[dict[str, Any]], source_mode: str) -> Path:
    payload = {selection_key_for_row(row): promptpg_problem_from_row(row) for row in rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = {
        "kind": "promptpg_test_file",
        "source_mode": source_mode,
        "rows": len(rows),
        "output_json": str(path),
    }
    path.with_suffix(".status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def maybe_build_promptpg_policy(args: argparse.Namespace, rows: list[dict[str, Any]], source_mode: str) -> Path | None:
    if args.policy_output and args.policy_output.exists():
        return args.policy_output
    ckpt_path = resolve_promptpg_ckpt(args.promptpg_ckpt)
    if ckpt_path is None:
        if args.require_policy:
            raise SystemExit(
                "promptpg_checkpoint_missing: EXAMPLE_SELECTION_REQUIRE_POLICY=1 but PROMPTPG_CKPT/--promptpg-ckpt is unset."
            )
        return None
    if not ckpt_path.exists():
        if args.require_policy:
            raise SystemExit(f"promptpg_checkpoint_missing: {ckpt_path}")
        return None

    expt_id = os.environ.get("EXPT_ID", "experiment_7_example_selection")
    if args.promptpg_generated_policy_output is not None:
        policy_json = args.promptpg_generated_policy_output
        output_dir = policy_json.parent
        policy_csv = output_dir / "promptpg_selected_examples.csv"
    else:
        output_dir = args.promptpg_output_dir / expt_id
        policy_json = output_dir / "promptpg_selected_examples.json"
        policy_csv = output_dir / "promptpg_selected_examples.csv"
    status_json = output_dir / "promptpg_status.json"
    promptpg_test_file = args.promptpg_test_file
    if promptpg_test_file is None and source_mode in FORMAL_SOURCE_MODES:
        promptpg_test_file = output_dir / f"{source_mode}_promptpg_test_examples.json"
        write_promptpg_test_file(promptpg_test_file, rows, source_mode)
    cmd = [
        sys.executable,
        "-B",
        str(REPO_ROOT / ".external/FINDER/In_Context_Selection" / "examples_extraction.py"),
        "--ckpt",
        str(ckpt_path),
        "--output-json",
        str(policy_json),
        "--output-csv",
        str(policy_csv),
        "--status-json",
        str(status_json),
        "--test_number",
        str(args.limit if args.limit and args.limit > 0 else -1),
        "--shot_number",
        str(args.shot_number),
    ]
    if promptpg_test_file is not None:
        cmd.extend(["--test-file", str(promptpg_test_file), "--test_split", source_mode])
    if args.promptpg_train_file is not None:
        cmd.extend(["--train-file", str(args.promptpg_train_file)])
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"PromptPG policy extraction failed with exit code {result.returncode}; see {status_json}")
    return policy_json


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


FINQA_DSL_OPERATORS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "greater",
    "exp",
    "table_average",
    "table_max",
    "table_min",
    "table_sum",
}


def split_top_level_expressions(program: str) -> list[str]:
    expressions: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(program):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            item = program[start:index].strip()
            if item:
                expressions.append(item)
            start = index + 1
    tail = program[start:].strip()
    if tail:
        expressions.append(tail)
    return expressions


def const_token_to_python(token: str) -> str:
    value = token.removeprefix("const_")
    if value.startswith("m") and value[1:].isdigit():
        return f"(-{value[1:]})"
    if value.isdigit():
        return value
    raise ValueError(f"Unsupported FinQA constant: {token}")


def normalize_dsl_tokens(text: str) -> str:
    text = re.sub(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*%", r"(\1 / 100)", text)
    text = re.sub(r"#(\d+)", r"ref_\1", text)
    return re.sub(r"\bconst_(?:m?\d+)\b", lambda match: const_token_to_python(match.group(0)), text)


def dsl_ast_to_python(node: ast.AST, original: str) -> str:
    if isinstance(node, ast.Expression):
        return dsl_ast_to_python(node.body, original)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        if re.fullmatch(r"ref_\d+", node.id):
            return f"step_{node.id.split('_', 1)[1]}"
        return node.id
    if isinstance(node, ast.UnaryOp):
        operand = dsl_ast_to_python(node.operand, original)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.UAdd):
            return f"(+{operand})"
    if isinstance(node, ast.BinOp):
        left = dsl_ast_to_python(node.left, original)
        right = dsl_ast_to_python(node.right, original)
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "**",
            ast.Mod: "%",
        }
        for op_type, op_text in op_map.items():
            if isinstance(node.op, op_type):
                return f"({left} {op_text} {right})"
    if isinstance(node, ast.List):
        return "[" + ", ".join(dsl_ast_to_python(item, original) for item in node.elts) + "]"
    if isinstance(node, ast.Tuple):
        return "(" + ", ".join(dsl_ast_to_python(item, original) for item in node.elts) + ")"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = node.func.id
        args = [dsl_ast_to_python(arg, original) for arg in node.args]
        if func == "add" and len(args) == 2:
            return f"({args[0]} + {args[1]})"
        if func == "subtract" and len(args) == 2:
            return f"({args[0]} - {args[1]})"
        if func == "multiply" and len(args) == 2:
            return f"({args[0]} * {args[1]})"
        if func == "divide" and len(args) == 2:
            return f"({args[0]} / {args[1]})"
        if func == "greater" and len(args) == 2:
            return f"({args[0]} > {args[1]})"
        if func == "exp" and len(args) == 2:
            return f"({args[0]} ** {args[1]})"
        if func in {"table_sum", "table_average", "table_max", "table_min"} and args:
            values = args[0] if len(args) == 1 and args[0].startswith("[") else "[" + ", ".join(args) + "]"
            if func == "table_sum":
                return f"sum({values})"
            if func == "table_average":
                return f"(sum({values}) / len({values}))"
            if func == "table_max":
                return f"max({values})"
            if func == "table_min":
                return f"min({values})"
    raise ValueError(f"Unsupported FinQA DSL node: {ast.dump(node, include_attributes=False)}")


def looks_like_finqa_dsl(program: str) -> bool:
    if "#" in program or re.search(r"\bconst_(?:m?\d+)\b", program):
        return True
    return any(re.search(rf"\b{re.escape(operator)}\s*\(", program) for operator in FINQA_DSL_OPERATORS)


def pythonize_finqa_program(program: Any) -> str:
    source = stringify(program).strip()
    if not source:
        return ""
    expressions = split_top_level_expressions(source.replace("\n", " "))
    if not expressions or not looks_like_finqa_dsl(source):
        return source.rstrip()
    lines: list[str] = []
    for index, raw_expression in enumerate(expressions):
        expression = normalize_dsl_tokens(raw_expression)
        parsed = ast.parse(expression, mode="eval")
        python_expression = dsl_ast_to_python(parsed, expression)
        lines.append(f"step_{index} = {python_expression}")
    lines.append(f"ans = step_{len(expressions) - 1}")
    return "\n".join(lines)


def answer_program_for_prompt(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, bool):
        return f"ans = {answer}"
    if isinstance(answer, (int, float)):
        return f"ans = {answer!r}"
    raw_answer = stringify(answer).strip()
    if not raw_answer:
        return ""
    if re.fullmatch(r"-?\d+(?:\.\d+)?", raw_answer):
        return f"ans = {raw_answer}"
    return f"ans = {raw_answer!r}"


def format_example_program_for_prompt(program: Any, answer: Any = None) -> str:
    source = stringify(program).strip()
    if not source:
        return answer_program_for_prompt(answer)
    try:
        return pythonize_finqa_program(source)
    except Exception:
        if looks_like_finqa_dsl(source):
            fallback = answer_program_for_prompt(answer)
            if fallback:
                return fallback
        return source.rstrip()


def token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def normalize_candidate(raw: dict[str, Any], candidate_id: str | int) -> dict[str, Any]:
    rel_fact = raw.get("rel_fact")
    if not rel_fact:
        rel_fact = raw.get("text") or raw.get("retrieved") or ""
    if isinstance(rel_fact, list):
        rel_fact = "\n".join(stringify(item) for item in rel_fact)
    return {
        "id": str(raw.get("id") or raw.get("pid") or candidate_id),
        "question": stringify(raw.get("question")),
        "answer": raw.get("answer"),
        "program": stringify(raw.get("program") or raw.get("solution")),
        "table": stringify(raw.get("table") or raw.get("table_text")),
        "rel_fact": stringify(rel_fact),
    }


def load_candidate_examples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [normalize_candidate(row, key) for key, row in payload.items() if isinstance(row, dict)]
    if isinstance(payload, list):
        return [normalize_candidate(row, index) for index, row in enumerate(payload) if isinstance(row, dict)]
    raise ValueError(f"Unsupported candidate JSON shape: {path}")


def normalize_policy_example(raw: dict[str, Any], index: int) -> dict[str, Any]:
    return normalize_candidate(
        {
            "id": raw.get("Shot_id") or raw.get("shot_id") or raw.get("id") or index,
            "question": raw.get("Shot_Question") or raw.get("shot_question") or raw.get("question"),
            "answer": raw.get("Shot_Answer") or raw.get("shot_answer") or raw.get("answer"),
            "program": raw.get("Shot_program") or raw.get("shot_program") or raw.get("program") or raw.get("solution"),
            "table": raw.get("Shot_Table") or raw.get("shot_table") or raw.get("table") or raw.get("table_text"),
            "rel_fact": raw.get("Shot_Rel_Fact") or raw.get("shot_rel_fact") or raw.get("rel_fact") or raw.get("text"),
        },
        index,
    )


def policy_key_values(raw: dict[str, Any]) -> list[str]:
    question = raw.get("Test_Question") or raw.get("test_question") or raw.get("question")
    source_row = raw.get("Test_source_csv_row") or raw.get("source_csv_row")
    keys = [
        raw.get("Test_selection_key"),
        raw.get("test_selection_key"),
        raw.get("selection_key"),
        raw.get("Test_id"),
        raw.get("test_id"),
        raw.get("target_id"),
        raw.get("id"),
        raw.get("source_id"),
        question,
        normalize_question(stringify(question)) if question is not None else None,
        question_key(question) if question is not None else None,
    ]
    if source_row is not None and question is not None:
        keys.insert(0, f"row:{source_row}|question:{normalize_question(stringify(question))}")
    return [str(key).strip() for key in keys if str(key or "").strip()]


def load_policy_selections(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for index, row in enumerate(rows):
            example = normalize_policy_example(row, index)
            for key in policy_key_values(row):
                grouped.setdefault(key, []).append(example)
        return grouped

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), dict):
            for key, value in payload["items"].items():
                if isinstance(value, dict) and isinstance(value.get("selected_examples"), list):
                    examples = [
                        normalize_policy_example(item, index)
                        for index, item in enumerate(value["selected_examples"])
                        if isinstance(item, dict)
                    ]
                    for item_key in [key, *policy_key_values(value)]:
                        grouped[str(item_key)] = examples
            return grouped
        for key, value in payload.items():
            if isinstance(value, dict) and isinstance(value.get("selected_examples"), list):
                examples = [
                    normalize_policy_example(item, index)
                    for index, item in enumerate(value["selected_examples"])
                    if isinstance(item, dict)
                ]
                for item_key in [key, *policy_key_values(value)]:
                    grouped[str(item_key)] = examples
            elif isinstance(value, list):
                grouped[str(key)] = [
                    normalize_policy_example(item, index) for index, item in enumerate(value) if isinstance(item, dict)
                ]
    elif isinstance(payload, list):
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            selected = row.get("selected_examples")
            if isinstance(selected, list):
                examples = [
                    normalize_policy_example(item, item_index)
                    for item_index, item in enumerate(selected)
                    if isinstance(item, dict)
                ]
            else:
                examples = [normalize_policy_example(row, index)]
            for key in policy_key_values(row):
                grouped.setdefault(key, []).extend(examples)
    else:
        raise ValueError(f"Unsupported policy output shape: {path}")
    return grouped


def row_policy_keys(row: dict[str, Any]) -> list[str]:
    question = row.get("question")
    keys = [
        row.get("selection_key"),
        selection_key_for_row(row),
        row.get("id"),
        row.get("source_id"),
        question,
        normalize_question(stringify(question)) if question is not None else None,
        question_key(question) if question is not None else None,
    ]
    return [str(key).strip() for key in keys if str(key or "").strip()]


def row_stable_id(row: dict[str, Any]) -> str:
    for key in ("id", "source_id", "source_csv_row"):
        value = stringify(row.get(key)).strip()
        if value:
            return value
    return selection_key_for_row(row)


def prompt_type_train_csv(prompt_type: str | None, override: Path | None) -> Path | None:
    if override is not None:
        return override
    if prompt_type is None:
        return None
    return PROMPT_TYPE_TRAIN_CSV.get(prompt_type)


def build_example_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        keys = {
            row_stable_id(row),
            stringify(row.get("source_csv_row")).strip(),
            selection_key_for_row(row),
            normalize_question(stringify(row.get("question"))),
            question_key(row.get("question")),
        }
        for key in keys:
            if key:
                lookup.setdefault(key, row)
    return lookup


def resolve_selected_id(example: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> str | None:
    keys = [
        stringify(example.get("id")).strip(),
        stringify(example.get("source_id")).strip(),
        stringify(example.get("source_csv_row")).strip(),
        normalize_question(stringify(example.get("question"))),
        question_key(example.get("question")),
    ]
    for key in keys:
        if key and key in lookup:
            return row_stable_id(lookup[key])
    return None


def cache_selected_example_payload(example: dict[str, Any], rank: int) -> dict[str, Any]:
    item = normalize_policy_example(example, rank - 1)
    for key in (
        "source_id",
        "source_csv_row",
        "source_csv_position",
        "selection_key",
        "table_text",
        "retrieved",
        "retrieved_source",
        "flow_scope",
    ):
        if key in example and example.get(key) not in (None, ""):
            item[key] = example.get(key)
    item["selection_rank"] = rank
    item["selection_source"] = stringify(example.get("selection_source") or "experiment7_selection_cache_materialized")
    return item


def load_selection_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        raise SystemExit(f"blocked_example_selection_cache: missing selection cache {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "experiment7_selection_cache":
        raise SystemExit(f"blocked_example_selection_cache: unsupported cache format {path}")
    return payload


def cache_lookup_entry(item: dict[str, Any], key: str, key_type: str) -> dict[str, Any]:
    entry = dict(item)
    entry["_cache_lookup_key"] = key
    entry["_cache_lookup_key_type"] = key_type
    return entry


def cache_items_by_key(
    cache_payload: dict[str, Any],
    allow_legacy_selection_binding: bool = False,
) -> dict[str, dict[str, Any]]:
    raw_items = cache_payload.get("items")
    if not isinstance(raw_items, dict):
        raise SystemExit("blocked_example_selection_cache: cache missing items object")
    lookup: dict[str, dict[str, Any]] = {}

    def add_key(item: dict[str, Any], key: Any, key_type: str) -> None:
        value = stringify(key).strip()
        if value:
            lookup.setdefault(value, cache_lookup_entry(item, value, key_type))

    for key, item in raw_items.items():
        if not isinstance(item, dict):
            continue
        normalized_question = stringify(item.get("normalized_question")).strip()
        raw_item_key = stringify(key).strip()
        item_selection_key = stringify(item.get("selection_key")).strip()
        if raw_item_key and (raw_item_key.startswith("row:") or raw_item_key.startswith("question:")):
            add_key(item, raw_item_key, "selection_key")
        elif allow_legacy_selection_binding:
            add_key(item, raw_item_key, "legacy_cache_item_key")
        if item_selection_key and (item_selection_key.startswith("row:") or item_selection_key.startswith("question:")):
            add_key(item, item_selection_key, "selection_key")
        elif allow_legacy_selection_binding:
            add_key(item, item_selection_key, "legacy_selection_key")
        if normalized_question:
            add_key(item, question_key(normalized_question), "question_key")
        if allow_legacy_selection_binding:
            add_key(item, item.get("stable_id"), "legacy_stable_id")
            add_key(item, item.get("source_csv_row"), "legacy_source_csv_row")
            add_key(item, normalized_question, "legacy_normalized_question")
            if item.get("question"):
                add_key(item, item.get("question"), "legacy_raw_question")
                add_key(item, question_key(item.get("question")), "question_key")
    return lookup


def row_cache_key_candidates(row: dict[str, Any], allow_legacy_selection_binding: bool) -> list[tuple[str, str]]:
    question = row.get("question")
    keys: list[tuple[Any, str]] = [
        (row.get("selection_key"), "selection_key"),
        (selection_key_for_row(row), "selection_key"),
    ]
    if question is not None:
        keys.append((question_key(question), "question_key"))
    if allow_legacy_selection_binding:
        keys.extend(
            [
                (row.get("id"), "legacy_row_id"),
                (row.get("source_id"), "legacy_source_id"),
                (question, "legacy_raw_question"),
                (normalize_question(stringify(question)) if question is not None else None, "legacy_normalized_question"),
            ]
        )
    output: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, key_type in keys:
        value = stringify(key).strip()
        if value and value not in seen:
            output.append((value, key_type))
            seen.add(value)
    return output


def selected_from_cache(
    row: dict[str, Any],
    cache_map: dict[str, dict[str, Any]],
    example_lookup: dict[str, dict[str, Any]],
    shot_number: int,
    allow_legacy_selection_binding: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    cache_item = None
    for key, _ in row_cache_key_candidates(row, allow_legacy_selection_binding):
        cache_item = cache_map.get(key)
        if cache_item:
            break
    if cache_item is None:
        return [], None, ["target_row_not_in_selection_cache"]
    materialized_examples = cache_item.get("selected_examples")
    if isinstance(materialized_examples, list) and materialized_examples:
        resolved = []
        for rank, example in enumerate(materialized_examples[:shot_number], start=1):
            if not isinstance(example, dict):
                continue
            item = cache_selected_example_payload(example, rank)
            item["selection_source"] = "experiment7_selection_cache_materialized"
            resolved.append(item)
        if resolved:
            return resolved, cache_item, []

    selected_ids = cache_item.get("selected_example_ids")
    if not isinstance(selected_ids, list):
        # Backward-compatible read for older smoke caches; not used for formal handoff.
        selected_examples = cache_item.get("selected_examples")
        if isinstance(selected_examples, list):
            selected_ids = [resolve_selected_id(item, example_lookup) for item in selected_examples if isinstance(item, dict)]
        else:
            selected_ids = []
    if not selected_ids:
        for fallback_key in ("selected_example_source_rows", "selected_example_policy_ids"):
            fallback_ids = cache_item.get(fallback_key)
            if isinstance(fallback_ids, list) and fallback_ids:
                selected_ids = fallback_ids
                break
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    for rank, selected_id in enumerate([str(item) for item in selected_ids if item is not None][:shot_number], start=1):
        example = example_lookup.get(selected_id)
        if example is None:
            missing.append(selected_id)
            continue
        item = dict(example)
        item["selection_rank"] = rank
        item["selection_source"] = "experiment7_selection_cache"
        resolved.append(item)
    return resolved, cache_item, missing


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"materialized selection row {line_no} is not an object: {path}")
            rows.append(item)
    return rows


def selected_from_materialized_row(row: dict[str, Any], materialized_row: dict[str, Any], shot_number: int) -> tuple[list[dict[str, Any]], list[str]]:
    target_selection_key = selection_key_for_row(row)
    materialized_selection_key = stringify(materialized_row.get("selection_key")).strip()
    if not materialized_selection_key:
        return [], ["materialized_selection_key_missing"]
    if materialized_selection_key != target_selection_key:
        return [], ["materialized_selection_key_mismatch"]
    materialized_meta = materialized_row.get("example_selection")
    if not isinstance(materialized_meta, dict) or materialized_meta.get("selection_binding_status") != "passed":
        return [], ["materialized_selection_binding_not_passed"]
    if normalize_question(stringify(row.get("question"))) != normalize_question(stringify(materialized_row.get("question"))):
        return [], ["materialized_question_order_mismatch"]
    selected_raw = materialized_row.get("selected_examples")
    if not isinstance(selected_raw, list) or not selected_raw:
        return [], ["materialized_selected_examples_missing"]
    selected: list[dict[str, Any]] = []
    for rank, example in enumerate(selected_raw[:shot_number], start=1):
        if not isinstance(example, dict):
            continue
        item = dict(example)
        item["selection_rank"] = rank
        item["selection_source"] = "experiment7_materialized_selection_cache"
        selected.append(item)
    if not selected:
        return [], ["materialized_selected_examples_empty"]
    return selected, []


def attach_materialized_selection(
    rows: list[dict[str, Any]],
    materialized_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    source_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(materialized_rows) < len(rows):
        raise SystemExit(
            "blocked_example_selection_cache: materialized selection rows are fewer than target rows; "
            f"materialized_rows={len(materialized_rows)} rows={len(rows)}"
        )
    output: list[dict[str, Any]] = []
    cache_missing_ids: dict[str, list[str]] = {}
    formal_rows = 0
    cache_rows = 0
    missing_rows = 0
    identity_backfill_rows = 0
    for index, row in enumerate(rows):
        materialized_row = materialized_rows[index]
        row = dict(row)
        if row.get("source_csv_row") is None and not stringify(row.get("selection_key")).strip():
            materialized_source_row = materialized_row.get("source_csv_row")
            materialized_question = stringify(materialized_row.get("question"))
            materialized_key = stringify(materialized_row.get("selection_key")).strip()
            if materialized_source_row is not None and materialized_question.strip():
                expected_materialized_key = (
                    f"row:{materialized_source_row}|question:{normalize_question(materialized_question)}"
                )
                if (
                    normalize_question(stringify(row.get("question")))
                    == normalize_question(materialized_question)
                    and materialized_key == expected_materialized_key
                ):
                    row["source_csv_row"] = materialized_source_row
                    row["selection_key"] = materialized_key
                    identity_backfill_rows += 1
        selected, missing = selected_from_materialized_row(row, materialized_row, args.shot_number)
        selection_key = selection_key_for_row(row)
        if selected and not missing:
            cache_rows += 1
            formal_finder_ready = args.formal_finder_ready and source_mode in FORMAL_SOURCE_MODES
            if formal_finder_ready:
                formal_rows += 1
            selection_status = "materialized_selection_cache"
        else:
            missing_rows += 1
            formal_finder_ready = False
            selection_status = "blocked_example_selection_cache"
            cache_missing_ids[selection_key] = missing or ["missing_materialized_selection"]
        item = dict(row)
        item["selected_examples"] = selected
        if selected:
            item["prompt"] = build_in_context_prompt(selected)
            item["prompt_scope"] = "in_context_examples_only"
            item["target_prompt_scope"] = "runner_appends_target"
        item["selection_key"] = selection_key
        item["example_selection"] = {
            "selection_key": selection_key,
            "stable_id": row_stable_id(item),
            "source_csv_row": item.get("source_csv_row"),
            "normalized_question": normalize_question(stringify(item.get("question"))),
            "selection_mode": args.selection_mode,
            "selection_status": selection_status,
            "selection_binding_status": "passed" if formal_finder_ready else selection_status,
            "selection_source": "experiment7_materialized_selection_cache",
            "selection_cache_json": str(args.selection_cache_json) if args.selection_cache_json else None,
            "materialized_selection_jsonl": str(args.materialized_selection_jsonl),
            "missing_cache_ids": missing,
            "formal_finder_ready": formal_finder_ready,
            "cache_key_found": bool(selected and not missing),
            "cache_key_found_value": selection_key if selected and not missing else None,
            "cache_match_key_type": "selection_key" if selected and not missing else None,
            "allow_legacy_selection_binding": False,
            "flow_scope": "formal_retriever_conditioned_generator" if formal_finder_ready else item.get("flow_scope"),
            "note": "Selected examples were pre-materialized from the formal selection cache and aligned by target question order.",
        }
        output.append(item)
    summary = {
        "rows": len(rows),
        "policy_rows": 0,
        "cache_rows": cache_rows,
        "similarity_fallback_rows": 0,
        "missing_selection_rows": missing_rows,
        "formal_finder_ready_rows": formal_rows,
        "cache_missing_ids": cache_missing_ids,
        "cache_match_key_types": {"selection_key": cache_rows} if cache_rows else {},
        "legacy_cache_alias_rows": 0,
        "row_number_collision_rows": 0,
        "identity_backfill_rows": identity_backfill_rows,
        "materialized_selection_jsonl": str(args.materialized_selection_jsonl),
    }
    return output, summary


def score_candidate(row: dict[str, Any], candidate: dict[str, Any]) -> float:
    target_tokens = token_set(" ".join([stringify(row.get("question")), stringify(row.get("text"))]))
    candidate_tokens = token_set(" ".join([candidate["question"], candidate["rel_fact"]]))
    if not target_tokens or not candidate_tokens:
        return 0.0
    overlap = len(target_tokens & candidate_tokens)
    return overlap / math.sqrt(len(target_tokens) * len(candidate_tokens))


def select_by_similarity(row: dict[str, Any], candidates: list[dict[str, Any]], shot_number: int) -> list[dict[str, Any]]:
    target_question = stringify(row.get("question")).strip().lower()
    scored = []
    for candidate in candidates:
        if candidate["question"].strip().lower() == target_question:
            continue
        scored.append((score_candidate(row, candidate), candidate["id"], candidate))
    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:shot_number]
    ordered = list(reversed(top))
    selected = []
    for prompt_position, (score, _, candidate) in enumerate(ordered, start=1):
        item = dict(candidate)
        item["selection_score"] = score
        item["selection_rank"] = prompt_position
        item["selection_source"] = "finqa_train_similarity_fallback"
        selected.append(item)
    return selected


def selected_from_policy(
    row: dict[str, Any], policy_map: dict[str, list[dict[str, Any]]], shot_number: int
) -> list[dict[str, Any]]:
    for key in row_policy_keys(row):
        examples = policy_map.get(key)
        if examples:
            selected = []
            for index, example in enumerate(examples[:shot_number], start=1):
                item = dict(example)
                item["selection_rank"] = index
                item["selection_source"] = "promptpg_policy_output"
                selected.append(item)
            return selected
    return []


def format_facts(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(stringify(item).strip() for item in value if stringify(item).strip())
    return stringify(value).strip()


def format_example_prompt(example: dict[str, Any], include_code: bool) -> str:
    table = stringify(example.get("table") or example.get("table_text")).strip()
    retrieved = format_facts(example.get("rel_fact") or example.get("retrieved") or example.get("text"))
    question = stringify(example.get("question")).strip()
    intro = (
        "Read the following table and probable relevant facts, and then the python code below "
        "that answers the question, the answer can be a float/int or bool:\n"
        if include_code
        else "Read the following table and probable relevant facts, and then write code to answer "
        "a question, the answer can be a float/int or bool:\n"
    )
    prompt = (
        intro
        + f"Table: \n{table}\n"
        "Probable relevant facts:\n "
        f"{retrieved}\n\n"
        f"Question: {question}\n"
        "#Python code below\n"
    )
    if include_code:
        program = format_example_program_for_prompt(example.get("program"), example.get("answer"))
        if program:
            prompt += program.rstrip() + "\n"
    return prompt.rstrip()


def build_in_context_prompt(selected_examples: list[dict[str, Any]]) -> str:
    parts = [format_example_prompt(example, include_code=True) for example in selected_examples]
    return "\n\n".join(part for part in parts if part)


def build_full_prompt(row: dict[str, Any], selected_examples: list[dict[str, Any]]) -> str:
    parts = [build_in_context_prompt(selected_examples), format_example_prompt(row, include_code=False)]
    return "\n\n".join(part for part in parts if part)


def attach_selection(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    policy_map: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    source_mode: str,
    cache_map: dict[str, dict[str, Any]] | None = None,
    example_lookup: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.shot_number < 0:
        raise ValueError("--shot-number must be non-negative.")

    cache_map = cache_map or {}
    example_lookup = example_lookup or {}
    output = []
    policy_rows = 0
    cache_rows = 0
    formal_rows = 0
    fallback_rows = 0
    missing_rows = 0
    cache_missing_ids: dict[str, list[str]] = {}
    cache_match_key_types: dict[str, int] = {}
    legacy_cache_alias_rows = 0
    for row in rows:
        selected: list[dict[str, Any]] = []
        selection_status = "none"
        formal_finder_ready = False
        cache_item = None
        cache_lookup_key = None
        cache_lookup_key_type = None
        missing_cache_ids: list[str] = []
        if args.selection_mode == "cache":
            selected, cache_item, missing_cache_ids = selected_from_cache(
                row,
                cache_map,
                example_lookup,
                args.shot_number,
                allow_legacy_selection_binding=args.allow_legacy_selection_binding,
            )
            if isinstance(cache_item, dict):
                cache_lookup_key = cache_item.get("_cache_lookup_key")
                cache_lookup_key_type = cache_item.get("_cache_lookup_key_type")
                if cache_lookup_key_type:
                    cache_match_key_types[str(cache_lookup_key_type)] = cache_match_key_types.get(str(cache_lookup_key_type), 0) + 1
            is_legacy_cache_alias = bool(cache_lookup_key_type and str(cache_lookup_key_type).startswith("legacy_"))
            if selected and not missing_cache_ids:
                cache_rows += 1
                if is_legacy_cache_alias:
                    legacy_cache_alias_rows += 1
                    selection_status = "blocked_legacy_row_alias_collision" if args.formal_finder_ready else "legacy_selection_cache_alias"
                    formal_finder_ready = False
                else:
                    selection_status = "selection_cache"
                    formal_finder_ready = args.formal_finder_ready and source_mode in FORMAL_SOURCE_MODES
                    if formal_finder_ready:
                        formal_rows += 1
            else:
                selection_status = "blocked_example_selection_cache"
                cache_missing_ids[selection_key_for_row(row)] = missing_cache_ids or ["missing_cache_item"]
        else:
            if args.selection_mode in {"auto", "policy"}:
                selected = selected_from_policy(row, policy_map, args.shot_number)
                if selected:
                    selection_status = "promptpg_policy"
                    formal_finder_ready = source_mode in FORMAL_SOURCE_MODES
                    policy_rows += 1
                    if formal_finder_ready:
                        formal_rows += 1
            if not selected and args.selection_mode in {"auto", "similarity"} and args.shot_number > 0:
                selected = select_by_similarity(row, candidates, args.shot_number)
                if selected:
                    selection_status = "similarity_fallback_no_promptpg_checkpoint"
                    fallback_rows += 1
        if not selected:
            missing_rows += 1

        item = dict(row)
        item["selected_examples"] = selected
        if selected:
            item["prompt"] = build_in_context_prompt(selected)
            item["prompt_scope"] = "in_context_examples_only"
            item["target_prompt_scope"] = "runner_appends_target"
        item["selection_key"] = selection_key_for_row(item)
        selection_binding_status = "passed" if formal_finder_ready else selection_status
        item["example_selection"] = {
            "selection_key": item["selection_key"],
            "stable_id": row_stable_id(item),
            "source_csv_row": item.get("source_csv_row"),
            "normalized_question": normalize_question(stringify(item.get("question"))),
            "selection_mode": args.selection_mode,
            "selection_status": selection_status,
            "selection_binding_status": selection_binding_status,
            "formal_finder_ready": formal_finder_ready,
            "shot_number_requested": args.shot_number,
            "shot_number_selected": len(selected),
            "candidate_json": str(args.candidate_json),
            "policy_output": str(args.policy_output) if args.policy_output else None,
            "selection_cache_json": str(args.selection_cache_json) if args.selection_cache_json else None,
            "target_prompt_type": args.target_prompt_type,
            "prompt_type_train_csv": str(args.prompt_type_train_csv) if args.prompt_type_train_csv else None,
            "cache_item_selection_key": cache_item.get("selection_key") if isinstance(cache_item, dict) else None,
            "cache_key_found": cache_lookup_key is not None,
            "cache_key_found_value": cache_lookup_key,
            "cache_match_key_type": cache_lookup_key_type,
            "allow_legacy_selection_binding": args.allow_legacy_selection_binding,
            "missing_cache_ids": missing_cache_ids,
            "source_mode": source_mode,
            "notes": (
                "Selection cache resolved selected ids and prompt-type train CSV examples."
                if selection_status == "selection_cache"
                else "PromptPG/policy selected examples are present."
                if formal_finder_ready
                else "This route is not a formal PromptPG reproduction unless selection_status is promptpg_policy or selection_cache."
            ),
        }
        if not formal_finder_ready:
            item["finder_alignment"] = "smoke_or_feasibility_fallback"
        output.append(item)

    summary = {
        "rows": len(output),
        "policy_rows": policy_rows,
        "cache_rows": cache_rows,
        "similarity_fallback_rows": fallback_rows,
        "missing_selection_rows": missing_rows,
        "formal_finder_ready_rows": formal_rows,
        "cache_missing_ids": cache_missing_ids,
        "cache_match_key_types": cache_match_key_types,
        "legacy_cache_alias_rows": legacy_cache_alias_rows,
        "row_number_collision_rows": legacy_cache_alias_rows,
    }
    return output, summary


def default_selection_cache_json(output_json: Path) -> Path:
    return output_json.with_name("selection_cache.json")


def formal_target_gold_missing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing_rows = []
    for index, row in enumerate(rows):
        missing = [
            key
            for key in ("id", "answer", "program")
            if row.get(key) is None or (isinstance(row.get(key), str) and not row.get(key).strip())
        ]
        if missing:
            missing_rows.append(
                {
                    "row_index": index,
                    "question": stringify(row.get("question"))[:160],
                    "missing": missing,
                    "source": row.get("source"),
                }
            )
            if len(missing_rows) >= 5:
                break
    return missing_rows


def require_formal_target_gold(rows: list[dict[str, Any]], source_mode: str, source_path: Path) -> None:
    if source_mode not in FORMAL_SOURCE_MODES:
        return
    missing = formal_target_gold_missing(rows)
    if not missing:
        return
    raise SystemExit(
        "blocked_formal_target_gold_missing: formal Experiment 7 target rows require id, answer, and program "
        f"before prompt construction; source={source_path} missing_examples={json.dumps(missing, ensure_ascii=False)}"
    )


def write_selection_cache(
    path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    source_path: Path,
    source_mode: str,
    selection_summary: dict[str, Any],
) -> None:
    questions = [stringify(row.get("question")).strip() for row in rows]
    normalized_questions = [normalize_question(question) for question in questions]
    duplicate_questions = sorted(
        question for question in set(normalized_questions) if normalized_questions.count(question) > 1
    )
    source_lookup: dict[str, dict[str, Any]] = {}
    if source_path.suffix.lower() == ".csv" and source_mode in FORMAL_CSV_SOURCE_MODES:
        source_lookup = build_example_lookup(load_csv_examples(source_path, -1, source_mode))

    items = {}
    unresolved_selected_examples: dict[str, list[str]] = {}
    for row in rows:
        key = selection_key_for_row(row)
        selected = row.get("selected_examples") if isinstance(row.get("selected_examples"), list) else []
        selected_ids: list[str] = []
        selected_policy_ids: list[str] = []
        selected_source_rows: list[Any] = []
        unresolved: list[str] = []
        for example in selected:
            if not isinstance(example, dict):
                continue
            policy_id = stringify(example.get("id")).strip()
            if policy_id:
                selected_policy_ids.append(policy_id)
            resolved_id = resolve_selected_id(example, source_lookup) if source_lookup else policy_id
            if resolved_id:
                selected_ids.append(resolved_id)
                source_row = source_lookup.get(resolved_id, {}).get("source_csv_row") if source_lookup else None
                if source_row is not None:
                    selected_source_rows.append(source_row)
            else:
                unresolved.append(policy_id or normalize_question(stringify(example.get("question"))))
        if unresolved:
            unresolved_selected_examples[key] = unresolved
        item_meta = row.get("example_selection") if isinstance(row.get("example_selection"), dict) else {}
        materialized_selected_examples = [
            cache_selected_example_payload(example, rank)
            for rank, example in enumerate(selected, start=1)
            if isinstance(example, dict)
        ]
        items[key] = {
            "selection_key": key,
            "stable_id": row_stable_id(row),
            "source_csv_row": row.get("source_csv_row"),
            "normalized_question": normalize_question(stringify(row.get("question"))),
            "selected_example_ids": selected_ids,
            "selected_examples": materialized_selected_examples,
            "selected_example_source_rows": selected_source_rows,
            "selected_example_policy_ids": selected_policy_ids,
            "selection_status": item_meta.get("selection_status"),
            "formal_finder_ready": item_meta.get("formal_finder_ready", False),
            "shot_number_selected": len(materialized_selected_examples) or len(selected_ids),
            "question": stringify(row.get("question")),
        }
    payload = {
        "kind": "experiment7_selection_cache",
        "contract_version": 3,
        "artifact_contract": "stable target keys plus materialized selected examples; legacy selected ids remain lookup-compatible",
        "source_mode": source_mode,
        "source": str(source_path),
        "model_route": os.environ.get("EXAMPLE_SELECTION_ENGINE") or os.environ.get("ENGINE"),
        "shot_number": args.shot_number,
        "rows": len(rows),
        "unique_normalized_questions": len(set(normalized_questions)),
        "duplicate_normalized_questions": duplicate_questions,
        "question_sha256": hashlib.sha256("\n".join(questions).encode("utf-8")).hexdigest(),
        "key_strategy": "stable source row + normalized question; question-only keys remain lookup-compatible",
        "policy_output": str(args.policy_output) if args.policy_output else None,
        "selection_summary": selection_summary,
        "unresolved_selected_examples": unresolved_selected_examples,
        "items": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_json = args.output_json or default_output()
    if args.input_json is None and args.input_csv is None:
        args.input_json = default_input_json()
        if args.input_json is None:
            raise SystemExit(
                "example_selection.py requires --input-json for formal matched retriever artifacts "
                "or --input-csv with --formal-csv-source for formal CSV routes. "
                "The default finqa_flan_o:finqa_test matched artifact is missing."
            )

    if args.input_json is not None:
        examples = load_matched_examples(args.input_json, args.limit)
        source_mode = "matched_retriever_artifact"
        source_path = args.input_json
    else:
        if args.formal_csv_source:
            examples = load_csv_examples(args.input_csv, args.limit, args.formal_csv_source)
            source_mode = args.formal_csv_source
        elif not args.allow_relfact_smoke:
            raise SystemExit("--input-csv is smoke-only; pass --formal-csv-source for formal routes or --allow-relfact-smoke for isolated smoke.")
        else:
            examples = load_smoke_examples(args.input_csv, args.limit)
            source_mode = "csv_rel_fact_smoke_only"
        source_path = args.input_csv

    if args.formal_finder_ready:
        require_formal_target_gold(examples, source_mode, source_path)

    policy_map: dict[str, list[dict[str, Any]]] = {}
    candidates: list[dict[str, Any]] = []
    cache_map: dict[str, dict[str, Any]] = {}
    example_lookup: dict[str, dict[str, Any]] = {}

    if args.selection_mode == "cache":
        if args.selection_cache_json is None:
            raise SystemExit("blocked_example_selection_cache: --selection-cache-json is required in cache mode")
        cache_payload = load_selection_cache(args.selection_cache_json)
        cache_source_mode = stringify(cache_payload.get("source_mode")).strip()
        cache_source = stringify(cache_payload.get("source")).strip()
        if args.formal_finder_ready and not args.allow_legacy_selection_binding:
            if cache_source_mode != source_mode:
                raise SystemExit(
                    "blocked_example_selection_cache: selection cache source_mode mismatch; "
                    f"cache_source_mode={cache_source_mode or 'missing'} target_source_mode={source_mode}. "
                    "Generate a target-specific selection cache from the matched retriever artifact first."
                )
            if source_mode == "matched_retriever_artifact" and cache_source:
                cache_source_path = Path(cache_source).expanduser()
                target_source_path = source_path.expanduser()
                if cache_source_path.resolve() != target_source_path.resolve():
                    raise SystemExit(
                        "blocked_example_selection_cache: selection cache source path mismatch; "
                        f"cache_source={cache_source_path} target_source={target_source_path}. "
                        "Generate a target-specific selection cache from this retriever matched JSON."
                    )
        cache_map = cache_items_by_key(
            cache_payload,
            allow_legacy_selection_binding=args.allow_legacy_selection_binding,
        )
        train_csv = prompt_type_train_csv(args.target_prompt_type, args.prompt_type_train_csv)
        if train_csv is None:
            raise SystemExit("blocked_example_selection_cache: --target-prompt-type or --prompt-type-train-csv is required in cache mode")
        if not train_csv.exists():
            raise SystemExit(f"blocked_example_selection_cache: prompt-type train CSV missing {train_csv}")
        args.prompt_type_train_csv = train_csv
        example_lookup = build_example_lookup(load_csv_examples(train_csv, -1, "finqa_train_formal"))
    else:
        generated_policy_output = maybe_build_promptpg_policy(args, examples, source_mode)
        if generated_policy_output is not None:
            args.policy_output = generated_policy_output

        policy_map = load_policy_selections(args.policy_output)
        if args.require_policy and not policy_map:
            raise SystemExit(
                "EXAMPLE_SELECTION_REQUIRE_POLICY=1 but no usable PromptPG/policy output was provided. "
                "Set EXAMPLE_SELECTION_POLICY_OUTPUT or disable the requirement for similarity fallback."
            )
        candidates = [] if args.selection_mode == "policy" else load_candidate_examples(args.candidate_json)

    if args.selection_mode == "cache" and args.materialized_selection_jsonl is not None:
        if not args.materialized_selection_jsonl.exists():
            raise SystemExit(f"blocked_example_selection_cache: materialized selection jsonl missing {args.materialized_selection_jsonl}")
        materialized_rows = load_jsonl_rows(args.materialized_selection_jsonl)
        selected_examples, selection_summary = attach_materialized_selection(
            examples, materialized_rows, args, source_mode
        )
    else:
        selected_examples, selection_summary = attach_selection(
            examples, candidates, policy_map, args, source_mode, cache_map=cache_map, example_lookup=example_lookup
        )
    if args.selection_mode == "cache" and args.require_cache and selection_summary["formal_finder_ready_rows"] != selection_summary["rows"]:
        raise SystemExit(
            "blocked_example_selection_cache: cache missing or selected ids could not be backfilled for at least one row; "
            f"cache_rows={selection_summary['cache_rows']} rows={selection_summary['rows']} missing={selection_summary['cache_missing_ids']}"
        )
    if args.require_policy and selection_summary["formal_finder_ready_rows"] != selection_summary["rows"]:
        raise SystemExit(
            "PromptPG/policy selected examples are missing for at least one formal target row; "
            f"source_mode={source_mode} formal_rows={selection_summary['formal_finder_ready_rows']} rows={selection_summary['rows']}."
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(selected_examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selection_cache_json = args.selection_cache_json or default_selection_cache_json(output_json)
    if args.selection_mode != "cache":
        write_selection_cache(selection_cache_json, selected_examples, args, source_path, source_mode, selection_summary)
    print(
        json.dumps(
            {
                "rows": len(selected_examples),
                "output_json": str(output_json),
                "source": str(source_path),
                "source_mode": source_mode,
                "selection_cache_json": str(selection_cache_json),
                "selection": selection_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
