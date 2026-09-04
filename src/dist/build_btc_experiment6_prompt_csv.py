#!/usr/bin/env python3
"""Build a BTC custom prompt CSV for Experiment 6 retriever-to-binding generation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

import build_btc_finflier_demo as demo


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_SOURCE = WORKSPACE_ROOT / "data" / "src" / "BTC-USD_2024-12-24.xlsx"
DEFAULT_OUTPUT = REPO_ROOT / "Experiment" / "btc_finflier_custom" / "btc_20241224_rel_fact_instruction.csv"


def clean_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = pd.read_excel(args.source, sheet_name="Sheet1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Unnamed: 0",
        "Sentence",
        "input",
        "Question",
        "GT_Answer",
        "GT_Program",
        "Pre_Text",
        "Post_Text",
        "Tables",
        "Table_Text",
        "Rel_Fact",
        "Source",
        "Narrative_Data",
        "Narrative_Text",
        "Binding_Result",
        "Binding_Reason",
        "Prompt_Mode",
        "Generator_Model",
        "data",
        "text",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in frame.iterrows():
            data = str(row.get("data") or "")
            text = str(row.get("text") or "")
            result = str(row.get("result") or "")
            reason = str(row.get("reason") or "")
            source = demo.stable_case_id(row.get("file"), index + 2) + f"_{index + 1}"
            writer.writerow(
                {
                    "Unnamed: 0": index,
                    "Sentence": text,
                    "input": demo.build_prompt(data, text),
                    "Question": "What data-text bindings are described in the narrative?",
                    "GT_Answer": "",
                    "GT_Program": "",
                    "Pre_Text": text,
                    "Post_Text": "",
                    "Tables": data,
                    "Table_Text": data,
                    "Rel_Fact": result,
                    "Source": source,
                    "Narrative_Data": data,
                    "Narrative_Text": text,
                    "Binding_Result": result,
                    "Binding_Reason": reason,
                    "Prompt_Mode": "btc_20241224",
                    "Generator_Model": "gold_from_BTC-USD_2024-12-24_xlsx_for_eval_only",
                    "data": data,
                    "text": text,
                }
            )
    print(args.output)


if __name__ == "__main__":
    main()
