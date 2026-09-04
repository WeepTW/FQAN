"""Path-safe result organization for FINDER-style retriever outputs.

The canonical retrieved artifact follows the FQAN retriever integration: predicted
RetFact fragments are matched back to the same-row context/table sentences,
then question top-3 context/table matches are appended. CSV ``Rel_Fact`` is
kept only as an evaluation/diagnostic label.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retriever_json_schema import assembler_schema_prediction, normalize_prompt_mode, parse_retfact_schema, schema_required


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DATA_JSON = first_existing_path(
    REPO_ROOT / "Data_Target_Module" / "Finqa" / "finqa_test_with_table_text.json",
    WORKSPACE_ROOT / "src" / "code" / "Data" / "Data_Target_Module" / "Finqa" / "finqa_test_with_table_text.json",
    WORKSPACE_ROOT / "src" / "code" / "Data-FINDER" / "Data_Target_Module" / "Finqa" / "finqa_test_with_table_text.json",
)
DEFAULT_RELFACT_CSVS = {
    "raw": WORKSPACE_ROOT / "data" / "src" / "FINDER" / "finqa_test_rel_fact_instruction.csv",
    "original": WORKSPACE_ROOT / "data" / "finqa_original" / "finqa_test_rel_fact_instruction.csv",
    "zero-shot": WORKSPACE_ROOT / "data" / "finqa_zero_shot" / "finqa_test_rel_fact_instruction.csv",
    "many-shot": WORKSPACE_ROOT / "data" / "finqa_many_shot" / "finqa_test_rel_fact_instruction.csv",
    "dynamic-shot": WORKSPACE_ROOT / "data" / "finqa_dynamic_shot" / "finqa_test_rel_fact_instruction.csv",
}
PRIMARY_CONTEXT_MATCH_TYPES = frozenset({"prediction_fragment"})
AUXILIARY_RETFACT_MATCH_TYPES = frozenset({"retfact_vs_rel_fact_label"})

RAW_PREDICTION_PATHS = {
    "flan_t5_large": REPO_ROOT
    / "Data_Target_Module"
    / "lora_flan_retriever"
    / "lora_flan_large_corrected_prediction_finqa_rel_fact.txt",
    "mistral_v0_3": REPO_ROOT
    / "Data_Target_Module"
    / "mistral_retriever"
    / "mistral_finqa_rel_fact_file.txt",
    "apollo": REPO_ROOT / "Data_Target_Module" / "Apollo" / "finqa_apollo_rel_fact_file.txt",
}

MATCHED_OUTPUT_PATHS = {
    "flan_t5_large": REPO_ROOT
    / "Data_Target_Module"
    / "lora_flan_retriever"
    / "output"
    / "best_matched_with_retrieved_facts_and_questions.json",
    "mistral_v0_3": REPO_ROOT
    / "Data_Target_Module"
    / "mistral_retriever"
    / "output"
    / "best_matched_with_retrieved_facts_and_questions_mistral.json",
    "t5gemma_2_1b_1b": REPO_ROOT
    / "Data_Target_Module"
    / "t5gemma_retriever"
    / "output"
    / "best_matched_with_retrieved_facts_and_questions_t5gemma.json",
    "apollo": REPO_ROOT
    / "Data_Target_Module"
    / "Apollo"
    / "output"
    / "best_matched_with_retrieved_facts_and_questions_apollo.json",
}


@dataclass(frozen=True)
class MatchPlan:
    dataset: str
    retriever_model: str
    prompt_mode: str
    input_txt: Path
    data_json: Path
    relfact_csv: Path
    output_json: Path
    embedding_batch_size: int = 256

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "retriever_model": self.retriever_model,
            "prompt_mode": self.prompt_mode,
            "input_txt": str(self.input_txt),
            "data_json": str(self.data_json),
            "relfact_csv": str(self.relfact_csv),
            "output_json": str(self.output_json),
            "embedding_batch_size": self.embedding_batch_size,
        }


def cache_folder() -> str | None:
    return os.environ.get("SENTENCE_TRANSFORMERS_HOME") or os.environ.get("HF_HOME")


def match_embedding_device() -> str:
    return os.environ.get("MATCH_EMBED_DEVICE", "cpu")


def effective_embedding_batch_size(requested: int) -> int:
    if match_embedding_device() != "cpu":
        return requested
    cap = int(os.environ.get("MATCH_EMBED_CPU_BATCH_SIZE", "64"))
    return max(1, min(requested, cap))


def default_input_path(retriever_model: str) -> Path:
    try:
        return RAW_PREDICTION_PATHS[retriever_model]
    except KeyError as exc:
        raise ValueError(f"Unsupported retriever model for matching: {retriever_model}") from exc


def default_output_path(retriever_model: str) -> Path:
    try:
        return MATCHED_OUTPUT_PATHS[retriever_model]
    except KeyError as exc:
        raise ValueError(f"Unsupported retriever model for matching: {retriever_model}") from exc


def default_relfact_csv(prompt_mode: str) -> Path:
    normalized_prompt_mode = normalize_prompt_mode(prompt_mode)
    try:
        return DEFAULT_RELFACT_CSVS[normalized_prompt_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt mode for Rel_Fact matching: {prompt_mode}") from exc


def build_match_plan(
    dataset: str,
    retriever_model: str,
    prompt_mode: str = "original",
    input_txt: Path | None = None,
    data_json: Path | None = None,
    relfact_csv: Path | None = None,
    output_json: Path | None = None,
    embedding_batch_size: int = 256,
) -> MatchPlan:
    if dataset != "finqa":
        raise ValueError("Only FinQA matching is active in the thesis workflow.")
    normalized_prompt_mode = normalize_prompt_mode(prompt_mode)
    return MatchPlan(
        dataset=dataset,
        retriever_model=retriever_model,
        prompt_mode=normalized_prompt_mode,
        input_txt=input_txt or default_input_path(retriever_model),
        data_json=data_json or DEFAULT_DATA_JSON,
        relfact_csv=relfact_csv or default_relfact_csv(normalized_prompt_mode),
        output_json=output_json or default_output_path(retriever_model),
        embedding_batch_size=embedding_batch_size,
    )


def validate_plan(plan: MatchPlan) -> list[str]:
    errors = []
    if not plan.input_txt.exists():
        errors.append(f"Missing raw prediction file: {plan.input_txt}")
    if not plan.data_json.exists():
        errors.append(f"Missing FinQA table/text file: {plan.data_json}")
    if not plan.relfact_csv.exists():
        errors.append(f"Missing Rel_Fact CSV file: {plan.relfact_csv}")
    if plan.output_json.exists() and plan.output_json.is_dir():
        errors.append(f"Output path is a directory: {plan.output_json}")
    if plan.embedding_batch_size < 1:
        errors.append(f"Embedding batch size must be positive: {plan.embedding_batch_size}")
    return errors


def nonempty(parts: list[str]) -> list[str]:
    return [part.strip() for part in parts if len(part.strip()) > 1]


def prediction_text(line: str) -> str:
    marker = "Pred:"
    start = line.find(marker)
    return line[start + len(marker) :] if start >= 0 else line


def repaired_schema_text(line: str, prompt_mode: str) -> str:
    text = prediction_text(line).strip()
    if not schema_required(prompt_mode):
        return text
    result = parse_retfact_schema(text)
    if result.valid and result.ret_fact.strip():
        return text
    repaired = assembler_schema_prediction(text)
    repaired_result = parse_retfact_schema(repaired)
    if repaired_result.valid and repaired_result.ret_fact.strip():
        return repaired
    return text


def prediction_fragments(line: str, prompt_mode: str) -> list[str]:
    text = prediction_text(line).strip()
    if not schema_required(prompt_mode):
        return nonempty(text.split(";"))
    result = parse_retfact_schema(text)
    if result.valid and result.ret_fact.strip():
        return nonempty(result.ret_fact.split(";"))
    return []


def prediction_retfact_text(line: str, prompt_mode: str) -> str:
    text = prediction_text(line).strip()
    if not schema_required(prompt_mode):
        return text.strip()
    result = parse_retfact_schema(text)
    if result.valid and result.ret_fact.strip():
        return result.ret_fact.strip()
    return ""


def prediction_schema_repair(line: str, prompt_mode: str) -> dict[str, Any] | None:
    if not schema_required(prompt_mode):
        return None
    text = prediction_text(line)
    raw_result = parse_retfact_schema(text)
    if raw_result.valid and raw_result.ret_fact.strip():
        return None
    repaired = assembler_schema_prediction(text)
    repaired_result = parse_retfact_schema(repaired)
    if not (repaired_result.valid and repaired_result.ret_fact.strip()):
        return None
    return {
        "repaired_json_retfact_schema": True,
        "raw_errors": list(raw_result.errors),
        "raw_prediction_preview": " ".join(text.split())[:500],
        "repaired_retfact_preview": repaired_result.ret_fact[:500],
    }


def prediction_schema_failure(line: str, prompt_mode: str) -> dict[str, Any] | None:
    if not schema_required(prompt_mode):
        return None
    text = prediction_text(line)
    result = parse_retfact_schema(text)
    if result.valid and result.ret_fact.strip():
        return None
    return {
        "valid_json_retfact_schema": False,
        "errors": list(result.errors),
        "raw_prediction_preview": " ".join(prediction_text(line).split())[:500],
    }


def match_prediction_fragments(line: str, prompt_mode: str) -> list[str]:
    text = repaired_schema_text(line, prompt_mode)
    if not schema_required(prompt_mode):
        return nonempty(text.split(";"))
    result = parse_retfact_schema(text)
    if result.valid and result.ret_fact.strip():
        return nonempty(result.ret_fact.split(";"))
    return []


def match_prediction_retfact_text(line: str, prompt_mode: str) -> str:
    text = repaired_schema_text(line, prompt_mode)
    if not schema_required(prompt_mode):
        return text.strip()
    result = parse_retfact_schema(text)
    if result.valid and result.ret_fact.strip():
        return result.ret_fact.strip()
    return ""


def prediction_records(text: str) -> list[str]:
    """Return one prediction record per True/Pred block.

    Some generative retrievers emit newline-heavy predictions.  Legacy matching
    assumed one record per physical line, which breaks those outputs.  This
    parser keeps the same True/Pred semantics while joining continuation lines.
    """

    records: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        starts_record = line.startswith("True:") and "Pred:" in line
        if starts_record and current:
            records.append(" ".join(current))
            current = []
        current.append(line)
    if current:
        records.append(" ".join(current))
    return records


def prediction_record_to_line(record: dict[str, Any]) -> str:
    true_label = str(
        record.get("true_label")
        or record.get("target_label")
        or record.get("true")
        or ""
    )
    predicted_label = str(
        record.get("predicted_label")
        or record.get("prediction")
        or record.get("pred")
        or record.get("predicted")
        or record.get("model_output")
        or record.get("predicted_retfact")
        or ""
    )
    return f"True: {normalize_inline(true_label)} Pred: {normalize_inline(predicted_label)}"


def normalize_inline(text: str) -> str:
    return " ".join(str(text).split())


def json_prediction_records(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        for key in ("records", "predictions", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return json_prediction_records(value)
        return [prediction_record_to_line(payload)]
    if isinstance(payload, list):
        lines = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("JSON prediction list items must be objects.")
            lines.append(prediction_record_to_line(item))
        return lines
    raise ValueError("JSON prediction artifact must be an object, list, or JSONL object stream.")


def read_prediction_records(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        items = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
        return json_prediction_records(items)
    if suffix == ".json":
        return json_prediction_records(json.loads(text))
    return prediction_records(text)


def read_relfact_labels(path: Path) -> list[str]:
    csv.field_size_limit(sys.maxsize)
    labels: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"Rel_Fact"}.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            labels.append(str(row.get("Rel_Fact", "")).strip())
    return labels


def fallback_sentence_tokenize(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return re.split(r"(?<=[.!?])\s+", normalized)


def sentence_tokenize(text: str) -> list[str]:
    try:
        from nltk import sent_tokenize

        return sent_tokenize(text)
    except LookupError:
        return fallback_sentence_tokenize(text)


def context_sentences(example: dict[str, Any]) -> list[str]:
    text = sentence_tokenize(example.get("text") or "")
    table_text = example.get("table_text")
    table = []
    if table_text is not None and not isinstance(table_text, float):
        table = str(table_text).split(";")
    return nonempty(text + table)


def top_matches(
    query: str,
    candidates: list[str],
    embeddings,
    model,
    top_k: int,
    matched_by: str,
    query_index: int,
) -> list[dict[str, Any]]:
    import torch
    from sentence_transformers import util

    if not candidates or not str(query).strip():
        return []
    embedded_query = model.encode(query)
    legacy_scores = util.dot_score(embedded_query, embeddings)
    cosine_scores = util.cos_sim(embedded_query, embeddings)
    k = min(top_k, len(candidates))
    values, indices = torch.topk(legacy_scores, k)
    cosine_values = cosine_scores[0][indices[0]].tolist()

    matches = []
    for rank, (index, legacy_score, cosine_score) in enumerate(
        zip(indices.tolist()[0], values.tolist()[0], cosine_values),
        start=1,
    ):
        matches.append(
            {
                "text": candidates[index],
                "rank": rank,
                "matched_by": matched_by,
                "query_index": query_index,
                "query": query,
                "cosine_similarity": float(cosine_score),
                "legacy_dot_score": float(legacy_score),
                "score_selection": "legacy_sentence_transformers_dot_score",
            }
        )
    return matches


def encode_sentences(model, texts: list[str], embedding_batch_size: int):
    try:
        return model.encode(
            texts,
            batch_size=embedding_batch_size,
            show_progress_bar=False,
        )
    except TypeError:
        try:
            return model.encode(texts, batch_size=embedding_batch_size)
        except TypeError:
            return model.encode(texts)


def encode_normalized(model, texts: list[str], embedding_batch_size: int):
    import torch
    import torch.nn.functional as F

    def as_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    try:
        return model.encode(
            texts,
            batch_size=embedding_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except TypeError:
        try:
            embeddings = model.encode(
                texts,
                batch_size=embedding_batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        except TypeError:
            embeddings = model.encode(texts)
        embeddings = as_tensor(embeddings)
        return F.normalize(embeddings, p=2, dim=1)


def relfact_label_matches(
    queries: list[str],
    rel_fact_labels: list[str],
    model,
    embedding_batch_size: int,
) -> list[dict[str, Any] | None]:
    if len(queries) != len(rel_fact_labels):
        raise ValueError("queries and Rel_Fact labels must have the same length.")

    matches: list[dict[str, Any] | None] = [None] * len(queries)
    valid_indexes = [
        index
        for index, (query, label) in enumerate(zip(queries, rel_fact_labels))
        if query.strip() and label.strip()
    ]
    if not valid_indexes:
        return matches

    valid_queries = [queries[index] for index in valid_indexes]
    valid_labels = [rel_fact_labels[index] for index in valid_indexes]
    import torch

    with torch.inference_mode():
        query_embeddings = encode_normalized(model, valid_queries, embedding_batch_size)
        label_embeddings = encode_normalized(model, valid_labels, embedding_batch_size)
        cosine_scores = (query_embeddings * label_embeddings).sum(dim=1).detach().cpu().tolist()

    for offset, index in enumerate(valid_indexes):
        score = float(cosine_scores[offset])
        rel_fact_label = rel_fact_labels[index]
        matches[index] = {
            "text": rel_fact_label,
            "rank": 1,
            "matched_by": "retfact_vs_rel_fact_label",
            "query_index": 0,
            "query": queries[index],
            "target_rel_fact": rel_fact_label,
            "cosine_similarity": score,
            "legacy_dot_score": score,
            "score_selection": "batched_direct_retfact_to_relfact_label_cosine",
        }
    return matches


def unique_texts(matches: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(match["text"] for match in matches))


def unique_scored_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for match in matches:
        key = (str(match.get("matched_by", "")), str(match["text"]))
        existing = unique.get(key)
        if existing is None:
            unique[key] = {**match, "duplicate_match_count": 1}
            continue
        existing["duplicate_match_count"] += 1
        if match["legacy_dot_score"] > existing["legacy_dot_score"]:
            unique[key] = {**match, "duplicate_match_count": existing["duplicate_match_count"]}
    return list(unique.values())


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def score_summary(matched: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        match
        for item in matched
        for match in item.get("retrieved_with_scores", [])
    ]
    primary_context_matches = [
        match
        for match in matches
        if str(match.get("matched_by", "unknown")) in PRIMARY_CONTEXT_MATCH_TYPES
    ]
    question_matches = [
        match
        for match in matches
        if str(match.get("matched_by", "unknown")) == "question"
    ]
    relfact_diagnostic_matches = [
        match
        for match in matches
        if str(match.get("matched_by", "unknown")) in AUXILIARY_RETFACT_MATCH_TYPES
    ]
    cosine_scores = [
        float(match["cosine_similarity"])
        for match in matches
        if match.get("cosine_similarity") is not None
    ]
    legacy_scores = [
        float(match["legacy_dot_score"])
        for match in matches
        if match.get("legacy_dot_score") is not None
    ]
    by_matched_by: dict[str, dict[str, Any]] = {}
    for matched_by in sorted({str(match.get("matched_by", "unknown")) for match in matches}):
        group = [match for match in matches if str(match.get("matched_by", "unknown")) == matched_by]
        by_matched_by[matched_by] = {
            "cosine_similarity_summary": numeric_summary(
                [
                    float(match["cosine_similarity"])
                    for match in group
                    if match.get("cosine_similarity") is not None
                ]
            ),
            "legacy_dot_score_summary": numeric_summary(
                [
                    float(match["legacy_dot_score"])
                    for match in group
                    if match.get("legacy_dot_score") is not None
                ]
            ),
        }
    return {
        "score_selection": "finder_context_sentence_matching",
        "matched_sentences": len(matches),
        "primary_metric": "primary_prediction_fragment_legacy_dot_score_summary.mean",
        "primary_prediction_fragment_cosine_summary": numeric_summary(
            [
                float(match["cosine_similarity"])
                for match in primary_context_matches
                if match.get("cosine_similarity") is not None
            ]
        ),
        "primary_prediction_fragment_legacy_dot_score_summary": numeric_summary(
            [
                float(match["legacy_dot_score"])
                for match in primary_context_matches
                if match.get("legacy_dot_score") is not None
            ]
        ),
        "auxiliary_question_cosine_summary": numeric_summary(
            [
                float(match["cosine_similarity"])
                for match in question_matches
                if match.get("cosine_similarity") is not None
            ]
        ),
        "auxiliary_question_legacy_dot_score_summary": numeric_summary(
            [
                float(match["legacy_dot_score"])
                for match in question_matches
                if match.get("legacy_dot_score") is not None
            ]
        ),
        "auxiliary_retfact_label_cosine_summary": numeric_summary(
            [
                float(match["cosine_similarity"])
                for match in relfact_diagnostic_matches
                if match.get("cosine_similarity") is not None
            ]
        ),
        "auxiliary_retfact_label_legacy_dot_score_summary": numeric_summary(
            [
                float(match["legacy_dot_score"])
                for match in relfact_diagnostic_matches
                if match.get("legacy_dot_score") is not None
            ]
        ),
        "cosine_similarity_summary": numeric_summary(cosine_scores),
        "legacy_dot_score_summary": numeric_summary(legacy_scores),
        "by_matched_by": by_matched_by,
        "score_type": "legacy_dot_score",
        "mean": sum(legacy_scores) / len(legacy_scores) if legacy_scores else None,
        "min": min(legacy_scores) if legacy_scores else None,
        "max": max(legacy_scores) if legacy_scores else None,
    }


def similarity_jsonl_row(row_index: int, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "id": item.get("id"),
        "question": item.get("question"),
        "predicted_retfact_for_match": item.get("predicted_retfact_for_match"),
        "prediction_fragments_for_match": item.get("prediction_fragments_for_match", []),
        "retrieved_with_scores": item.get("retrieved_with_scores", []),
        "score_summary": score_summary([item]),
        "score_selection_note": (
            "Top-k context matches are selected with SentenceTransformers util.dot_score, "
            "matching FINDER legacy code; cosine_similarity is recorded for visibility."
        ),
    }


def write_similarity_jsonl(matched: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, item in enumerate(matched):
            handle.write(json.dumps(similarity_jsonl_row(row_index, item), ensure_ascii=False) + "\n")
    return {
        "rows": len(matched),
        "similarity_jsonl_output": str(output_path),
    }


def read_matched_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Matched artifact must be a JSON list: {path}")
    return payload


def summarize_matched_json(
    matched_json: Path,
    similarity_jsonl_output: Path | None = None,
) -> dict[str, Any]:
    matched = read_matched_json(matched_json)
    result: dict[str, Any] = {
        "rows": len(matched),
        "matched_json": str(matched_json),
        "score_summary": score_summary(matched),
    }
    if similarity_jsonl_output is not None:
        result.update(write_similarity_jsonl(matched, similarity_jsonl_output))
    return result


def schema_validation_summary(
    matched: list[dict[str, Any]],
    prompt_mode: str,
    schema_failures: int,
    schema_repairs: int = 0,
) -> dict[str, Any]:
    primary_rows = 0
    for item in matched:
        if any(
            str(match.get("matched_by", "")) in PRIMARY_CONTEXT_MATCH_TYPES
            for match in item.get("retrieved_with_scores", [])
        ):
            primary_rows += 1
    requires_schema = schema_required(prompt_mode)
    return {
        "schema_required": requires_schema,
        "rows": len(matched),
        "schema_failures": schema_failures,
        "schema_repairs": schema_repairs,
        "valid_schema_rows": len(matched) - schema_failures,
        "valid_primary_retfact_rows": primary_rows,
        "valid_primary_context_rows": primary_rows,
        "artifact_valid_for_non_original_generator": (
            not requires_schema
            or (schema_failures == 0 and primary_rows == len(matched))
        ),
    }


def run_match(
    plan: MatchPlan,
    similarity_jsonl_output: Path | None = None,
    expected_rows: int | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    errors = validate_plan(plan)
    if errors:
        raise FileNotFoundError("; ".join(errors))

    data = json.loads(plan.data_json.read_text(encoding="utf-8"))
    relfact_labels = read_relfact_labels(plan.relfact_csv)
    lines = read_prediction_records(plan.input_txt)

    if len(lines) > len(data):
        raise ValueError(
            f"Prediction file has {len(lines)} records but data has only {len(data)} examples."
        )
    if len(lines) > len(relfact_labels):
        raise ValueError(
            f"Prediction file has {len(lines)} records but Rel_Fact CSV has only {len(relfact_labels)} examples."
        )

    available_rows = min(len(data), len(relfact_labels))
    required_rows = expected_rows if expected_rows is not None else available_rows
    if required_rows < 0:
        raise ValueError(f"expected_rows must be non-negative; got {required_rows}.")
    if required_rows > available_rows:
        raise ValueError(
            f"expected_rows={required_rows} exceeds available matched inputs: "
            f"data_rows={len(data)}, relfact_rows={len(relfact_labels)}."
        )
    if not allow_partial and len(lines) != required_rows:
        raise ValueError(
            f"Prediction file has {len(lines)} records but expected {required_rows}. "
            "Use --allow-partial only for explicit partial smoke runs."
        )
    if allow_partial and expected_rows is not None and len(lines) > expected_rows:
        raise ValueError(
            f"Prediction file has {len(lines)} records but expected_rows={expected_rows}."
        )

    from sentence_transformers import SentenceTransformer

    embedding_device = match_embedding_device()
    embedding_batch_size = effective_embedding_batch_size(plan.embedding_batch_size)
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_folder(), device=embedding_device)

    matched = []
    schema_failures = 0
    schema_repairs = 0
    for index, line in enumerate(lines):
        item = dict(data[index])
        rel_fact_label = relfact_labels[index]
        repair = prediction_schema_repair(line, plan.prompt_mode)
        if repair is not None:
            item["retriever_schema_repair"] = repair
            schema_repairs += 1
        else:
            failure = prediction_schema_failure(line, plan.prompt_mode)
            if failure is not None:
                item["retriever_schema_failure"] = failure
                schema_failures += 1

        predicted_retfact = match_prediction_retfact_text(line, plan.prompt_mode)
        fragments = match_prediction_fragments(line, plan.prompt_mode)
        candidates = context_sentences(item)
        context_matches: list[dict[str, Any]] = []
        if candidates:
            embeddings = encode_sentences(model, candidates, embedding_batch_size)
            for fragment_index, fragment in enumerate(fragments):
                context_matches.extend(
                    top_matches(
                        fragment,
                        candidates,
                        embeddings,
                        model,
                        top_k=1,
                        matched_by="prediction_fragment",
                        query_index=fragment_index,
                    )
                )
            context_matches.extend(
                top_matches(
                    str(item.get("question", "")),
                    candidates,
                    embeddings,
                    model,
                    top_k=3,
                    matched_by="question",
                    query_index=0,
                )
            )

        item["rel_fact_label"] = rel_fact_label
        item["predicted_retfact_for_match"] = predicted_retfact
        item["prediction_fragments_for_match"] = fragments
        item["retrieved"] = unique_texts(context_matches)
        item["retrieved_with_scores"] = unique_scored_matches(context_matches)
        matched.append(item)

    predicted_retfacts = [
        str(item.get("predicted_retfact_for_match", ""))
        for item in matched
    ]
    diagnostic_matches = relfact_label_matches(
        predicted_retfacts,
        relfact_labels[: len(matched)],
        model,
        embedding_batch_size,
    )
    for item, diagnostic_match in zip(matched, diagnostic_matches):
        if diagnostic_match is not None:
            item["retrieved_with_scores"] = unique_scored_matches(
                item["retrieved_with_scores"] + [diagnostic_match]
            )

    plan.output_json.parent.mkdir(parents=True, exist_ok=True)
    plan.output_json.write_text(json.dumps(matched, ensure_ascii=False) + "\n", encoding="utf-8")
    validation = schema_validation_summary(matched, plan.prompt_mode, schema_failures, schema_repairs)
    result = {
        "rows": len(matched),
        "expected_rows": required_rows,
        "allow_partial": allow_partial,
        "output_json": str(plan.output_json),
        "prompt_mode": plan.prompt_mode,
        "embedding_device": embedding_device,
        "embedding_batch_size_requested": plan.embedding_batch_size,
        "embedding_batch_size_effective": embedding_batch_size,
        "score_summary": score_summary(matched),
        "schema_failures": schema_failures,
        "schema_repairs": schema_repairs,
        "schema_validation": validation,
    }
    if similarity_jsonl_output is not None:
        result.update(write_similarity_jsonl(matched, similarity_jsonl_output))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize retriever outputs into matched FinQA artifacts.")
    parser.add_argument("action", choices=["match", "validate", "summarize"])
    parser.add_argument("--dataset", default="finqa")
    parser.add_argument("--retriever-model", default="mistral_v0_3")
    parser.add_argument("--prompt-mode", default="original")
    parser.add_argument("--input-txt", type=Path)
    parser.add_argument("--data-json", type=Path)
    parser.add_argument("--relfact-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--matched-json",
        type=Path,
        help="Existing matched JSON artifact to summarize or export without rerunning matching.",
    )
    parser.add_argument(
        "--similarity-jsonl-output",
        type=Path,
        help="Optional JSONL companion artifact with per-row cosine and legacy dot-score details.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.environ.get("MATCH_EMBED_BATCH_SIZE", "256")),
        help="Sentence-transformer embedding batch size for context matching diagnostics.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--expected-rows",
        type=int,
        help="Required prediction row count for formal matching. Defaults to all available data/Rel_Fact rows.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow fewer prediction rows than expected; use only for explicit partial smoke runs.",
    )
    parser.add_argument(
        "--require-valid-schema",
        action="store_true",
        help="For non-original prompt modes, fail if any row lacks a valid RetFact JSON match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "summarize":
        matched_json = args.matched_json or args.output_json
        errors = []
        if matched_json is None:
            errors.append("Missing --matched-json or --output-json for summarize.")
        elif not matched_json.exists():
            errors.append(f"Missing matched JSON file: {matched_json}")
        payload: dict[str, Any] = {
            "matched_json": str(matched_json) if matched_json is not None else None,
            "errors": errors,
        }
        if not errors and matched_json is not None:
            payload["result"] = summarize_matched_json(
                matched_json,
                similarity_jsonl_output=args.similarity_jsonl_output,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if errors:
            raise SystemExit(2)
        return

    plan = build_match_plan(
        dataset=args.dataset,
        retriever_model=args.retriever_model,
        prompt_mode=args.prompt_mode,
        input_txt=args.input_txt,
        data_json=args.data_json,
        relfact_csv=args.relfact_csv,
        output_json=args.output_json,
        embedding_batch_size=args.embedding_batch_size,
    )
    errors = validate_plan(plan)
    payload: dict[str, Any] = {"plan": plan.to_dict(), "errors": errors}
    if args.action == "match" and args.execute:
        payload["result"] = run_match(
            plan,
            similarity_jsonl_output=args.similarity_jsonl_output,
            expected_rows=args.expected_rows,
            allow_partial=args.allow_partial,
        )
        validation = payload["result"].get("schema_validation", {})
        if args.require_valid_schema and not validation.get("artifact_valid_for_non_original_generator", False):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            raise SystemExit(3)
    elif args.action == "match":
        payload["note"] = "Dry run only. Add --execute to write the matched artifact."
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
