#!/usr/bin/env python3
"""CPU smoke checks for Progress 5 FINDER-style matching gates."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from result_organization import build_match_plan, run_match  # noqa: E402
from retriever_json_schema import label_for_prompt_mode  # noqa: E402


VOCAB = (
    "revenue",
    "gross",
    "margin",
    "improved",
    "costs",
    "safe",
    "question",
    "binding",
    "reason",
)


def install_fake_sentence_transformers() -> None:
    module = types.ModuleType("sentence_transformers")

    def vectorize(text: str) -> torch.Tensor:
        value = str(text).lower()
        counts = [float(value.count(token)) for token in VOCAB]
        counts.append(1.0)
        return torch.tensor(counts, dtype=torch.float32)

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, value, *args, **kwargs):
            if isinstance(value, list):
                if not value:
                    return torch.empty((0, len(VOCAB) + 1), dtype=torch.float32)
                return torch.stack([vectorize(item) for item in value])
            return vectorize(value)

    class FakeUtil:
        @staticmethod
        def dot_score(query, embeddings):
            query = query if query.dim() == 2 else query.unsqueeze(0)
            embeddings = embeddings if embeddings.dim() == 2 else embeddings.unsqueeze(0)
            return query @ embeddings.T

        @staticmethod
        def cos_sim(query, embeddings):
            query = query if query.dim() == 2 else query.unsqueeze(0)
            embeddings = embeddings if embeddings.dim() == 2 else embeddings.unsqueeze(0)
            return torch.nn.functional.cosine_similarity(query.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

    module.SentenceTransformer = FakeSentenceTransformer
    module.util = FakeUtil
    sys.modules["sentence_transformers"] = module


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def assert_no_text_leak(values: list[str], forbidden: list[str]) -> None:
    joined = "\n".join(values).lower()
    for marker in forbidden:
        assert marker.lower() not in joined, f"unexpected leaked text: {marker}"


def main() -> None:
    install_fake_sentence_transformers()
    root = Path(tempfile.mkdtemp(prefix="fqan_progress5_matching_"))
    data_json = root / "data.json"
    relfact_csv = root / "relfact.csv"
    original_txt = root / "original_predictions.txt"
    schema_txt = root / "schema_predictions.txt"
    original_out = root / "original_matched.json"
    schema_out = root / "schema_matched.json"
    schema_out_repeat = root / "schema_matched_repeat.json"

    data = [
        {
            "question": "What happened to revenue and gross margin?",
            "text": "Revenue increased. Operating costs stayed flat.",
            "table_text": "gross margin improved; cash rose",
        },
        {
            "question": "Which safe item is relevant?",
            "text": "Only question should match.",
            "table_text": "safe table sentence; backup context",
        },
    ]
    write_json(data_json, data)
    relfact_csv.write_text(
        "input,Rel_Fact\n"
        '"prompt","Revenue target label not in context"\n'
        '"prompt","Target label should not match malformed schema"\n',
        encoding="utf-8",
    )
    original_txt.write_text("True: target Pred: Revenue increased; gross margin improved\n", encoding="utf-8")
    valid_schema = label_for_prompt_mode("Revenue increased", "zero-shot")
    malformed_schema = '{"Binding":[{"Text":"do not match this binding text"}],"Reason":"bad reason text"}'
    schema_txt.write_text(
        f"True: target Pred: {valid_schema}\nTrue: target Pred: {malformed_schema}\n",
        encoding="utf-8",
    )

    original_result = run_match(
        build_match_plan(
            dataset="finqa",
            retriever_model="mistral_v0_3",
            prompt_mode="original",
            input_txt=original_txt,
            data_json=data_json,
            relfact_csv=relfact_csv,
            output_json=original_out,
        )
    )
    original_payload = json.loads(original_out.read_text(encoding="utf-8"))
    assert original_result["schema_failures"] == 0
    assert "Revenue increased." in original_payload[0]["retrieved"]
    assert "gross margin improved" in original_payload[0]["retrieved"]
    assert original_payload[0]["retrieved"] == list(dict.fromkeys(original_payload[0]["retrieved"]))
    assert "Revenue target label not in context" not in original_payload[0]["retrieved"]
    assert any(
        match["matched_by"] == "prediction_fragment"
        for match in original_payload[0]["retrieved_with_scores"]
    ), "original route should match Pred fragments against same-row context/table sentences"
    assert any(
        match["matched_by"] == "question"
        for match in original_payload[0]["retrieved_with_scores"]
    ), "original route should append question top-3 context/table matches"
    assert any(
        match["matched_by"] == "retfact_vs_rel_fact_label"
        for match in original_payload[0]["retrieved_with_scores"]
    ), "Rel_Fact label comparison should remain diagnostic"

    schema_plan = build_match_plan(
        dataset="finqa",
        retriever_model="mistral_v0_3",
        prompt_mode="zero-shot",
        input_txt=schema_txt,
        data_json=data_json,
        relfact_csv=relfact_csv,
        output_json=schema_out,
    )
    schema_result = run_match(schema_plan)
    schema_payload = json.loads(schema_out.read_text(encoding="utf-8"))
    prediction_queries = [
        match["query"]
        for match in schema_payload[0]["retrieved_with_scores"]
        if match["matched_by"] == "prediction_fragment"
    ]
    assert prediction_queries == ["Revenue increased"], prediction_queries
    assert "Revenue increased." in schema_payload[0]["retrieved"]
    assert_no_text_leak(schema_payload[0]["retrieved"], ["do not match this binding text", "bad reason text"])
    assert "retriever_schema_failure" in schema_payload[1], "malformed schema must be recorded"
    assert not any(
        match["matched_by"] == "prediction_fragment"
        for match in schema_payload[1]["retrieved_with_scores"]
    ), "malformed schema must not match arbitrary generated text"
    assert_no_text_leak(schema_payload[1]["retrieved"], ["do not match this binding text", "bad reason text"])
    assert schema_result["schema_failures"] == 1
    assert schema_result["schema_validation"]["valid_primary_retfact_rows"] == 1
    assert schema_result["schema_validation"]["valid_primary_context_rows"] == 1
    assert not schema_result["schema_validation"]["artifact_valid_for_non_original_generator"]

    repeat_plan = build_match_plan(
        dataset="finqa",
        retriever_model="mistral_v0_3",
        prompt_mode="zero-shot",
        input_txt=schema_txt,
        data_json=data_json,
        relfact_csv=relfact_csv,
        output_json=schema_out_repeat,
    )
    run_match(repeat_plan)
    repeat_payload = json.loads(schema_out_repeat.read_text(encoding="utf-8"))
    assert [row["retrieved"] for row in schema_payload] == [
        row["retrieved"] for row in repeat_payload
    ], "retrieved ordering must be deterministic"

    print(
        json.dumps(
            {
                "status": "ok",
                "tmp_dir": str(root),
                "original_rows": original_result["rows"],
                "schema_rows": schema_result["rows"],
                "schema_failures": schema_result["schema_failures"],
                "schema_validation": schema_result["schema_validation"],
                "schema_prediction_queries": prediction_queries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
