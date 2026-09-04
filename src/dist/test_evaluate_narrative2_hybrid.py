#!/usr/bin/env python3
"""Deterministic and metamorphic tests for narrative2 hybrid v2."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_narrative2_hybrid as evaluator


NUMERIC = {"absoluteTolerance": 1e-9, "relativeTolerance": 1e-9}
ALIASES = {
    "rise": "increase",
    "increased": "increase",
    "increase": "increase",
    "fell": "decrease",
    "decrease": "decrease",
}


def binding(
    *,
    object_name: list[str] | None = None,
    data_name: object = "Revenue",
    position: object = None,
    trend: object = "increase",
    num: object = None,
    text: object = "Revenue increased.",
) -> dict[str, object]:
    return {
        "ObjectName": object_name or ["Revenue"],
        "DataName": data_name,
        "Position": (
            [{"Begin": [0, 1], "End": [0, 1]}]
            if position is None else position
        ),
        "Trend": trend,
        "Num": [1.0, 2.0] if num is None else num,
        "Text": text,
    }


JUDGE_CONFIG = {
    "baseUrl": "http://localhost:1/v1",
    "apiKeyEnvironment": "NARRATIVE2_TEST_API_KEY",
    "defaultApiKey": "test",
    "requestTimeoutSeconds": 1,
    "model": "gpt-5.5",
    "reasoningEffort": "medium",
    "minimumConfidence": 0.8,
    "maxAttempts": 1,
    "retryDelaysSeconds": [],
    "maxTokens": 512,
}


class DisabledJudge(evaluator.SemanticJudge):
    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__(JUDGE_CONFIG, checkpoint_path, disabled=True)


class FakeJudge(evaluator.SemanticJudge):
    def __init__(self, checkpoint_path: Path, response: dict[str, object]) -> None:
        self.response = response
        super().__init__(JUDGE_CONFIG, checkpoint_path)

    def _request(self, prompt: str, seed: int) -> tuple[str, str, str]:
        del prompt, seed
        return json.dumps(self.response), "gpt-5.5", "test-response"


class HybridEvaluatorV2Tests(unittest.TestCase):
    def test_main_treats_partial_no_ranking_as_success(self) -> None:
        for status, expected in (
            ("completed", 0),
            ("development_partial_no_ranking", 0),
            ("incomplete_no_ranking", 2),
        ):
            with self.subTest(status=status), patch.object(
                evaluator,
                "parse_args",
                return_value=SimpleNamespace(),
            ), patch.object(
                evaluator,
                "build",
                return_value={"status": status},
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(evaluator.main([]), expected)

    def test_hard_identity_is_case_type_value_and_order_sensitive(self) -> None:
        self.assertNotEqual(
            evaluator.fixed_canonical("Revenue"),
            evaluator.fixed_canonical("revenue"),
        )
        self.assertNotEqual(
            evaluator.fixed_canonical(" Revenue "),
            evaluator.fixed_canonical("Revenue"),
        )
        self.assertNotEqual(
            evaluator.fixed_canonical([0, 1]),
            evaluator.fixed_canonical(["0", 1]),
        )
        self.assertNotEqual(
            evaluator.fixed_canonical([0, 1]),
            evaluator.fixed_canonical([1, 0]),
        )
        self.assertEqual(
            evaluator.fixed_canonical({"b": 2, "a": 1}),
            evaluator.fixed_canonical({"a": 1.0, "b": 2.0}),
        )

    def test_binding_reordering_is_invariant_but_anchor_change_is_not(self) -> None:
        gold = [binding(data_name="A"), binding(data_name="B")]
        prediction = [binding(data_name="B"), binding(data_name="A")]
        pairs, missing, extra = evaluator.align_bindings(gold, prediction)
        self.assertEqual((pairs, missing, extra), ([(0, 1), (1, 0)], [], []))

        changed = [binding(data_name="a")]
        pairs, missing, extra = evaluator.align_bindings(
            [binding(data_name="A")], changed
        )
        self.assertEqual((pairs, missing, extra), ([], [0], [0]))

    def test_duplicate_anchor_reordering_preserves_pair_content(self) -> None:
        gold = [
            binding(object_name=["Alpha"], text="Alpha increased."),
            binding(object_name=["Beta"], text="Beta increased."),
        ]
        prediction = [gold[1].copy(), gold[0].copy()]

        def paired_names(
            left: list[dict[str, object]],
            right: list[dict[str, object]],
        ) -> list[tuple[object, object]]:
            pairs, missing, extra = evaluator.align_bindings(left, right)
            self.assertEqual((missing, extra), ([], []))
            return sorted(
                (left[left_index]["ObjectName"], right[right_index]["ObjectName"])
                for left_index, right_index in pairs
            )

        expected = [(["Alpha"], ["Alpha"]), (["Beta"], ["Beta"])]
        self.assertEqual(paired_names(gold, prediction), expected)
        self.assertEqual(
            paired_names(list(reversed(gold)), list(reversed(prediction))),
            expected,
        )

    def test_position_type_and_order_are_hard(self) -> None:
        gold = [binding(position=[{"Begin": [0, 1], "End": [0, 2]}])]
        wrong_type = [binding(position=[{"Begin": ["0", 1], "End": [0, 2]}])]
        wrong_order = [binding(position=[{"Begin": [1, 0], "End": [0, 2]}])]
        self.assertEqual(evaluator.align_bindings(gold, wrong_type)[0], [])
        self.assertEqual(evaluator.align_bindings(gold, wrong_order)[0], [])

    def test_numeric_units_percent_and_duplicates(self) -> None:
        counts, details = evaluator.numeric_counts(
            [1_000_000], ["$1 million"], 1e-9, 1e-9
        )
        self.assertEqual((counts.tp, counts.fp, counts.fn), (1, 0, 0))
        self.assertEqual(
            details["method"],
            "numeric_multiset_isclose_percentage_point_sensitive",
        )

        counts, _ = evaluator.numeric_counts(["12%"], ["12 percent"], 1e-9, 1e-9)
        self.assertEqual((counts.tp, counts.fp, counts.fn), (1, 0, 0))

        counts, _ = evaluator.numeric_counts(["12%"], [0.12], 1e-9, 1e-9)
        self.assertEqual((counts.tp, counts.fp, counts.fn), (0, 1, 1))

        counts, _ = evaluator.numeric_counts([1], [1, 1], 1e-9, 1e-9)
        self.assertEqual((counts.tp, counts.fp, counts.fn), (1, 1, 0))

    def test_empty_set_metric_is_well_defined(self) -> None:
        self.assertEqual(evaluator.Counts().as_dict()["f1"], 1.0)
        counts, _ = evaluator.numeric_counts(None, [], 1e-9, 1e-9)
        self.assertEqual((counts.tp, counts.fp, counts.fn), (0, 0, 0))

    def test_nfkc_whitespace_exact_fast_path_needs_no_judge(self) -> None:
        gold = binding()
        prediction = binding(
            object_name=["  Revenue  "],
            trend="increased",
            num=[2.0, 1.0],
            text="  Revenue   increased. ",
        )
        with tempfile.TemporaryDirectory() as directory:
            judge = DisabledJudge(Path(directory) / "checkpoint.jsonl")
            counts, details, calls = evaluator.evaluate_aligned_binding(
                "S1",
                "Revenue increased.",
                "[]",
                gold,
                prediction,
                0,
                0,
                ALIASES,
                NUMERIC,
                judge,
            )
        self.assertEqual(calls, 0)
        self.assertTrue(details["allFieldsExact"])
        self.assertTrue(all(value.fp == value.fn == 0 for value in counts.values()))

    def test_legal_object_rewrite_can_pass_only_with_validated_pair(self) -> None:
        gold = binding(object_name=["net revenue"])
        prediction = binding(object_name=["revenue, net"])
        plan = evaluator.build_semantic_plan("[]", gold, prediction, 0, ALIASES)
        decision_id = plan["objectDecisionId"]
        with tempfile.TemporaryDirectory() as directory:
            judge = DisabledJudge(Path(directory) / "checkpoint.jsonl")
            counts, details, calls = evaluator.evaluate_aligned_binding(
                "S1",
                "Net revenue increased.",
                "[]",
                gold,
                prediction,
                0,
                0,
                ALIASES,
                NUMERIC,
                judge,
                semantic_plan=plan,
                judge_results_override={
                    decision_id: {
                        "accepted": True,
                        "matchedPairs": [{"goldIndex": 0, "predictionIndex": 0}],
                    }
                },
            )
        self.assertEqual(calls, 0)
        self.assertEqual((counts["ObjectName"].tp, counts["ObjectName"].fp), (1, 0))
        self.assertEqual(
            details["fields"]["ObjectName"]["matchedPairs"],
            [{"goldIndex": 0, "predictionIndex": 0}],
        )

    def test_trend_or_number_mutation_must_reduce_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            judge = DisabledJudge(Path(directory) / "checkpoint.jsonl")
            trend_counts, _, _ = evaluator.evaluate_aligned_binding(
                "S1", "Revenue increased.", "[]",
                binding(), binding(trend="decrease"),
                0, 0, ALIASES, NUMERIC, judge,
            )
            num_counts, _, _ = evaluator.evaluate_aligned_binding(
                "S1", "Revenue increased.", "[]",
                binding(), binding(num=[9.0]),
                0, 0, ALIASES, NUMERIC, judge,
            )
        self.assertEqual(trend_counts["Trend"].as_dict()["f1"], 0.0)
        self.assertLess(num_counts["Num"].as_dict()["f1"], 1.0)

    def test_duplicate_binding_never_increases_true_positives(self) -> None:
        gold = [binding()]
        predictions = [binding(), binding()]
        pairs, missing, extra = evaluator.align_bindings(gold, predictions)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(missing, [])
        self.assertEqual(extra, [1])

    def test_decision_ids_are_unique_within_source_batch(self) -> None:
        first = evaluator.build_semantic_plan(
            "[]", binding(object_name=["a"], text="first"),
            binding(object_name=["b"], text="other"), 0, ALIASES,
        )
        second = evaluator.build_semantic_plan(
            "[]", binding(object_name=["c"], text="second"),
            binding(object_name=["d"], text="different"), 1, ALIASES,
        )
        identifiers = [
            item["decisionId"] for item in first["decisions"] + second["decisions"]
        ]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_blinded_ab_orders_are_opposites_for_audit(self) -> None:
        decision = {
            "decisionId": "binding_0_Text",
            "field": "Text",
            "gold": "A claim",
            "prediction": "B claim",
        }
        with tempfile.TemporaryDirectory() as directory:
            judge = DisabledJudge(Path(directory) / "checkpoint.jsonl")
            primary, primary_map = judge.public_decisions("S1", [decision], False)
            swapped, swapped_map = judge.public_decisions("S1", [decision], True)
        self.assertNotEqual(primary_map["binding_0_Text"], swapped_map["binding_0_Text"])
        self.assertEqual(primary[0]["A"], swapped[0]["B"])
        self.assertNotIn("gold", primary[0])
        self.assertNotIn("prediction", primary[0])

    def test_judge_rejects_extra_schema_fields(self) -> None:
        response = {
            "decisions": [{
                "decisionId": "binding_0_Text",
                "equivalent": True,
                "matchedPairs": [],
                "confidence": 0.9,
                "evidenceSpan": "Revenue increased.",
                "reasonCode": "same_claim",
            }],
            "extra": "not allowed",
        }
        with tempfile.TemporaryDirectory() as directory:
            judge = FakeJudge(Path(directory) / "checkpoint.jsonl", response)
            with self.assertRaises(evaluator.JudgeError):
                judge.decide(
                    "S1", "Revenue increased.",
                    [{
                        "decisionId": "binding_0_Text",
                        "field": "Text",
                        "gold": "Revenue rose.",
                        "prediction": "Revenue increased.",
                    }],
                )

    def test_low_confidence_or_nonverbatim_evidence_is_rejected(self) -> None:
        for confidence, evidence in ((0.79, "Revenue increased."), (0.95, "revenue increased")):
            response = {
                "decisions": [{
                    "decisionId": "binding_0_Text",
                    "equivalent": True,
                    "matchedPairs": [],
                    "confidence": confidence,
                    "evidenceSpan": evidence,
                    "reasonCode": "same_claim",
                }]
            }
            with self.subTest(confidence=confidence, evidence=evidence):
                with tempfile.TemporaryDirectory() as directory:
                    judge = FakeJudge(Path(directory) / "checkpoint.jsonl", response)
                    result = judge.decide(
                        "S1", "Revenue increased.",
                        [{
                            "decisionId": "binding_0_Text",
                            "field": "Text",
                            "gold": "Revenue rose.",
                            "prediction": "Revenue increased.",
                        }],
                    )
                self.assertFalse(result["binding_0_Text"]["accepted"])

    def test_invalid_object_pair_is_audited_and_fails_closed(self) -> None:
        invalid_pairs = (
            [{"goldIndex": 1, "predictionIndex": 0}],
            [
                {"goldIndex": 0, "predictionIndex": 0},
                {"goldIndex": 0, "predictionIndex": 0},
            ],
        )
        for matched_pairs in invalid_pairs:
            response = {
                "decisions": [{
                    "decisionId": "binding_0_ObjectName",
                    "equivalent": True,
                    "matchedPairs": matched_pairs,
                    "confidence": 0.99,
                    "evidenceSpan": "Revenue",
                    "reasonCode": "same_entity",
                }]
            }
            with self.subTest(matched_pairs=matched_pairs):
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "checkpoint.jsonl"
                    judge = FakeJudge(checkpoint, response)
                    result = judge.decide(
                        "S1", "Revenue increased.",
                        [{
                            "decisionId": "binding_0_ObjectName",
                            "field": "ObjectName",
                            "gold": ["Revenue"],
                            "prediction": ["revenue"],
                        }],
                    )["binding_0_ObjectName"]
                    checkpoint_record = json.loads(
                        checkpoint.read_text(encoding="utf-8").splitlines()[0]
                    )
                self.assertFalse(result["accepted"])
                self.assertEqual(result["matchedPairs"], [])
                self.assertIn("validationError", result)
                self.assertEqual(
                    checkpoint_record["rawResponse"], json.dumps(response)
                )

    def test_legacy_otn_uses_canonical_per_row_vocabulary_sets(self) -> None:
        targets = [{
            "source": "S1",
            "targetBindings": [
                binding(object_name=["Revenue"], trend="increase", num=[1, 2]),
                binding(object_name=["Sales"], trend="decrease", num=[3]),
            ],
        }]
        predictions = [{
            "source": "S1",
            "formatValid": True,
            "result": [
                binding(object_name=["Revenue"], trend="increase", num=[1, 2]),
                binding(object_name=["Revenue"], trend="increase", num=[4]),
            ],
        }]
        legacy = evaluator.legacy_exact_otn_metrics(targets, predictions)
        for field in ("ObjectName", "Trend", "Num"):
            self.assertAlmostEqual(legacy["byField"][field]["f1"], 2 / 3)
        wrapped = {"legacyExactOtn": legacy}
        self.assertAlmostEqual(evaluator.legacy_field_f1(wrapped, "ObjectName"), 2 / 3)

    def test_runtime_blocked_row_is_scored_empty_and_aggregation_stays_no_ranking(self) -> None:
        config = evaluator.load_config(
            evaluator.REPO_ROOT / "config" / "experiment6_narrative2_evaluation.json"
        )
        target = {
            "source": "S1",
            "excelRow": 2,
            "targetBindings": [binding()],
        }
        raw_response = ""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions_path = root / "predictions.jsonl"
            evaluator.write_jsonl(predictions_path, [{
                "source": "S1",
                "run": 1,
                "rawResponse": raw_response,
                "rawResponseSha256": evaluator.sha256_text(raw_response),
                "result": [],
                "formatValid": False,
                "parserDiagnostic": {"error": "generation_runtime_blocked"},
                "inputText": "Revenue increased.",
                "inputData": "[]",
            }])
            manifest = {
                "outputId": "6_llama_z",
                "sourceId": "llama4",
                "promptMode": "zero-shot",
                "part": 2,
                "official": True,
                "route": "direct-binding",
                "effectiveRoute": "direct-binding",
                "run": 1,
                "seed": 101,
                "expectedRows": 1,
                "acceptedRows": 0,
                "rejectedRows": 1,
                "runtimeBlockedRows": 1,
                "formatComplianceRate": 0.0,
                "status": "runtime_blocked",
                "requestedModel": "llama4",
                "actualModel": "/models/llama4",
                "adapter": None,
                "quantization": "w4a16",
                "runtimeProfile": "llama4-formal",
                "reasoningEffort": None,
                "converterModel": None,
                "runtimeSeconds": 600.0,
                "files": {"predictions": str(predictions_path)},
                "hashes": {
                    "predictions": evaluator.sha256_file(predictions_path),
                    "prompts": "prompt-hash",
                },
            }
            result = evaluator.evaluate_case(
                manifest,
                [target],
                config,
                root / "evaluation" / "run_01",
                judge_disabled=True,
            )
            self.assertEqual(result["status"], "runtime_blocked_scored_no_ranking")
            self.assertEqual(result["runtimeBlockedRows"], 1)
            self.assertEqual(result["hybrid6MacroF1"], 0.0)
            self.assertEqual(result["rows"][0]["predictedBindings"], 0)

            run_results = []
            manifests = {}
            for run_number in range(1, 11):
                run_result = dict(result)
                run_result["run"] = run_number
                run_result["runtimeBlockedRows"] = 1 if run_number == 1 else 0
                run_result["status"] = (
                    "runtime_blocked_scored_no_ranking"
                    if run_number == 1 else "completed"
                )
                run_results.append(run_result)
                run_manifest = dict(manifest)
                run_manifest["run"] = run_number
                run_manifest["seed"] = 100 + run_number
                run_manifest["runtimeBlockedRows"] = 1 if run_number == 1 else 0
                run_manifest["status"] = (
                    "runtime_blocked" if run_number == 1 else "completed"
                )
                manifests[("6_llama_z", run_number)] = run_manifest
            aggregate = evaluator.aggregate_case(
                "6_llama_z",
                run_results,
                manifests,
                {
                    "topK": 3,
                    "expectedRuns": 10,
                    "promptBuilder": {
                        "manyShotCount": 26,
                        "dynamicShotCount": 10,
                    },
                    "inputWorkbook": {"sha256": "data-hash"},
                    "protocol": "experiment6-narrative2-full-v2",
                },
            )
        self.assertEqual(
            aggregate["runtime"]["completion_status"],
            "runtime_blocked_scored_no_ranking",
        )
        self.assertEqual(aggregate["runtime"]["runtime_blocked_rows"], 1)
        self.assertEqual(
            aggregate["scores"]["completion_status"],
            "runtime_blocked_1_rows_across_10_runs",
        )

    def test_score_stats_reports_sample_sd(self) -> None:
        stats = evaluator.score_stats([0.0, 1.0])
        self.assertEqual(stats["mean"], 0.5)
        self.assertAlmostEqual(stats["sampleSd"], 2 ** -0.5)


if __name__ == "__main__":
    unittest.main()
