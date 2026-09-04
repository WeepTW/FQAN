#!/usr/bin/env python3
"""Regression tests for unified Experiment 6 Binding materialization v2."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

import evaluate_experiment6_binding_candidates_v1 as evaluator
from materialize_experiment6_binding_candidates import sha256_file
from materialize_experiment6_bindings_v2 import (
    PROTOCOL,
    materialize_run,
    structural_payload_result,
)


def binding(**updates):
    value = {
        "ObjectName": ["Revenue"],
        "DataName": "Revenue",
        "Position": [{"Begin": [1, 2], "End": [1, 2]}],
        "Trend": "None",
        "Num": [12.0],
        "Text": "Revenue was 12.",
    }
    value.update(updates)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class UnifiedBindingV2Tests(unittest.TestCase):
    def test_safe_structural_repairs_are_narrow(self) -> None:
        result, operations, reason = structural_payload_result({"result": [binding()]})
        self.assertEqual(reason, "valid")
        self.assertEqual(result, [binding()])
        self.assertEqual(operations, ["add-missing-reason-wrapper"])

        result, operations, reason = structural_payload_result(binding())
        self.assertEqual(reason, "valid")
        self.assertEqual(result, [binding()])
        self.assertEqual(operations, ["wrap-single-binding-result-array"])

        scalar = binding(ObjectName="Revenue")
        result, operations, reason = structural_payload_result({"Binding": [scalar]})
        self.assertEqual(reason, "valid")
        self.assertEqual(result, [binding()])
        self.assertEqual(operations, ["wrap-objectname-singleton-array"])

        result, operations, reason = structural_payload_result(
            {"result": [binding(Num=["12m"])], "reason": ""}
        )
        self.assertIsNone(result)
        self.assertEqual(reason, "Num_not_finite_number_array")
        self.assertEqual(operations, [])

        result, _, reason = structural_payload_result(
            {"result": [binding()], "reason": "", "extra": "not allowed"}
        )
        self.assertIsNone(result)
        self.assertEqual(reason, "top_level_contract")

    def test_run_materialization_keeps_every_row_and_source_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generation = root / "generation"
            run_dir = generation / "cases" / "case_a" / "run_01"
            predictions = [
                {"index": 0, "source": "S0", "result": [binding()], "formatValid": True, "rawResponse": "strict"},
                {"index": 1, "source": "S1", "result": [], "formatValid": False, "rawResponse": "missing reason"},
                {"index": 2, "source": "S2", "result": [], "formatValid": False, "rawResponse": "numeric unit"},
            ]
            repairs = [
                {"index": 1, "source": "S1", "official": False, "excludedFromScores": True, "repair": {"available": True, "method": "balanced-json", "payload": {"result": [binding()]}}},
                {"index": 2, "source": "S2", "official": False, "excludedFromScores": True, "repair": {"available": True, "method": "balanced-json", "payload": {"result": [binding(Num=["12m"])], "reason": ""}}},
            ]
            prediction_path = run_dir / "predictions.jsonl"
            repair_path = run_dir / "repair_predictions.nonformal.jsonl"
            write_jsonl(prediction_path, predictions)
            write_jsonl(repair_path, repairs)
            manifest = {
                "outputId": "case_a",
                "run": 1,
                "seed": 7,
                "official": True,
                "protocol": "source-protocol",
                "compatibilityFingerprint": None,
                "route": "direct-binding",
                "declaredRoute": "direct-binding",
                "effectiveRoute": "direct-binding",
                "files": {"predictions": str(prediction_path), "nonformalRepair": str(repair_path)},
                "hashes": {"predictions": sha256_file(prediction_path), "nonformalRepair": sha256_file(repair_path)},
            }
            manifest_path = generation / "manifests" / "case_a__run_01.json"
            write_json(manifest_path, manifest)
            source_before = prediction_path.read_bytes()
            stage = root / "stage"
            stage.mkdir()
            final = root / "final"
            run_manifest, rows, bindings, rejected, _ = materialize_run(
                entry={
                    "sourceGroup": "historical",
                    "root": generation,
                    "manifest": manifest,
                    "manifestPath": manifest_path,
                    "requireRepairCoverage": False,
                },
                staging_root=stage,
                final_root=final,
                fingerprint="f" * 64,
                expected_rows=3,
            )
            self.assertEqual(prediction_path.read_bytes(), source_before)
            self.assertEqual(len(rows), 3)
            self.assertEqual(run_manifest["acceptedRows"], 2)
            self.assertEqual(run_manifest["rejectedRows"], 1)
            self.assertEqual(len(bindings), 2)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rows[1]["candidateStatus"], "safe_structural_repair_valid")
            self.assertEqual(rows[2]["rejectionReason"], "Num_not_finite_number_array")
            self.assertFalse(rows[2]["formatValid"])

    def test_evaluator_accepts_v2_only_when_explicit_and_candidate34_scope(self) -> None:
        manifest = {
            "protocol": PROTOCOL,
            "official": False,
            "diagnosticOnly": True,
            "claimEligible": False,
            "goldAccessed": False,
            "status": "completed_diagnostic_binding_candidates",
        }
        evaluator.validate_candidate_manifest(manifest, accepted_protocol=PROTOCOL)
        with self.assertRaises(evaluator.CandidateEvaluationError):
            evaluator.validate_candidate_manifest(manifest)
        args = evaluator.parse_args(
            [
                "--version", "v6.0.2",
                "--scope", "candidate34",
                "--candidate-root", "/tmp/candidate",
                "--evaluation-root", "/tmp/evaluation",
            ]
        )
        self.assertEqual(args.scope, "candidate34")
        self.assertIsNone(args.formal_reference)


if __name__ == "__main__":
    unittest.main()
