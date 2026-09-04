#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

from materialize_experiment6_binding_candidates import (
    PROTOCOL,
    build,
    read_json,
    read_jsonl,
    sha256_file,
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
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class BindingCandidateMaterializationTests(unittest.TestCase):
    def test_materializes_valid_binding_and_preserves_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generation = root / "generation"
            run_dir = generation / "cases" / "case_a" / "run_01"
            predictions = [
                {"index": index, "source": f"S{index}", "result": [], "formatValid": False, "rawResponse": "raw"}
                for index in range(3)
            ]
            repairs = [
                {
                    "index": 0,
                    "source": "S0",
                    "official": False,
                    "excludedFromScores": True,
                    "repair": {"available": True, "method": "balanced-json", "payload": {"result": [binding()], "reason": "ok"}},
                },
                {
                    "index": 1,
                    "source": "S1",
                    "official": False,
                    "excludedFromScores": True,
                    "repair": {"available": True, "method": "balanced-json", "payload": {"result": [binding(Num=[["12"]])], "reason": ""}},
                },
                {
                    "index": 2,
                    "source": "S2",
                    "official": False,
                    "excludedFromScores": True,
                    "repair": {"available": False, "method": "balanced-json", "payload": None},
                },
            ]
            prediction_path = run_dir / "predictions.jsonl"
            repair_path = run_dir / "repair_predictions.nonformal.jsonl"
            write_jsonl(prediction_path, predictions)
            write_jsonl(repair_path, repairs)
            source_fingerprint = "a" * 64
            manifest = {
                "outputId": "case_a",
                "run": 1,
                "seed": 7,
                "official": True,
                "protocol": "source-protocol",
                "compatibilityFingerprint": source_fingerprint,
                "route": "direct-binding",
                "declaredRoute": "direct-binding",
                "effectiveRoute": "direct-binding",
                "files": {"predictions": str(prediction_path), "nonformalRepair": str(repair_path)},
                "hashes": {"predictions": sha256_file(prediction_path), "nonformalRepair": sha256_file(repair_path)},
            }
            manifest_path = generation / "manifests" / "case_a__run_01.json"
            write_json(manifest_path, manifest)
            config = {
                "schemaVersion": 1,
                "protocol": PROTOCOL,
                "sourceProtocol": "source-protocol",
                "sourceCompatibilityFingerprint": source_fingerprint,
                "expectedCases": 1,
                "expectedRuns": 1,
                "expectedRows": 3,
                "caseIds": ["case_a"],
                "requiredBindingKeys": ["ObjectName", "DataName", "Position", "Trend", "Num", "Text"],
                "requireRepairCoverage": True,
            }
            config_path = root / "config.json"
            output = root / "binding_output"
            write_json(config_path, config)
            source_before = prediction_path.read_bytes()
            report = build(argparse.Namespace(generation_root=generation, output_root=output, config=config_path))

            self.assertEqual(report["counts"]["rows"], 3)
            self.assertEqual(report["counts"]["acceptedRows"], 1)
            self.assertEqual(report["counts"]["rejectedRows"], 2)
            self.assertEqual(report["counts"]["bindings"], 1)
            self.assertFalse(report["official"])
            self.assertFalse(report["goldAccessed"])
            self.assertEqual(prediction_path.read_bytes(), source_before)
            bindings = read_jsonl(output / "bindings.jsonl")
            self.assertEqual(bindings[0]["DataName"], "Revenue")
            self.assertEqual(len(bindings[0]["candidateId"]), 64)
            rows = read_jsonl(output / "rows.jsonl")
            self.assertEqual(rows[0]["Binding"], [binding()])
            self.assertEqual(rows[1]["rejectionReason"], "Num_not_finite_number_array")
            self.assertEqual(rows[2]["candidateStatus"], "repair_unavailable")
            run_manifest = read_json(output / "manifests" / "case_a__run_01.json")
            self.assertFalse(run_manifest["official"])
            self.assertEqual(run_manifest["acceptedRows"], 1)
            evaluator_rows = read_jsonl(Path(run_manifest["files"]["predictions"]))
            self.assertTrue(evaluator_rows[0]["formatValid"])
            self.assertFalse(evaluator_rows[1]["formatValid"])


if __name__ == "__main__":
    unittest.main()
