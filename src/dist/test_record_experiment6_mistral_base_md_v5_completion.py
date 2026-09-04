#!/usr/bin/env python3
"""Tests for audited Mistral v5 docs/log recording."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("record_experiment6_mistral_base_md_v5_completion.py")
SPEC = importlib.util.spec_from_file_location("mistral_completion_record", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompletionRecordTests(unittest.TestCase):
    def fixture(self, *, status: str = "complete"):
        workspace = Path(tempfile.mkdtemp())
        audit_path = workspace / "src/Experiment/run/scheduler/completion_audit.json"
        audit_path.parent.mkdir(parents=True)
        fields = {
            field: {"precision": 0.1, "recall": 0.2, "f1": 0.13}
            for field in MODULE.FIELDS
        }
        audit = {
            "status": status,
            "protocol": "experiment6-mistral-base-md-v5-completion-audit-v1",
            "generationRoot": str(workspace / "src/Experiment/generation"),
            "bindingRoot": str(workspace / "src/Experiment/binding"),
            "evaluationRoot": str(workspace / "src/Experiment/evaluation"),
            "compatibilityFingerprint": "f" * 64,
            "coverage": {"cases": 2, "caseRuns": 20, "rows": 1700},
            "promptIdentity": {
                "caseSources": 170,
                "runsPerCaseSource": 10,
                "stable": True,
                "goldMarkers": 0,
            },
            "judge": {
                "model": "gpt-5.5",
                "reasoningEffort": "medium",
                "minimumConfidence": 0.8,
            },
            "scores": {
                case: {"fields": fields, "macro": {"precision": 0.1, "recall": 0.2, "f1": 0.13}}
                for case in ("6_mistral_base_m", "6_mistral_base_d")
            },
        }
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        log_root = workspace / "docs/log"
        log_root.mkdir(parents=True)
        index = log_root / "index.json"
        index.write_text('{"entries": []}\n', encoding="utf-8")
        return workspace, audit_path, log_root, index

    def test_writes_six_field_report_and_atomic_index_entry(self) -> None:
        workspace, audit, log_root, index = self.fixture()
        result = MODULE.record(audit, log_root, index, workspace)
        report = Path(result["report"]).read_text(encoding="utf-8")
        self.assertIn("6_mistral_base_m", report)
        self.assertIn("ObjectName", report)
        for field in MODULE.FIELDS:
            self.assertIn(field, report)
        for metric in ("Precision", "Recall", "F1"):
            self.assertIn(metric, report)
        self.assertIn("Prompt identity: 170", report)
        self.assertIn("enabled evidence-gated", report)
        self.assertIn("$FQAN_ROOT/src/Experiment/generation", report)
        entry = json.loads(index.read_text(encoding="utf-8"))["entries"][0]
        self.assertEqual(entry["kind"], MODULE.KIND)
        self.assertEqual(entry["sha256"], MODULE.sha256_file(Path(result["report"])))

    def test_rejects_disabled_or_wrong_judge_audit(self) -> None:
        workspace, audit, log_root, index = self.fixture()
        payload = json.loads(audit.read_text(encoding="utf-8"))
        payload["judge"]["model"] = "disabled"
        audit.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.RecordError, "judge mismatch"):
            MODULE.record(audit, log_root, index, workspace)

    def test_rejects_incomplete_audit(self) -> None:
        workspace, audit, log_root, index = self.fixture(status="blocked")
        with self.assertRaisesRegex(MODULE.RecordError, "not complete"):
            MODULE.record(audit, log_root, index, workspace)


if __name__ == "__main__":
    unittest.main()
