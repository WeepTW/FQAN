#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

from build_experiment6_evaluation_overlay import OverlayError, build_overlay


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(root: Path, output_id: str, run: int, marker: str) -> None:
    run_root = root / "cases" / output_id / f"run_{run:02d}"
    run_root.mkdir(parents=True, exist_ok=True)
    prediction = run_root / "predictions.jsonl"
    prediction.write_text(
        json.dumps({"source": marker, "result": [], "formatValid": False}) + "\n",
        encoding="utf-8",
    )
    manifest_root = root / "manifests"
    manifest_root.mkdir(exist_ok=True)
    manifest = {
        "official": True,
        "outputId": output_id,
        "run": run,
        "status": "completed_with_format_errors",
        "finishedAt": "2026-08-14T00:00:00Z",
        "files": {"predictions": str(prediction)},
        "hashes": {"predictions": sha256(prediction)},
    }
    (manifest_root / f"{output_id}__run_{run:02d}.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


class OverlayTests(unittest.TestCase):
    def test_override_replaces_complete_case_without_copying_predictions(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            base = root / "base"
            override = root / "override"
            for case in ("a", "b"):
                make_run(base, case, 1, f"base-{case}")
            make_run(override, "a", 1, "override-a")
            output = root / "overlay"
            result = build_overlay(
                base,
                override,
                output,
                expected_cases=2,
                expected_override_cases=1,
                expected_runs=1,
                expected_rows=1,
            )
            self.assertEqual(result["cases"], 2)
            self.assertTrue((output / "cases" / "a").is_symlink())
            self.assertTrue((output / "cases" / "b").is_symlink())
            report = json.loads((output / "generation_report.json").read_text())
            self.assertEqual(report["caseOrigins"], {"a": "override", "b": "base"})
            linked = json.loads(
                (output / "cases" / "a" / "run_01" / "predictions.jsonl").read_text()
            )
            self.assertEqual(linked["source"], "override-a")

    def test_rejects_override_case_absent_from_base(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            base = root / "base"
            override = root / "override"
            make_run(base, "a", 1, "base")
            make_run(override, "x", 1, "override")
            with self.assertRaises(OverlayError):
                build_overlay(
                    base,
                    override,
                    root / "overlay",
                    expected_cases=1,
                    expected_override_cases=1,
                    expected_runs=1,
                    expected_rows=1,
                )


if __name__ == "__main__":
    unittest.main()
