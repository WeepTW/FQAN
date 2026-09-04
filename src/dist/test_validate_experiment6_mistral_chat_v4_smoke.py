#!/usr/bin/env python3
"""Tests for the Mistral v4 three-row smoke gate."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name(
    "validate_experiment6_mistral_chat_v4_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("mistral_smoke_gate", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


BINDING = {
    "ObjectName": ["revenue"],
    "DataName": "Revenue",
    "Position": [{"Begin": [0, 0], "End": [0, 0]}],
    "Trend": "increase",
    "Num": [1],
    "Text": "Revenue increased to 1.",
}


class SmokeGateTests(unittest.TestCase):
    def build_root(self, *, raw: str | None = None, route: str = "direct-binding") -> Path:
        root = Path(tempfile.mkdtemp())
        for subdir, (case_id, source) in MODULE.EXPECTED.items():
            smoke_root = root / subdir
            predictions = root / subdir / "cases" / case_id / "run_01" / "predictions.jsonl"
            predictions.parent.mkdir(parents=True)
            value = raw if raw is not None else json.dumps({"result": [BINDING]})
            prediction = {
                "source": source,
                "formatValid": raw is None,
                "result": [BINDING] if raw is None else [],
                "rawResponse": value,
            }
            predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            material = {
                "protocol": "test",
                "executionMode": f"formal;rows={source}",
                "model": "mistral_v0_3",
            }
            fingerprint = MODULE.sha256_json(material)
            (smoke_root / "compatibility_fingerprint.json").write_text(
                json.dumps({"sha256": fingerprint, "material": material}),
                encoding="utf-8",
            )
            manifest_dir = root / subdir / "manifests"
            manifest_dir.mkdir()
            manifest = {
                "outputId": case_id,
                "run": 1,
                "status": (
                    "completed" if raw is None else "completed_with_format_errors"
                ),
                "runtimeBlockedRows": 0,
                "declaredRoute": route,
                "effectiveRoute": route,
                "adapter": None,
                "converterModel": None,
                "compatibilityFingerprint": fingerprint,
                "files": {"predictions": str(predictions)},
            }
            (manifest_dir / f"{case_id}__run_01.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
        return root

    def test_accepts_strict_nonempty_bindings(self) -> None:
        self.assertEqual(MODULE.validate(self.build_root())["status"], "passed")

    def test_accepts_unique_gold_free_repair(self) -> None:
        raw = "model prefix\n" + json.dumps(BINDING)
        report = MODULE.validate(self.build_root(raw=raw))
        self.assertTrue(all(item["repairAvailable"] for item in report["smokes"]))

    def test_rejects_prompt_echo(self) -> None:
        raw = "## Output examples [EXAMPLE 01]\n[EXAMPLE 02]\n" + json.dumps(BINDING)
        with self.assertRaisesRegex(MODULE.SmokeError, "no strict or uniquely repairable"):
            MODULE.validate(self.build_root(raw=raw))

    def test_rejects_route_mismatch(self) -> None:
        with self.assertRaisesRegex(MODULE.SmokeError, "route mismatch"):
            MODULE.validate(self.build_root(route="adapter-converter"))


if __name__ == "__main__":
    unittest.main()
