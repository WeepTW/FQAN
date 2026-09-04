#!/usr/bin/env python3
"""Regression tests for the Mistral v5 completion audit."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name(
    "audit_experiment6_mistral_base_md_v5_completion.py"
)
SPEC = importlib.util.spec_from_file_location("mistral_completion_audit", MODULE_PATH)
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


class CompletionAuditTests(unittest.TestCase):
    def write_json(self, path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def build(self) -> tuple[Path, Path, Path]:
        root = Path(tempfile.mkdtemp())
        generation = root / "generation"
        manifests = generation / "manifests"
        manifests.mkdir(parents=True)
        config_path = Path(__file__).resolve().parents[1] / "config" / "experiment6_narrative2_generation_mistral_base_md_chat_template_v5.json"
        snapshot_path = generation / "generation_config.snapshot.json"
        snapshot = json.loads(config_path.read_text(encoding="utf-8"))
        self.write_json(snapshot_path, snapshot)
        self.write_json(
            generation / "compatibility_fingerprint.json",
            {
                "sha256": "f" * 64,
                "material": {
                    "generationConfigSha256": MODULE.sha256_file(snapshot_path),
                    "promptPolicyVersion": MODULE.PROMPT_POLICY_VERSION,
                },
            },
        )
        for case in sorted(MODULE.CASES):
            for run in sorted(MODULE.RUNS):
                run_dir = generation / "cases" / case / f"run_{run:02d}"
                run_dir.mkdir(parents=True)
                predictions = run_dir / "predictions.jsonl"
                predictions.write_text(
                    "".join(
                        json.dumps(
                            {
                                "source": f"Source_{index:03d}",
                                "promptSha256": MODULE.sha256_text(f"Prompt for {case} Source_{index:03d}"),
                                "formatValid": True,
                                "result": [BINDING],
                                "rawResponse": json.dumps({"result": [BINDING]}),
                            }
                        ) + "\n"
                        for index in range(85)
                    ),
                    encoding="utf-8",
                )
                artifacts = {
                    "predictions": predictions,
                    "rawResponse": run_dir / "raw_response.jsonl",
                    "prompts": run_dir / "prompts.jsonl",
                    "runtime": run_dir / "runtime.json",
                    "formatReport": run_dir / "format_report.json",
                    "nonformalRepair": run_dir / "repair_predictions.nonformal.jsonl",
                    "stage1Raw": run_dir / "stage1.jsonl",
                }
                for name in ("rawResponse", "formatReport"):
                    self.write_json(artifacts[name], {})
                prompt_rows = []
                for index in range(85):
                    prompt = f"Prompt for {case} Source_{index:03d}"
                    prompt_rows.append({
                        "source": f"Source_{index:03d}",
                        "directPrompt": prompt,
                        "directPromptSha256": MODULE.sha256_text(prompt),
                    })
                artifacts["prompts"].write_text(
                    "".join(json.dumps(row) + "\n" for row in prompt_rows),
                    encoding="utf-8",
                )
                artifacts["nonformalRepair"].write_text("", encoding="utf-8")
                artifacts["stage1Raw"].write_text("raw\n", encoding="utf-8")
                device = "0" if case.endswith("_m") else "1"
                self.write_json(
                    artifacts["runtime"],
                    {
                        "stages": [
                            {
                                "raw": {
                                    "execution_device": "cuda",
                                    "use_adapter": False,
                                    "converter_used": False,
                                    "generation_cache_used": False,
                                    "chat_template_applied": True,
                                    "structured_output": "off",
                                    "chat_template_sha256": MODULE.CHAT_TEMPLATE_SHA256,
                                    "max_input_tokens": 8192,
                                    "context_window": 12288,
                                    "batch_size_effective": 6,
                                    "cuda_visible_devices": device,
                                }
                            }
                        ]
                    },
                )
                manifest = {
                    "outputId": case,
                    "run": run,
                    "seed": MODULE.SEEDS[run],
                    "status": "completed_with_format_rejections",
                    "expectedRows": 85,
                    "acceptedRows": 80,
                    "rejectedRows": 5,
                    "runtimeBlockedRows": 0,
                    "declaredRoute": "direct-binding",
                    "effectiveRoute": "direct-binding",
                    "adapter": None,
                    "converterModel": None,
                    "actualModel": MODULE.MODEL,
                    "promptMode": MODULE.PROMPT_MODES[case],
                    "compatibilityFingerprint": "f" * 64,
                    "files": {name: str(path) for name, path in artifacts.items()},
                    "hashes": {name: MODULE.sha256_file(path) for name, path in artifacts.items()},
                }
                self.write_json(manifests / f"{case}__run_{run:02d}.json", manifest)

        binding = root / "binding"
        self.write_json(
            binding / "dataset_manifest.json",
            {
                "protocol": "experiment6-mistral-chat-repaired-projection-v1",
                "status": "complete",
                "official": False,
                "diagnosticOnly": True,
                "claimEligible": False,
                "goldAccessed": False,
                "counts": {"cases": 2, "caseRuns": 20, "rows": 1700},
            },
        )
        (binding / "sha256_inventory.tsv").write_text("path\tsize\tsha256\n", encoding="utf-8")
        judge_root = binding / "judge_examples"
        judge_specs = {}
        for name in ("canonical_examples.jsonl", "repair_manifest.jsonl", "judge_prompt_prefix_ObjectName.txt", "judge_prompt_prefix_Trend.txt", "judge_prompt_prefix_Text.txt"):
            path = judge_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + "\n", encoding="utf-8")
            judge_specs[name] = {"path": str(path), "sha256": MODULE.sha256_file(path)}
        self.write_json(
            judge_root / "manifest.json",
            {
                "validation": {"status": "passed", "canonicalRows": 26, "canonicalBindings": 55},
                "files": {
                    "canonicalExamples": judge_specs["canonical_examples.jsonl"],
                    "repairManifest": judge_specs["repair_manifest.jsonl"],
                    "promptPrefixes": {
                        "ObjectName": judge_specs["judge_prompt_prefix_ObjectName.txt"],
                        "Trend": judge_specs["judge_prompt_prefix_Trend.txt"],
                        "Text": judge_specs["judge_prompt_prefix_Text.txt"],
                    },
                },
            },
        )
        final = root / "evaluation"
        fields = {
            field: {"precision": 0.1, "recall": 0.2, "f1": 0.13}
            for field in MODULE.FIELDS
        }
        v610_path = root / "components" / "v610" / "evaluation_report.json"
        self.write_json(
            v610_path,
            {
                "protocol": "experiment6-binding-candidate-evaluation-v1",
                "scoringProtocol": "experiment6-reference-aligned-v6.1.0",
                "scope": "mistral-base-md",
                "diagnosticOnly": True,
                "claimEligible": False,
                "candidateValidation": {
                    "status": "passed",
                    "cases": 2,
                    "caseRuns": 20,
                    "rows": 1700,
                    "datasetManifestSha256": MODULE.sha256_file(binding / "dataset_manifest.json"),
                    "inventorySha256": MODULE.sha256_file(binding / "sha256_inventory.tsv"),
                },
            },
        )
        semantic_path = root / "components" / "semantic" / "evaluation_report.json"
        self.write_json(
            semantic_path,
            {
                "protocol": "narrative2-reference-aligned-hybrid-v5.1",
                "status": "completed",
                "completedCases": 2,
                "completedCaseRuns": 20,
                "formalPredictions": 1700,
                "judge": {
                    "model": "gpt-5.5",
                    "reasoningEffort": "medium",
                    "minimumConfidence": 0.8,
                    "disabled": False,
                },
            },
        )
        self.write_json(
            final / "evaluation_report.json",
            {
                "protocol": "experiment6-reference-aligned-v6.1-with-semantic-text-v1",
                "status": "completed",
                "inputs": {
                    "v610Report": str(v610_path),
                    "v610ReportSha256": MODULE.sha256_file(v610_path),
                    "semanticTextReport": str(semantic_path),
                    "semanticTextReportSha256": MODULE.sha256_file(semantic_path),
                },
                "judge": {
                    "model": "gpt-5.5",
                    "reasoningEffort": "medium",
                    "minimumConfidence": 0.8,
                    "disabled": False,
                },
                "cases": [
                    {"outputId": case, "runs": 10, "fields": fields, "macro": {"precision": 0.1, "recall": 0.2, "f1": 0.13}}
                    for case in sorted(MODULE.CASES)
                ],
            },
        )
        (final / "evaluation_report.md").write_text("report\n", encoding="utf-8")
        (final / "experiment_6_v6_欄位分數_mean.md").write_text("mean\n", encoding="utf-8")
        events = generation / "scheduler" / "finalizer_events.jsonl"
        events.parent.mkdir()
        events.write_text(
            json.dumps({"event": "binding_materialization_complete", "detail": str(binding)}) + "\n"
            + json.dumps({"event": "finalizer_complete", "detail": str(final)}) + "\n",
            encoding="utf-8",
        )
        return generation, events, generation / "scheduler" / "completion_audit.json"

    def test_accepts_complete_two_case_contract(self) -> None:
        generation, events, output = self.build()
        report = MODULE.audit(generation, events, output)
        self.assertEqual(report["coverage"]["rows"], 1700)
        self.assertEqual(report["promptEchoRows"], 0)
        self.assertEqual(report["promptIdentity"]["caseSources"], 170)
        self.assertTrue(report["promptIdentity"]["stable"])
        self.assertTrue(output.with_name("sha256_inventory.tsv").is_file())

    def test_rejects_prompt_hash_mismatch(self) -> None:
        generation, events, output = self.build()
        manifest_path = next((generation / "manifests").glob("*.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompts_path = Path(manifest["files"]["prompts"])
        prompts = MODULE.read_jsonl(prompts_path)
        prompts[0]["directPrompt"] += " changed"
        prompts_path.write_text(
            "".join(json.dumps(row) + "\n" for row in prompts), encoding="utf-8"
        )
        manifest["hashes"]["prompts"] = MODULE.sha256_file(prompts_path)
        self.write_json(manifest_path, manifest)
        with self.assertRaisesRegex(MODULE.AuditError, "prompt hash mismatch"):
            MODULE.audit(generation, events, output)

    def test_rejects_gold_marker_in_generation_prompt(self) -> None:
        generation, events, output = self.build()
        manifest_path = next((generation / "manifests").glob("*.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompts_path = Path(manifest["files"]["prompts"])
        prompts = MODULE.read_jsonl(prompts_path)
        prompts[0]["directPrompt"] += ' "gold_targets"'
        prompts[0]["directPromptSha256"] = MODULE.sha256_text(prompts[0]["directPrompt"])
        prompts_path.write_text(
            "".join(json.dumps(row) + "\n" for row in prompts), encoding="utf-8"
        )
        manifest["hashes"]["prompts"] = MODULE.sha256_file(prompts_path)
        self.write_json(manifest_path, manifest)
        with self.assertRaisesRegex(MODULE.AuditError, "gold marker"):
            MODULE.audit(generation, events, output)

    def test_rejects_component_hash_mismatch(self) -> None:
        generation, events, output = self.build()
        final_root = Path(MODULE.event_details(events)["finalizer_complete"])
        final_path = final_root / "evaluation_report.json"
        report = json.loads(final_path.read_text(encoding="utf-8"))
        report["inputs"]["v610ReportSha256"] = "0" * 64
        self.write_json(final_path, report)
        with self.assertRaisesRegex(MODULE.AuditError, "component provenance"):
            MODULE.audit(generation, events, output)

    def test_rejects_disabled_judge(self) -> None:
        generation, events, output = self.build()
        final_root = Path(MODULE.event_details(events)["finalizer_complete"])
        final_path = final_root / "evaluation_report.json"
        report = json.loads(final_path.read_text(encoding="utf-8"))
        report["judge"]["disabled"] = True
        self.write_json(final_path, report)
        with self.assertRaisesRegex(MODULE.AuditError, "judge identity"):
            MODULE.audit(generation, events, output)

    def test_rejects_route_mismatch(self) -> None:
        generation, events, output = self.build()
        path = next((generation / "manifests").glob("*.json"))
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["effectiveRoute"] = "adapter-converter"
        self.write_json(path, manifest)
        with self.assertRaisesRegex(MODULE.AuditError, "route mismatch"):
            MODULE.audit(generation, events, output)


    def test_rejects_v5_prompt_policy_drift(self) -> None:
        generation, events, output = self.build()
        snapshot_path = generation / "generation_config.snapshot.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot["mistralDirectChatTemplate"]["policy"] = "wrong-policy"
        self.write_json(snapshot_path, snapshot)
        compatibility_path = generation / "compatibility_fingerprint.json"
        compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
        compatibility["material"]["generationConfigSha256"] = MODULE.sha256_file(snapshot_path)
        self.write_json(compatibility_path, compatibility)
        with self.assertRaisesRegex(MODULE.AuditError, "frozen prompt policy"):
            MODULE.audit(generation, events, output)


if __name__ == "__main__":
    unittest.main()
