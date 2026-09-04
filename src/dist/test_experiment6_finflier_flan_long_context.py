#!/usr/bin/env python3
"""Regression tests for the single-FLAN long-context diagnostic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

import evaluate_experiment6_binding_candidates_v1 as evaluator
import experiment6_finflier_flan_long_context as workflow
import run_experiment6_narrative2_generation as generation


class LongContextContractTests(unittest.TestCase):
    def test_matrix_is_one_no_adapter_flan_case(self) -> None:
        case = workflow.matrix_case()
        self.assertEqual(case.output_id, workflow.CASE_ID)
        self.assertEqual(case.source_id, "flan_t5_large")
        self.assertEqual(case.route, "direct-binding")
        config = generation.load_config(workflow.CONFIG)
        self.assertEqual(config["inputType"], "FinFlier")
        self.assertEqual(config["expectedFormalPredictions"], 850)
        self.assertEqual(config["retriever"]["batchSize"], 1)
        self.assertEqual(config["directBinding"]["maxNewTokens"], 4096)
        self.assertEqual(config["directBinding"]["familyMaxInputTokens"]["flan"], 16896)
        self.assertEqual(config["directBinding"]["familyContextTokens"]["flan"], 20992)

    def test_ft005_row_filter_is_exact(self) -> None:
        config = generation.load_config(workflow.CONFIG)
        rows, report = generation.read_input_rows(config, 0, workflow.LONGEST_SOURCE)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source, workflow.LONGEST_SOURCE)
        self.assertEqual(report["rows"], 1)

    def test_device_override_does_not_change_scientific_arguments(self) -> None:
        gpu = workflow.runner_command(Path("/tmp/root"), device="1", run=3)
        cpu = workflow.runner_command(Path("/tmp/root"), device="cpu", run=3)
        gpu_without_device = gpu[:]
        cpu_without_device = cpu[:]
        index = gpu_without_device.index("--cuda-visible-devices")
        del gpu_without_device[index : index + 2]
        index = cpu_without_device.index("--cuda-visible-devices")
        del cpu_without_device[index : index + 2]
        self.assertEqual(gpu_without_device, cpu_without_device)

    def test_full_token_contract_and_no_truncation(self) -> None:
        report = {
            "tokens": {
                "directBinding": {
                    workflow.CASE_ID: {
                        "family": "flan",
                        "route": "direct-binding",
                        "maxInputAllowed": 16896,
                        "contextWindow": 20992,
                        "maxNewTokens": 4096,
                        "maxObserved": 16574,
                        "minObserved": 9675,
                        "maxPromptPlusCompletion": 20670,
                        "measurements": 85,
                        "truncationAllowed": False,
                        "structuredOutput": "off",
                        "adapter": None,
                        "converter": None,
                    }
                }
            }
        }
        workflow.validate_token_report(report, measurements=85)
        report["tokens"]["directBinding"][workflow.CASE_ID]["truncationAllowed"] = True
        with self.assertRaises(workflow.OrchestrationError):
            workflow.validate_token_report(report, measurements=85)

    def test_row_execution_manifest_partitions_gpu_and_cpu_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = workflow.run_dir(root, 1)
            run_root.mkdir(parents=True)
            prompts = [
                {
                    "index": index,
                    "source": f"s{index}",
                    "directPromptSha256": str(index) * 64,
                }
                for index in range(3)
            ]
            (run_root / "prompts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in prompts), encoding="utf-8"
            )
            attempts = [
                {
                    "device": "gpu",
                    "checkpointIndicesAfter": [0],
                    "oom": True,
                },
                {
                    "device": "cpu",
                    "checkpointIndicesAfter": [0, 1, 2],
                    "oom": False,
                },
            ]
            path = workflow.write_execution_manifest(root, 1, attempts, expected_rows=3)
            rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
            self.assertEqual([row["device"] for row in rows], ["gpu", "cpu", "cpu"])
            self.assertEqual([row["directPromptSha256"] for row in rows], [str(i) * 64 for i in range(3)])

    def test_evaluator_accepts_long_context_scope(self) -> None:
        args = evaluator.parse_args(
            [
                "--version",
                "v6.1.0",
                "--scope",
                "flan-long-context",
                "--candidate-root",
                "/tmp/candidates",
                "--evaluation-root",
                "/tmp/evaluation",
            ]
        )
        self.assertEqual(args.scope, "flan-long-context")

    def test_no_adapter_flan_runner_forces_generate(self) -> None:
        source = (workflow.PATHS.dist / "run_experiment6_binding_generation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('if not use_adapter:\n            command.extend(["--infer-method", "generate"])', source)


if __name__ == "__main__":
    unittest.main()
