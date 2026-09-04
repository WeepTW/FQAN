#!/usr/bin/env python3
"""Route-first regressions for corrected Experiment 6 execution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate_narrative2_reference_aligned_v5 as evaluator_v5
import experiment6_corrected12 as orchestrator
import record_experiment6_corrected12 as recorder
import run_experiment6_narrative2_generation as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
CORRECTED_CONFIG = REPO_ROOT / "config" / "experiment6_narrative2_generation_corrected_12.json"
FULL_CONFIG = REPO_ROOT / "config" / "experiment6_narrative2_generation.json"



def minimal_row() -> runner.InputRow:
    prompts = {
        mode: f"prompt {mode}"
        for mode in ("original", "zero-shot", "many-shot", "dynamic-shot")
    }
    return runner.InputRow(
        index=0,
        number="1",
        source="S1",
        data_raw='[{"Revenue":1}]',
        data_compact="__row__,Revenue\n0,1",
        text="Revenue increased.",
        retriever_prompts=prompts,
        direct_prompts=prompts,
        shot_ids={
            "original": (),
            "zero-shot": (),
            "many-shot": tuple(range(26)),
            "dynamic-shot": tuple(range(10)),
        },
    )


def valid_raw() -> str:
    return json.dumps(
        {
            "result": [
                {
                    "ObjectName": ["Revenue"],
                    "DataName": "Revenue",
                    "Position": [{"Begin": [0, 1], "End": [0, 1]}],
                    "Trend": "increase",
                    "Num": [185],
                    "Text": "Revenue increased to 185.",
                }
            ],
            "reason": "test",
        }
    )


class RouteFirstTests(unittest.TestCase):
    def test_corrected_matrix_is_exactly_the_twelve_invalidated_cases(self) -> None:
        config = runner.load_config(CORRECTED_CONFIG)
        cases = runner.expand_matrix(config)
        self.assertEqual(
            {case.output_id for case in cases},
            {
                "6_flan_base_z",
                "6_flan_base_m",
                "6_flan_base_d",
                "6_mistral_base_z",
                "6_mistral_base_m",
                "6_mistral_base_d",
                "6_t5gemma2_base_z",
                "6_t5gemma2_base_m",
                "6_t5gemma2_base_d",
                "6_FinFlier_flan_base",
                "6_FinFlier_mistral_base",
                "6_FinFlier_t5gemma2_base",
            },
        )
        self.assertTrue(all(case.route == "direct-binding" for case in cases))
        self.assertTrue(
            all(runner.effective_route(case, "formal") == case.route for case in cases)
        )

    def test_formal_base_direct_bypasses_adapter_and_converter(self) -> None:
        config = runner.load_config(CORRECTED_CONFIG)
        case = next(
            case
            for case in runner.expand_matrix(config)
            if case.output_id == "6_mistral_base_d"
        )
        captured: dict[str, object] = {}

        def fake_native(
            legacy_case,
            csv_path,
            prompt_mode,
            args,
            *,
            use_adapter,
            family_override,
            raw_suffix,
        ):
            del legacy_case, csv_path, prompt_mode
            captured.update(
                {
                    "useAdapter": use_adapter,
                    "familyOverride": family_override,
                    "rawSuffix": raw_suffix,
                    "structuredOutput": args.structured_output,
                }
            )
            return [valid_raw()], {
                "family": "mistral",
                "actual_engine": case.source_id,
                "adapter_dir": None,
                "use_adapter": False,
                "structured_output": args.structured_output,
            }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                runner.legacy, "run_retriever_case", side_effect=fake_native
            ), patch.object(
                runner,
                "run_converter",
                side_effect=AssertionError("formal base direct must not use converter"),
            ):
                manifest = runner.run_case_once(
                    case=case,
                    rows=[minimal_row()],
                    run_number=1,
                    run_seed=2026073101,
                    output_root=Path(directory),
                    config=config,
                    resume=False,
                    base_route_mode="formal",
                )
        self.assertEqual(manifest["declaredRoute"], "direct-binding")
        self.assertEqual(manifest["effectiveRoute"], "direct-binding")
        self.assertIsNone(manifest["adapter"])
        self.assertIsNone(manifest["converterModel"])
        self.assertFalse(captured["useAdapter"])
        self.assertEqual(captured["familyOverride"], "mistral")
        self.assertEqual(captured["rawSuffix"], ".direct_binding")
        self.assertEqual(captured["structuredOutput"], "off")

    def test_opaque_direct_source_never_uses_source_id_family_or_adapter_lookup(self) -> None:
        config = runner.load_config(CORRECTED_CONFIG)
        case = runner.legacy.MatrixCase(
            "opaque_case", "opaque-registry-id", "narrative_zero_shot"
        )
        namespace = runner.native_direct_namespace(
            Path("/tmp"), config, 1, "flan"
        )
        completed = SimpleNamespace(returncode=0)
        with patch.object(
            runner.legacy,
            "family_from_source_id",
            side_effect=AssertionError("source_id family fallback forbidden"),
        ), patch.object(
            runner.legacy,
            "adapter_dir_for",
            side_effect=AssertionError("adapter lookup forbidden"),
        ), patch.object(
            runner.legacy, "run_command", return_value=completed
        ) as run_command, patch.object(
            runner.legacy, "read_raw_prediction_lines", return_value=[valid_raw()]
        ):
            predictions, runtime = runner.legacy.run_retriever_case(
                case,
                Path("/tmp/opaque.csv"),
                "zero-shot",
                namespace,
                use_adapter=False,
                family_override="flan",
            )
        self.assertEqual(len(predictions), 1)
        self.assertEqual(runtime["family"], "flan")
        self.assertIsNone(runtime["adapter_dir"])
        command = run_command.call_args.args[0]
        self.assertEqual(command[command.index("--infer-method") + 1], "generate")
        self.assertIn("--no-adapter", command)

    def test_adapter_route_receives_exact_registry_path(self) -> None:
        config = runner.load_config(FULL_CONFIG)
        case = next(
            case
            for case in runner.expand_matrix(config)
            if case.output_id == "6_flan_z"
        )
        captured: dict[str, object] = {}

        def fake_retriever(
            legacy_case,
            csv_path,
            prompt_mode,
            args,
            *,
            use_adapter,
            family_override,
            adapter_dir_override,
        ):
            del legacy_case, csv_path, prompt_mode, args
            captured.update(
                {
                    "useAdapter": use_adapter,
                    "family": family_override,
                    "adapter": adapter_dir_override,
                }
            )
            return ["Revenue increased to 185."], {
                "family": family_override,
                "actual_engine": str(adapter_dir_override),
                "adapter_dir": str(adapter_dir_override),
                "use_adapter": True,
                "structured_output": "canonical",
            }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                runner.legacy, "run_retriever_case", side_effect=fake_retriever
            ), patch.object(
                runner,
                "run_converter",
                return_value=(
                    [valid_raw()],
                    {
                        "stage": "converter",
                        "requestedModel": "gpt-5.5",
                        "actualModel": "gpt-5.5",
                        "reasoningEffort": "medium",
                    },
                ),
            ):
                manifest = runner.run_case_once(
                    case=case,
                    rows=[minimal_row()],
                    run_number=1,
                    run_seed=2026073101,
                    output_root=Path(directory),
                    config=config,
                    resume=False,
                    base_route_mode="formal",
                )
        expected = REPO_ROOT / "Experiment" / "finqa_flan_z" / "retriever" / "model"
        self.assertTrue(captured["useAdapter"])
        self.assertEqual(captured["family"], "flan")
        self.assertEqual(captured["adapter"], expected)
        self.assertEqual(manifest["effectiveRoute"], "adapter-converter")
        self.assertEqual(manifest["converterModel"], "gpt-5.5")

    def test_resume_rejects_different_fingerprint(self) -> None:
        config = runner.load_config(CORRECTED_CONFIG)
        case = runner.expand_matrix(config)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifests" / f"{case.output_id}__run_01.json"
            runner.write_json(
                manifest_path,
                {"status": "completed", "compatibilityFingerprint": "old"},
            )
            with self.assertRaises(runner.ProtocolError):
                runner.run_case_once(
                    case=case,
                    rows=[minimal_row()],
                    run_number=1,
                    run_seed=2026073101,
                    output_root=root,
                    config=config,
                    resume=True,
                    base_route_mode="formal",
                    run_identity={"compatibilityFingerprint": "new"},
                )

    def test_no_resume_allows_only_non_overlapping_worker_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifests" / "6_flan_base_z__run_01.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "outputId": "6_flan_base_z",
                        "sourceId": "flan_t5_large",
                        "run": 1,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                no_resume=True,
                output_root=root,
                case=[],
                source_id=["mistral_v0_3"],
                run=[],
            )
            with patch.object(orchestrator, "run") as mocked_run:
                orchestrator.command_generate(args)
            mocked_run.assert_called_once()

            args.source_id = ["flan_t5_large"]
            with self.assertRaises(orchestrator.OrchestrationError):
                orchestrator.command_generate(args)

    def test_recorder_requires_v5_1_method_judge_counts_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_root = root / "evaluation_reference_aligned_v5"
            evaluation_root.mkdir()
            table_paths = {}
            for name, filename in (
                ("scorecard", "reference_aligned_case_scorecard.tsv"),
                ("fieldScorecard", "reference_aligned_field_scorecard.tsv"),
                ("ablationScorecard", "reference_aligned_ablation_scorecard.tsv"),
                ("perRun", "reference_aligned_per_run.tsv"),
            ):
                path = evaluation_root / filename
                path.write_text("header\n", encoding="utf-8")
                table_paths[name] = str(path)
            evaluation = {
                "protocol": evaluator_v5.PROTOCOL,
                "method": evaluator_v5.method_metadata(),
                "judge": {
                    "model": "gpt-5.5",
                    "reasoningEffort": "medium",
                    "minimumConfidence": 0.8,
                },
                "sourceEvaluationConfigProtocol": (
                    "experiment6-narrative2-reference-aligned-v5.1-corrected12"
                ),
                "sourceEvaluationConfigSchemaVersion": 6,
                "evaluationConfigSha256": recorder.sha256_file(
                    REPO_ROOT
                    / "config"
                    / "experiment6_narrative2_evaluation_corrected_12.json"
                ),
                "status": "completed",
                "completedCases": 12,
                "completedCaseRuns": 120,
                "formalPredictions": 10200,
                "generationRoot": str(root),
                "tables": table_paths,
            }
            recorder.validate_evaluation_report(root, evaluation)
            for field, invalid in (
                ("protocol", "narrative2-reference-aligned-hybrid-v5"),
                ("completedCaseRuns", 119),
            ):
                with self.subTest(field=field):
                    changed = dict(evaluation)
                    changed[field] = invalid
                    with self.assertRaisesRegex(
                        RuntimeError, "refusing incompatible evaluation report"
                    ):
                        recorder.validate_evaluation_report(root, changed)

    def test_completion_evidence_gate_checks_rows_sources_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "cases" / "6_flan_base_z" / "run_01"
            run_dir.mkdir(parents=True)
            sources = [f"S{index:03d}" for index in range(85)]
            prompt_hashes = {source: f"hash-{source}" for source in sources}

            def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
                path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )

            predictions = run_dir / "predictions.jsonl"
            raw_response = run_dir / "raw_response.jsonl"
            prompts = run_dir / "prompts.jsonl"
            runtime = run_dir / "runtime.json"
            format_report = run_dir / "format_report.json"
            stage_raw = run_dir / "raw" / "native.txt"
            stage_raw.parent.mkdir()
            write_jsonl(
                predictions,
                [
                    {
                        "index": index,
                        "source": source,
                        "promptSha256": prompt_hashes[source],
                    }
                    for index, source in enumerate(sources)
                ],
            )
            write_jsonl(
                raw_response,
                [
                    {
                        "index": index,
                        "source": source,
                        "promptSha256": prompt_hashes[source],
                    }
                    for index, source in enumerate(sources)
                ],
            )
            write_jsonl(
                prompts,
                [
                    {
                        "index": index,
                        "source": source,
                        "directPromptSha256": prompt_hashes[source],
                    }
                    for index, source in enumerate(sources)
                ],
            )
            runtime.write_text("{}\n", encoding="utf-8")
            format_report.write_text("{}\n", encoding="utf-8")
            stage_raw.write_text("native output\n", encoding="utf-8")
            files = {
                "predictions": str(predictions),
                "rawResponse": str(raw_response),
                "prompts": str(prompts),
                "runtime": str(runtime),
                "formatReport": str(format_report),
                "stage1Raw": str(stage_raw),
            }
            manifest = {
                "run": 1,
                "seed": 2026073101,
                "expectedRows": 85,
                "acceptedRows": 85,
                "rejectedRows": 0,
                "runtimeBlockedRows": 0,
                "baseRouteMode": "formal",
                "declaredRoute": "direct-binding",
                "effectiveRoute": "direct-binding",
                "adapter": None,
                "converterModel": None,
                "sourceId": "flan_t5_large",
                "requestedModel": "flan_t5_large",
                "actualModel": "google/flan-t5-large",
                "resolvedSource": {"baseModel": "google/flan-t5-large"},
                "sourceRegistry": {"sha256": "registry"},
                "compatibilityFingerprint": "fingerprint",
                "executionTopology": {
                    "cudaVisibleDevices": "0",
                    "status": "resolved",
                    "name": "NVIDIA RTX A4500",
                },
                "files": files,
                "hashes": {
                    name: orchestrator.file_sha256(Path(path))
                    for name, path in files.items()
                },
            }
            self.assertEqual(orchestrator.validate_manifest_artifacts(root, manifest), [])

            predictions.write_text(
                predictions.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            self.assertIn(
                "predictions SHA-256 mismatch",
                orchestrator.validate_manifest_artifacts(root, manifest),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
