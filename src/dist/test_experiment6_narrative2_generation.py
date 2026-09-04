#!/usr/bin/env python3
"""Contract tests for the Experiment 6 narrative2 v2 runner."""

from __future__ import annotations

import copy
import csv
import io
import importlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external/FINDER/Retriever Codes"))
import evaluate_narrative2_reference_aligned_v5 as evaluator_v5
import run_experiment6_narrative2_generation as runner
import new_full_finqa_run as generator_runtime
import verify_experiment6_retrievers as retriever_verifier
import seq2seq_retriever


CONFIG_PATH = runner.REPO_ROOT / "config" / "experiment6_narrative2_generation.json"


class FakeUsage:
    def model_dump(self) -> dict[str, int]:
        return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class FakeCompletions:
    def __init__(self, failures: int, content: str) -> None:
        self.failures = failures
        self.content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("transport timeout")
        return SimpleNamespace(
            model="gpt-5.5",
            id="fake",
            usage=FakeUsage(),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self.content),
                finish_reason="stop",
            )],
        )


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class ConcurrentFakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def create(self, **kwargs):
        del kwargs
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        return SimpleNamespace(
            model="gpt-5.5",
            id="fake-concurrent",
            usage=FakeUsage(),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self.content),
                finish_reason="stop",
            )],
        )


def minimal_row() -> runner.InputRow:
    prompts = {
        mode: f"prompt {mode}"
        for mode in ("original", "zero-shot", "many-shot", "dynamic-shot")
    }
    shots = {
        "original": (),
        "zero-shot": (),
        "many-shot": tuple(range(26)),
        "dynamic-shot": tuple(range(10)),
    }
    return runner.InputRow(
        index=0,
        number="1",
        source="S1",
        data_raw='[{"Revenue":1}]',
        data_compact='__row__,Revenue\n0,1',
        text="Revenue increased.",
        retriever_prompts=prompts,
        direct_prompts=prompts,
        shot_ids=shots,
    )


class Narrative2GenerationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = runner.load_config(CONFIG_PATH)
        cls.cases = runner.expand_matrix(cls.config)

    def test_matrix_is_exact_9_24_17_4_plus_four_controls(self) -> None:
        official = [case for case in self.cases if case.official]
        controls = [case for case in self.cases if not case.official]
        self.assertEqual(len(official), 54)
        self.assertEqual(len(controls), 4)
        self.assertEqual(
            {part: sum(case.part == part for case in official) for part in range(1, 5)},
            {1: 9, 2: 24, 3: 17, 4: 4},
        )
        self.assertEqual(len({case.output_id for case in self.cases}), 58)
        self.assertEqual(self.config["runs"], list(range(1, 11)))
        self.assertTrue(any(case.source_id == "llama4" for case in official))
        self.assertFalse(any(case.source_id == "llama3_3" for case in official))
        self.assertIs(self.config["runtimeRoutes"]["qwen3_6"]["enableThinking"], False)
        self.assertEqual(self.config["runtimeRoutes"]["qwen3_6"]["responseFormat"], "json_schema")
        self.assertEqual(
            self.config["runtimeRoutes"]["qwen3_6"]["responseSchemaSha256"],
            runner.sha256_file(
                runner.REPO_ROOT / self.config["runtimeRoutes"]["qwen3_6"]["responseSchemaPath"]
            ),
        )

    def test_prediction_counts_distinguish_full_matrix_from_selected_case(self) -> None:
        selected = [
            next(case for case in self.cases if case.output_id == "6_mistral_d")
        ]
        full = runner.prediction_count_summary(
            all_cases=self.cases,
            selected_cases=selected,
            configured_runs=self.config["runs"],
            selected_runs=self.config["runs"],
            expected_rows=self.config["expectedRows"],
            selected_rows=85,
        )
        self.assertEqual(full["expectedFormalPredictions"], 54 * 10 * 85)
        self.assertEqual(full["expectedControlPredictions"], 4 * 10 * 85)
        self.assertEqual(full["selectedFormalPredictions"], 850)
        self.assertEqual(full["selectedControlPredictions"], 0)

        smoke = runner.prediction_count_summary(
            all_cases=self.cases,
            selected_cases=selected,
            configured_runs=self.config["runs"],
            selected_runs=[1],
            expected_rows=self.config["expectedRows"],
            selected_rows=1,
        )
        self.assertEqual(smoke["selectedFormalPredictions"], 1)
        self.assertEqual(
            self.config["runtimeRoutes"]["llama4"]["responseFormat"],
            "json_schema",
        )
        self.assertEqual(
            self.config["runtimeRoutes"]["llama4"]["responseSchemaSha256"],
            runner.sha256_file(
                runner.REPO_ROOT / self.config["runtimeRoutes"]["llama4"]["responseSchemaPath"]
            ),
        )
        self.assertEqual(
            self.config["runtimeRoutes"]["mistral4"]["responseFormat"],
            "json_schema",
        )

    def test_retriever_scheduler_shards_are_exact_and_disjoint(self) -> None:
        scheduler = self.config["retriever"]["scheduler"]
        self.assertEqual(scheduler["familyWorkers"], 2)
        self.assertEqual(
            scheduler["familyWorkerOverrides"], {"flan": 3, "mistral": 3}
        )
        for family, shards in scheduler["familyShards"].items():
            expected_workers = scheduler["familyWorkerOverrides"].get(
                family, scheduler["familyWorkers"]
            )
            self.assertEqual(len(shards), expected_workers)
            flattened = [output_id for shard in shards for output_id in shard]
            expected = [
                case.output_id
                for case in self.cases
                if runner.family_for_source(case.source_id) == family
            ]
            self.assertEqual(len(flattened), 10)
            self.assertEqual(len(set(flattened)), 10)
            self.assertEqual(set(flattened), set(expected))

    def test_workbook_rows_shots_and_gold_isolation(self) -> None:
        rows, report = runner.read_input_rows(self.config, 0)
        self.assertEqual(len(rows), 85)
        self.assertEqual(report["fullWorkbookRows"], 85)
        for row in rows:
            self.assertEqual(len(row.shot_ids["many-shot"]), 26)
            self.assertEqual(len(row.shot_ids["dynamic-shot"]), 10)
            for prompt in (*row.retriever_prompts.values(), *row.direct_prompts.values()):
                self.assertNotIn('"targetBindings"', prompt)
                self.assertNotIn('"Binding_Result"', prompt)
                self.assertNotIn("BEGIN EXPERIMENT 6 TWO-STAGE APPENDIX", prompt)
                self.assertIn(runner.POSITION_INDEX_CONTRACT, prompt)
        self.assertEqual(report["chartNormalization"]["repairRows"], 1)
        self.assertEqual(report["chartNormalization"]["repairedSources"], ["Econ_280"])
        econ_280 = next(row for row in rows if row.source == "Econ_280")
        parsed = list(csv.reader(io.StringIO(econ_280.data_compact)))
        self.assertEqual(
            parsed[0], ["__row__", "Country", "Vaccination", "Vaccinatione"]
        )
        self.assertEqual(parsed[6][2], '"0.63%"')

    def test_four_prompt_bundles_are_frozen_and_route_aware(self) -> None:
        rows, _ = runner.read_input_rows(self.config, 0)
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            bound, report = runner.materialize_prompt_bundles(
                output_root, rows, self.config
            )
            self.assertEqual(report["rowCount"], 85)
            self.assertTrue(report["generationReadsFrozenMessages"])
            self.assertEqual(
                set(report["modes"]),
                {"original", "zero-shot", "many-shot", "dynamic-shot"},
            )
            for mode, directory in runner.BUNDLE_MODE_DIRECTORIES.items():
                mode_root = output_root / "input_bundles" / directory
                manifest = json.loads(
                    (mode_root / "bundle_manifest.json").read_text()
                )
                model_inputs = [
                    json.loads(line)
                    for line in (mode_root / "model_inputs.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(manifest["rowCount"], 85)
                self.assertEqual(manifest["promptMode"], mode)
                self.assertEqual(len(model_inputs), 85)
                self.assertEqual(manifest["routes"], ["retriever", "direct"])
            self.assertNotEqual(
                bound[0].retriever_prompts["zero-shot"],
                bound[0].direct_prompts["zero-shot"],
            )
            rebound, repeated_report = runner.materialize_prompt_bundles(
                output_root, rows, self.config
            )
            self.assertEqual(
                repeated_report["modes"]["many-shot"]["manifestSha256"],
                report["modes"]["many-shot"]["manifestSha256"],
            )
            self.assertEqual(
                rebound[0].direct_prompts,
                bound[0].direct_prompts,
            )
            corrupted = (
                output_root
                / "input_bundles"
                / "zero"
                / "messages"
                / "direct"
                / f"{rows[0].source}.txt"
            )
            corrupted.write_text("corrupted", encoding="utf-8")
            with self.assertRaisesRegex(
                runner.ProtocolError, "differs from regenerated bytes"
            ):
                runner.materialize_prompt_bundles(
                    output_root, rows, self.config
                )

    def test_compact_chart_is_deterministic_and_lossless_value_repr(self) -> None:
        raw = '{"Year":[2023,2024],"Revenue":[1.5,null]}'
        compact = runner.compact_chart_data(raw)
        self.assertEqual(compact, runner.compact_chart_data(raw))
        self.assertIn("__row__,Year,Revenue", compact)
        self.assertIn("2023,1.5", compact)
        self.assertIn("2024,null", compact)

    def test_compact_chart_repairs_only_bare_percentage_json_values(self) -> None:
        raw = '[{"Country":"India","Vaccination":0.63%}]'
        compact, audit = runner.compact_chart_data_with_audit(raw)
        parsed = list(csv.reader(io.StringIO(compact)))
        self.assertEqual(parsed[0], ["__row__", "Country", "Vaccination"])
        self.assertEqual(parsed[1], ["0", '"India"', '"0.63%"'])
        self.assertFalse(audit["inputStrictJson"])
        self.assertEqual(
            audit["repairRule"], "quote-bare-percentage-json-values-v1"
        )
        self.assertEqual(audit["repairCount"], 1)
        untouched, count = runner.repair_bare_percentage_json_values(
            '[{"Text":"literal :0.63% text","Value":1}]'
        )
        self.assertEqual(count, 0)
        self.assertEqual(
            untouched, '[{"Text":"literal :0.63% text","Value":1}]'
        )

    def test_compact_chart_blocks_unapproved_invalid_json(self) -> None:
        with self.assertRaisesRegex(
            runner.ProtocolError, "no approved lossless repair"
        ):
            runner.compact_chart_data('[{"Value":not-json}]')

    def test_position_contract_excludes_synthetic_row_column(self) -> None:
        contract = runner.POSITION_INDEX_CONTRACT
        self.assertIn("never count __row__ as a data column", contract)
        self.assertIn("data_column_index 0 is the first", contract)
        self.assertIn("exactly and case-sensitively equal", contract)
        self.assertIn("measured series", contract)
        self.assertIn("do not substitute the axis/category label cell", contract)
        self.assertIn("keep Num empty", contract)
        self.assertIn(contract, runner.DIRECT_SYSTEM_PROMPT)
        self.assertIn(contract, runner.CONVERTER_SYSTEM_PROMPT)
        case = runner.MatrixCase(
            output_id="6_flan_z", source_id="finqa_flan_z",
            prompt_mode="zero-shot", route="adapter-converter",
            part=1, official=True,
        )
        self.assertIn(contract, runner.converter_prompt(minimal_row(), "fact", case))

    def test_native_retriever_routes_receive_canonical_8192_and_128(self) -> None:
        source_by_family = {
            "flan": "flan_t5_large",
            "mistral": "mistral_v0_3",
            "t5gemma2": "t5gemma_2_1b_1b",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "input.csv"
            csv_path.write_text("input,Rel_Fact,Source\nx,__BLINDED__,S1\n", encoding="utf-8")
            for family, source_id in source_by_family.items():
                with self.subTest(family=family):
                    pred_dir = root / family
                    (pred_dir / "raw").mkdir(parents=True)
                    namespace = runner.retriever_namespace(
                        pred_dir, self.config, 1, family=family
                    )
                    captured: list[str] = []

                    def fake_run(command, log_path, timeout, cuda):
                        del log_path, timeout, cuda
                        captured.extend(command)
                        output = Path(command[command.index("--output-txt") + 1])
                        output.parent.mkdir(parents=True, exist_ok=True)
                        if family == "t5gemma2":
                            output.write_text(
                                json.dumps({"predicted_label": '{"RetFact":"x"}'}) + "\n",
                                encoding="utf-8",
                            )
                        else:
                            output.write_text('True: x Pred: {"RetFact":"x"}\n', encoding="utf-8")
                        return SimpleNamespace(returncode=0)

                    case = runner.legacy.MatrixCase("case", source_id, "narrative_zero_shot")
                    with patch.object(runner.legacy, "run_command", side_effect=fake_run):
                        predictions, runtime = runner.legacy.run_retriever_case(
                            case, csv_path, "zero-shot", namespace, use_adapter=False,
                        )
                    self.assertEqual(len(predictions), 1)
                    self.assertEqual(
                        captured[captured.index("--structured-output") + 1],
                        "canonical",
                    )
                    length_flag = "--max-input-length" if family == "mistral" else "--max-length"
                    self.assertEqual(captured[captured.index(length_flag) + 1], "8192")
                    self.assertEqual(captured[captured.index("--max-new-tokens") + 1], "128")
                    self.assertEqual(runtime["structured_output"], "canonical")
                    self.assertEqual(namespace.batch_size, 1)
                    self.assertEqual(
                        namespace.cuda_visible_devices,
                        self.config["retriever"]["familyCudaVisibleDevices"][family],
                    )
                    if family == "t5gemma2":
                        self.assertEqual(
                            captured[captured.index("--cache-safe-input-tokens") + 1],
                            str(
                                self.config["retriever"]["generationCache"]["t5gemma2"][
                                    "disableAboveInputTokens"
                                ]
                            ),
                        )

    def test_retriever_runtime_profile_records_effective_batch(self) -> None:
        case = runner.MatrixCase(
            "6_mistral_z", "finqa_mistral_z", "zero-shot",
            "adapter-converter", 1, True,
        )
        runtime = runner.public_model_runtime(case, {
            "family": "mistral",
            "adapter_dir": "/tmp/adapter",
            "batch_size": 4,
        })
        self.assertEqual(runtime["runtimeProfile"], "mistral-canonical-batch4")
        self.assertEqual(runtime["quantization"], "4bit-nf4")

    def test_explicit_output_contract_is_not_overridden_by_legacy_suffix(self) -> None:
        prompt = 'instruction\\n\\n## Output contract\\nReturn exactly {"result":[],"reason":""}'
        rendered = runner.legacy.direct_binding_prompt({"input": prompt})
        self.assertEqual(rendered, prompt)
        self.assertNotIn('"Binding"', rendered)

    def test_strict_format_failure_is_zero_and_repair_is_nonformal(self) -> None:
        valid = (
            '{"result":[{"ObjectName":["Revenue"],"DataName":"Revenue",'
            '"Position":[{"Begin":[0,1],"End":[0,1]}],"Trend":"increase",'
            '"Num":[1],"Text":"Revenue increased."}],"reason":"ok"}'
        )
        result, reason, report = runner.strict_parse_output(valid, converter=True)
        self.assertTrue(report["valid"])
        self.assertEqual(len(result), 1)
        self.assertEqual(reason, "ok")

        fence = chr(96) * 3
        fenced = fence + "json\n" + valid + "\n" + fence
        result, _, report = runner.strict_parse_output(fenced, converter=True)
        self.assertFalse(report["valid"])
        self.assertEqual(result, [])
        self.assertTrue(runner.nonformal_repair(fenced)["available"])

    def test_nonformal_repair_rejects_nan_and_remains_serializable(self) -> None:
        repair = runner.nonformal_repair('{"result":[{"Num":[NaN]}]}')
        self.assertEqual(repair, {"available": False})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repair.jsonl"
            runner.write_jsonl(path, [repair])
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), repair
            )


    def test_converter_retries_transport_three_times_but_not_format(self) -> None:
        case = runner.MatrixCase(
            "control_converter_zero", "blank_candidate", "zero-shot",
            "converter-control", 0, False,
        )
        valid = '{"result":[],"reason":"candidate insufficient"}'
        config = copy.deepcopy(self.config)
        config["converter"]["retryDelaysSeconds"] = [0, 0]
        completions = FakeCompletions(2, valid)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "OpenAI", return_value=FakeClient(completions)):
                outputs, _ = runner.run_converter(
                    case=case,
                    rows=[minimal_row()],
                    candidates=[""],
                    run_dir=Path(directory),
                    run_seed=123,
                    config=config,
                )
        self.assertEqual(completions.calls, 3)
        self.assertEqual(outputs, [valid])

        invalid_completions = FakeCompletions(0, "not-json")
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with patch.object(runner, "OpenAI", return_value=FakeClient(invalid_completions)):
                outputs, _ = runner.run_converter(
                    case=case,
                    rows=[minimal_row()],
                    candidates=[""],
                    run_dir=run_dir,
                    run_seed=123,
                    config=config,
                )
            self.assertEqual(invalid_completions.calls, 1)
            checkpoint = [
                json.loads(line)
                for line in (run_dir / "converter_raw_responses.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(checkpoint[0]["status"], "completed_format_invalid")
            self.assertEqual(checkpoint[0]["formalResult"], [])
            self.assertFalse(checkpoint[0]["nonformalRepair"]["available"])

    def test_prediction_contract_has_nested_model_prompt_hash_and_raw_response(self) -> None:
        row = minimal_row()
        case = runner.MatrixCase(
            "6_flan_z", "finqa_flan_z", "zero-shot",
            "adapter-converter", 1, True,
        )
        runtime = {
            "requestedModel": "finqa_flan_z",
            "actualModel": "google/flan-t5-large",
            "adapter": "/adapter",
            "runtimeProfile": "flan-canonical-batch1",
        }
        predictions, _, _ = runner.normalize_predictions(
            case=case,
            rows=[row],
            raw_predictions=['{"result":[],"reason":"candidate insufficient"}'],
            run_number=1,
            run_seed=123,
            runtime=runtime,
            converter=True,
        )
        prediction = predictions[0]
        self.assertEqual(prediction["model"]["requestedModel"], "finqa_flan_z")
        self.assertEqual(prediction["model"]["actualModel"], "google/flan-t5-large")
        self.assertEqual(
            prediction["promptSha256"],
            runner.sha256_text(row.retriever_prompts["zero-shot"]),
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            runner.write_jsonl(run_dir / "converter_raw_responses.jsonl", [{
                "index": 0,
                "status": "completed_format_valid",
                "stopReason": "stop",
                "usage": {"total_tokens": 15},
                "transportErrors": [],
            }])
            raw_records = runner.canonical_raw_response_records(
                predictions, run_dir, converter=True,
            )
        self.assertEqual(raw_records[0]["stopReason"], "stop")
        self.assertEqual(raw_records[0]["tokenUsage"]["total_tokens"], 15)
        self.assertEqual(raw_records[0]["promptSha256"], prediction["promptSha256"])

    def test_converter_parallelism_keeps_ordered_atomic_checkpoint(self) -> None:
        case = runner.MatrixCase(
            "control_converter_zero", "blank_candidate", "zero-shot",
            "converter-control", 0, False,
        )
        config = copy.deepcopy(self.config)
        config["converter"]["parallelism"] = 2
        completions = ConcurrentFakeCompletions(
            '{"result":[],"reason":"candidate insufficient"}'
        )
        rows = [copy.deepcopy(minimal_row()) for _ in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with patch.object(
                runner, "OpenAI", return_value=FakeClient(completions)
            ):
                outputs, runtime = runner.run_converter(
                    case=case,
                    rows=rows,
                    candidates=[""] * len(rows),
                    run_dir=run_dir,
                    run_seed=123,
                    config=config,
                )
            checkpoint = [
                json.loads(line)
                for line in (
                    run_dir / "converter_raw_responses.jsonl"
                ).read_text().splitlines()
                if line.strip()
            ]
        self.assertEqual(completions.max_active, 2)
        self.assertEqual(runtime["parallelism"], 2)
        self.assertEqual([item["index"] for item in checkpoint], list(range(4)))
        self.assertEqual(len(outputs), 4)

    def test_runtime_provenance_separates_profile_from_quantization(self) -> None:
        api_case = runner.MatrixCase(
            "6_gpt5.5_z", "gpt5_5", "zero-shot", "direct-binding", 2, True,
        )
        api_runtime = runner.public_model_runtime(api_case, {
            "config": {
                "engine": "gpt5_5",
                "actual_model": "gpt-5.5",
                "route": "chatmock_openai_compatible",
                "runtime_profile": "formal",
            },
        })
        self.assertEqual(api_runtime["runtimeProfile"], "formal")
        self.assertIsNone(api_runtime["quantization"])

        qwen_case = runner.MatrixCase(
            "6_qwen_z", "qwen3_6", "zero-shot", "direct-binding", 2, True,
        )
        with patch.dict(runner.os.environ, {
            "VLLM_RUNTIME_PROFILE": "qwen_fp8_tp2_precise_kv",
            "VLLM_QUANTIZATION": "fp8",
            "QWEN3_6_ENABLE_THINKING": "false",
            "GENERATOR_RESPONSE_FORMAT": "json_schema",
            "GENERATOR_RESPONSE_SCHEMA_PATH": "config/experiment6_narrative2_binding_schema.json",
        }, clear=False):
            qwen_runtime = runner.public_model_runtime(qwen_case, {
                "config": {
                    "engine": "qwen3_6",
                    "actual_model": "Qwen/Qwen3.6-35B-A3B-FP8",
                    "route": "local_vllm_openai_compatible",
                    "runtime_profile": "formal",
                },
            })
        self.assertEqual(qwen_runtime["runtimeProfile"], "qwen_fp8_tp2_precise_kv")
        self.assertEqual(qwen_runtime["quantization"], "fp8")
        self.assertIs(qwen_runtime["thinkingEnabled"], False)
        self.assertEqual(qwen_runtime["responseFormat"], "json_schema")
        self.assertEqual(
            qwen_runtime["responseSchemaPath"],
            "config/experiment6_narrative2_binding_schema.json",
        )

    def test_runtime_blocked_rows_preserve_model_and_error_provenance(self) -> None:
        case = runner.MatrixCase(
            "6_gpt4.1_z", "gpt4_1", "zero-shot", "direct-binding", 4, True,
        )
        provenance = runner.failure_model_provenance(case, self.config, [])
        self.assertEqual(provenance["requestedModel"], "gpt4_1")
        self.assertIsNone(provenance["actualModel"])
        self.assertEqual(provenance["intendedModel"], "gpt4_1")

        rows = [minimal_row(), copy.deepcopy(minimal_row())]
        error = {"type": "RuntimeError", "message": "missing credentials"}
        records = runner.runtime_blocked_row_records(
            case, rows, 3, 2026073103, provenance, error,
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(all(item["status"] == "runtime_blocked" for item in records))
        self.assertTrue(all(item["requestedModel"] == "gpt4_1" for item in records))
        self.assertTrue(all(item["actualModel"] is None for item in records))
        self.assertTrue(all(item["error"]["message"] == "missing credentials" for item in records))

        adapter_case = runner.MatrixCase(
            "6_flan_z", "finqa_flan_z", "zero-shot", "retriever-converter", 1, True,
        )
        adapter_provenance = runner.failure_model_provenance(
            adapter_case, self.config, [],
        )
        self.assertEqual(adapter_provenance["intendedModel"], "google/flan-t5-large")
        self.assertIsNotNone(adapter_provenance["adapter"])

    def test_json_schema_response_format_is_opt_in_and_validated(self) -> None:
        with patch.dict(runner.os.environ, {
            "GENERATOR_RESPONSE_FORMAT": "json_schema",
            "GENERATOR_RESPONSE_SCHEMA_PATH": "config/experiment6_narrative2_binding_schema.json",
        }, clear=False):
            response_format = generator_runtime.generator_response_format()
            self.assertEqual(response_format["type"], "json_schema")
            self.assertEqual(
                response_format["json_schema"]["name"],
                "experiment6_narrative2_binding_v2",
            )
        with patch.dict(runner.os.environ, {"GENERATOR_RESPONSE_FORMAT": "invalid"}, clear=False):
            with self.assertRaises(ValueError):
                generator_runtime.generator_response_format()

    def test_direct_response_metadata_preserves_finish_reason_and_usage(self) -> None:
        result = SimpleNamespace(
            id="response-1",
            model="qwen3_6",
            usage=FakeUsage(),
            choices=[SimpleNamespace(finish_reason="length")],
        )
        config = SimpleNamespace(model="qwen3_6")
        metadata = generator_runtime.response_generation_metadata(result, config)
        self.assertEqual(metadata["responseModel"], "qwen3_6")
        self.assertEqual(metadata["finishReasons"], ["length"])
        self.assertEqual(metadata["usage"]["completion_tokens"], 5)

        fake_runtime = SimpleNamespace(
            generate_text=lambda *args, **kwargs: ["raw output"],
            last_generation_metadata=lambda: metadata,
        )
        outputs, returned = runner.legacy.generate_text_with_timeout(
            fake_runtime, SimpleNamespace(), "prompt", "system", 128, 0,
            return_metadata=True,
        )
        self.assertEqual(outputs, ["raw output"])
        self.assertEqual(returned, metadata)

    def test_gpt55_direct_response_identity_is_required(self) -> None:
        config = SimpleNamespace(engine="gpt5_5", actual_model="gpt-5.5")
        runner.legacy.require_response_model_identity(
            "gpt5_5", config, {"responseModel": "gpt-5.5"},
        )
        with self.assertRaises(runner.legacy.ResponseModelIdentityError) as raised:
            runner.legacy.require_response_model_identity(
                "gpt5_5", config, {"responseModel": "gpt-5.4"},
            )
        self.assertEqual(
            runner.legacy.generation_failure_category(
                generator_runtime, raised.exception,
            ),
            "runtime_blocked_model_identity",
        )

    def test_parallel_child_defers_response_identity_to_parent(self) -> None:
        class RecordingQueue:
            def __init__(self) -> None:
                self.payload = None

            def put(self, payload) -> None:
                self.payload = payload

        response_metadata = {"responseModel": "gpt-5.4"}
        fake_runtime = SimpleNamespace(
            generate_text=lambda *args, **kwargs: ["raw output"],
            last_generation_metadata=lambda: response_metadata,
        )
        result_queue = RecordingQueue()
        config = SimpleNamespace(engine="gpt5_5", actual_model="gpt-5.5")
        with patch.object(
            runner.legacy.importlib,
            "import_module",
            return_value=fake_runtime,
        ):
            runner.legacy._parallel_generation_target(
                result_queue,
                "fake_runtime",
                config,
                "prompt",
                "system",
                128,
            )

        self.assertTrue(result_queue.payload["ok"])
        self.assertEqual(result_queue.payload["metadata"], response_metadata)
        with self.assertRaises(runner.legacy.ResponseModelIdentityError):
            runner.legacy.require_response_model_identity(
                "gpt5_5",
                config,
                result_queue.payload["metadata"],
            )

    def test_direct_row_timeout_continues_and_resume_retries_only_failed_row(self) -> None:
        case = runner.legacy.MatrixCase(
            "6_llama_z", "llama4", "narrative_zero",
        )
        prompt_rows = [{"Source": f"S{index}"} for index in range(3)]
        prompts = [f"prompt-{index}" for index in range(3)]
        args = SimpleNamespace(
            binding_generator_parallelism=1,
            binding_generator_total_timeout_seconds=0,
        )
        fake_runtime = SimpleNamespace()
        fake_config = SimpleNamespace(to_public_dict=lambda: {"engine": "llama4"})
        metadata = {
            "responseId": "response",
            "requestResponseFormat": "json_schema",
            "finishReasons": ["stop"],
        }

        def first_pass(*call_args, **call_kwargs):
            del call_kwargs
            prompt = call_args[2]
            if prompt == "prompt-1":
                raise runner.legacy.RowTimeoutError(
                    "generator row timed out after 600s"
                )
            return ([f'{{"result":[],"reason":"{prompt}"}}'], metadata)

        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.jsonl"
            with patch.object(
                runner.legacy,
                "resolve_generator_runtime",
                return_value=(fake_runtime, fake_config, "available"),
            ), patch.object(
                runner.legacy,
                "generate_text_with_timeout",
                side_effect=first_pass,
            ):
                outputs, runtime = (
                    runner.legacy.generate_binding_predictions_with_engine(
                        case=case,
                        prompt_rows=prompt_rows,
                        prompts=prompts,
                        engine="llama4",
                        args=args,
                        raw_path=raw_path,
                        stage="direct_binding_generation",
                        max_tokens=4096,
                        row_timeout_seconds=600,
                    )
                )

            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            self.assertEqual(
                [row["status"] for row in rows],
                ["completed", "runtime_blocked", "completed"],
            )
            self.assertTrue(outputs[0])
            self.assertEqual(outputs[1], "")
            self.assertTrue(outputs[2])
            self.assertEqual(runtime["processed_rows"], 3)
            self.assertEqual(runtime["completed_rows"], 2)
            self.assertEqual(runtime["runtime_blocked_rows"], 1)
            self.assertEqual(runtime["status"], "runtime_blocked")

            retry_calls = []

            def retry_failed(*call_args, **call_kwargs):
                del call_kwargs
                prompt = call_args[2]
                retry_calls.append(prompt)
                return ([f'{{"result":[],"reason":"{prompt}"}}'], metadata)

            with patch.object(
                runner.legacy,
                "resolve_generator_runtime",
                return_value=(fake_runtime, fake_config, "available"),
            ), patch.object(
                runner.legacy,
                "generate_text_with_timeout",
                side_effect=retry_failed,
            ):
                resumed_outputs, resumed_runtime = (
                    runner.legacy.generate_binding_predictions_with_engine(
                        case=case,
                        prompt_rows=prompt_rows,
                        prompts=prompts,
                        engine="llama4",
                        args=args,
                        raw_path=raw_path,
                        stage="direct_binding_generation",
                        max_tokens=4096,
                        row_timeout_seconds=600,
                    )
                )

            self.assertEqual(retry_calls, [])
            self.assertEqual(resumed_outputs[1], "")
            self.assertEqual(resumed_runtime["completed_rows"], 2)
            self.assertEqual(resumed_runtime["runtime_blocked_rows"], 1)
            self.assertEqual(resumed_runtime["status"], "runtime_blocked")

            with patch.object(
                runner.legacy,
                "resolve_generator_runtime",
                return_value=(fake_runtime, fake_config, "available"),
            ), patch.object(
                runner.legacy,
                "generate_text_with_timeout",
                side_effect=retry_failed,
            ):
                changed_timeout_outputs, changed_timeout_runtime = (
                    runner.legacy.generate_binding_predictions_with_engine(
                        case=case,
                        prompt_rows=prompt_rows,
                        prompts=prompts,
                        engine="llama4",
                        args=args,
                        raw_path=raw_path,
                        stage="direct_binding_generation",
                        max_tokens=4096,
                        row_timeout_seconds=1200,
                    )
                )

            self.assertEqual(retry_calls, ["prompt-1"])
            self.assertTrue(all(changed_timeout_outputs))
            self.assertEqual(changed_timeout_runtime["completed_rows"], 3)
            self.assertEqual(changed_timeout_runtime["runtime_blocked_rows"], 0)
            self.assertEqual(changed_timeout_runtime["status"], "completed")

    def test_direct_parallel_timeout_checkpoint_is_ordered_atomic_and_not_retried(self) -> None:
        case = runner.legacy.MatrixCase(
            "6_llama_z", "llama4", "narrative_zero",
        )
        prompt_rows = [{"Source": f"S{index}"} for index in range(4)]
        prompts = [f"prompt-{index}" for index in range(4)]
        args = SimpleNamespace(
            binding_generator_parallelism=2,
            binding_generator_total_timeout_seconds=0,
        )

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            module_name = "experiment6_spawn_test_runtime"
            call_log = directory_path / "calls.log"
            (directory_path / f"{module_name}.py").write_text(
                """import json
import time
_last = {}
def generate_text(config, prompt, system_prompt, profile, max_tokens):
    del system_prompt, profile, max_tokens
    with open(config.endpoint, "a", encoding="utf-8") as handle:
        handle.write(prompt + "\\n")
    if prompt == "prompt-1":
        time.sleep(2)
    global _last
    _last = {
        "responseId": "response-" + prompt,
        "requestResponseFormat": "json_schema",
        "finishReasons": ["stop"],
    }
    return [json.dumps({"result": [], "reason": prompt})]
def last_generation_metadata():
    return dict(_last)
def classify_generation_exception(exc):
    del exc
    return "transport"
""",
                encoding="utf-8",
            )
            sys.path.insert(0, directory)
            importlib.invalidate_caches()
            fake_runtime = importlib.import_module(module_name)
            fake_config = generator_runtime.EngineConfig(
                requested_engine="llama4",
                engine="llama4",
                route="local_vllm_openai_compatible",
                model="llama4",
                actual_model="llama4",
                formal_model="llama4",
                runtime_profile="test_batch2",
                endpoint=str(call_log),
                api_version=None,
                api_key=None,
                missing_credentials=[],
                credential_sources={},
                credential_files=[],
                credential_warnings=[],
            )
            raw_path = directory_path / "raw.jsonl"
            try:
                with patch.object(
                    runner.legacy,
                    "resolve_generator_runtime",
                    return_value=(fake_runtime, fake_config, "available"),
                ):
                    outputs, runtime = (
                        runner.legacy.generate_binding_predictions_with_engine(
                            case=case,
                            prompt_rows=prompt_rows,
                            prompts=prompts,
                            engine="llama4",
                            args=args,
                            raw_path=raw_path,
                            stage="direct_binding_generation",
                            max_tokens=4096,
                            row_timeout_seconds=1,
                        )
                    )

                rows = [
                    json.loads(line) for line in raw_path.read_text().splitlines()
                ]
                self.assertEqual([row["index"] for row in rows], list(range(4)))
                self.assertEqual(
                    [row["status"] for row in rows],
                    ["completed", "runtime_blocked", "completed", "completed"],
                )
                self.assertEqual(rows[1]["error"]["row_timeout_seconds"], 1)
                self.assertEqual(
                    rows[0]["response"]["executionBatch"],
                    {
                        "bindingGeneratorParallelism": 2,
                        "parallelProcessStartMethod": "fork",
                    },
                )
                self.assertEqual(
                    rows[1]["execution"],
                    {
                        "bindingGeneratorParallelism": 2,
                        "parallelProcessStartMethod": "fork",
                        "runtimeProfile": "test_batch2",
                        "actualModel": "llama4",
                    },
                )
                self.assertEqual(runtime["completed_rows"], 3)
                self.assertEqual(runtime["runtime_blocked_rows"], 1)
                self.assertEqual(outputs[1], "")
                self.assertEqual(list(directory_path.glob("*.tmp")), [])
                calls_before_resume = call_log.read_text().splitlines()
                self.assertCountEqual(calls_before_resume, prompts)

                with patch.object(
                    runner.legacy,
                    "resolve_generator_runtime",
                    return_value=(fake_runtime, fake_config, "available"),
                ):
                    resumed_outputs, resumed_runtime = (
                        runner.legacy.generate_binding_predictions_with_engine(
                            case=case,
                            prompt_rows=prompt_rows,
                            prompts=prompts,
                            engine="llama4",
                            args=args,
                            raw_path=raw_path,
                            stage="direct_binding_generation",
                            max_tokens=4096,
                            row_timeout_seconds=1,
                        )
                    )

                self.assertEqual(call_log.read_text().splitlines(), calls_before_resume)
                self.assertEqual(resumed_outputs[1], "")
                self.assertEqual(resumed_runtime["runtime_blocked_rows"], 1)
            finally:
                sys.modules.pop(module_name, None)
                sys.path.remove(directory)

    def test_direct_runtime_config_records_effective_env_provenance(self) -> None:
        case = runner.legacy.MatrixCase(
            "6_llama_z", "llama4", "narrative_zero",
        )
        fake_config = generator_runtime.EngineConfig(
            requested_engine="llama4",
            engine="llama4",
            route="local_vllm_openai_compatible",
            model="llama4",
            actual_model="alias-model",
            formal_model="llama4",
            runtime_profile="fallback_smoke",
            endpoint="http://localhost:8010/v1",
            api_version=None,
            api_key=None,
            missing_credentials=[],
            credential_sources={},
            credential_files=[],
            credential_warnings=[],
        )
        args = SimpleNamespace(
            binding_generator_parallelism=1,
            binding_generator_total_timeout_seconds=0,
        )
        metadata = {
            "requestResponseFormat": "json_schema",
            "finishReasons": ["stop"],
        }
        effective_environment = {
            "VLLM_RUNTIME_PROFILE": "llama4_formal_profile",
            "LLAMA4_MODEL_PATH": "/models/llama4-formal",
            "VLLM_QUANTIZATION": "w4a16",
            "VLLM_MAX_NUM_SEQS": "2",
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner.legacy,
            "resolve_generator_runtime",
            return_value=(SimpleNamespace(), fake_config, "available"),
        ), patch.object(
            runner.legacy,
            "generate_text_with_timeout",
            return_value=(['{"result":[],"reason":"ok"}'], metadata),
        ), patch.dict(os.environ, effective_environment, clear=False):
            raw_path = Path(directory) / "raw.jsonl"
            _, runtime = runner.legacy.generate_binding_predictions_with_engine(
                case=case,
                prompt_rows=[{"Source": "S0"}],
                prompts=["prompt-0"],
                engine="llama4",
                args=args,
                raw_path=raw_path,
                stage="direct_binding_generation",
                max_tokens=4096,
                row_timeout_seconds=600,
            )
            row = json.loads(raw_path.read_text().strip())

        self.assertEqual(
            runtime["config"]["runtime_profile"], "llama4_formal_profile",
        )
        self.assertEqual(runtime["config"]["actual_model"], "/models/llama4-formal")
        self.assertEqual(runtime["config"]["quantization"], "w4a16")
        self.assertEqual(row["execution"]["runtimeProfile"], "llama4_formal_profile")
        self.assertEqual(row["execution"]["actualModel"], "/models/llama4-formal")
        self.assertEqual(row["execution"]["quantization"], "w4a16")
        self.assertEqual(row["execution"]["serviceMaxNumSeqs"], 2)

    def test_no_resume_archives_prior_run_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run_dir = output_root / "cases" / "6_llama_z" / "run_01"
            raw_path = run_dir / "raw" / "6_llama_z.jsonl"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text('{"index":0,"status":"completed"}\n')
            manifest_path = output_root / "manifests" / "6_llama_z__run_01.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text('{"status":"completed"}\n')

            archive = runner.archive_no_resume_artifacts(
                output_root=output_root,
                run_dir=run_dir,
                manifest_path=manifest_path,
                output_id="6_llama_z",
                run_number=1,
            )

            self.assertIsNotNone(archive)
            assert archive is not None
            self.assertFalse(run_dir.exists())
            self.assertFalse(manifest_path.exists())
            self.assertTrue((archive / "run_dir" / "raw" / raw_path.name).is_file())
            self.assertTrue((archive / "manifest.json").is_file())

    def test_generation_snapshot_is_immutable_and_worker_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = runner.freeze_generation_snapshot(root, self.config)
            first_bytes = snapshot.read_bytes()
            runner.freeze_generation_snapshot(root, self.config)
            self.assertEqual(snapshot.read_bytes(), first_bytes)
            changed = json.loads(json.dumps(self.config))
            changed["seedBase"] += 1
            with self.assertRaises(runner.ProtocolError):
                runner.freeze_generation_snapshot(root, changed)

    def test_retriever_retry_is_bounded_and_auditable(self) -> None:
        attempts = []
        observed = []

        def operation():
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise RuntimeError(f"transient-{len(attempts)}")
            return "ok"

        with patch.object(runner.time, "sleep") as sleeper:
            result, used, errors = runner.retry_operation(
                operation,
                max_attempts=3,
                retry_delays_seconds=(5, 15),
                on_error=lambda attempt, exc, error: observed.append(
                    (attempt, type(exc).__name__, error["message"])
                ),
            )
        self.assertEqual(result, "ok")
        self.assertEqual(used, 3)
        self.assertEqual(len(errors), 2)
        self.assertEqual(observed[0], (1, "RuntimeError", "transient-1"))
        self.assertEqual([call.args[0] for call in sleeper.call_args_list], [5.0, 15.0])

    def test_token_preflight_has_no_truncation(self) -> None:
        rows, _ = runner.read_input_rows(self.config, 2)
        selected = [
            next(case for case in self.cases if case.source_id == source)
            for source in (
                "finqa_flan_z", "finqa_mistral_z", "finqa_t5gemma2_z",
                "qwen3_6", "llama4",
            )
        ]
        report = runner.token_preflight(rows, selected, self.config)
        for family in ("flan", "mistral", "t5gemma2"):
            self.assertFalse(report["families"][family]["truncationAllowed"])
            self.assertLessEqual(
                report["families"][family]["maxObserved"],
                report["families"][family]["maxAllowed"],
            )
        for source_id in ("qwen3_6", "llama4"):
            direct = report["directBinding"][source_id]
            self.assertFalse(direct["truncationAllowed"])
            self.assertLessEqual(direct["maxObserved"], direct["maxInputAllowed"])
            self.assertLessEqual(direct["maxPromptPlusCompletion"], direct["contextWindow"])
    def test_seq2seq_applies_experiment_run_seed(self) -> None:
        with patch.dict(
            seq2seq_retriever.os.environ,
            {"EXPERIMENT6_RUN_SEED": "2026073107"},
            clear=False,
        ):
            with patch.object(seq2seq_retriever, "set_seed") as mocked:
                self.assertEqual(
                    seq2seq_retriever.apply_experiment_run_seed(),
                    2026073107,
                )
                mocked.assert_called_once_with(2026073107)

    def test_t5gemma_cache_fallback_is_thresholded_without_truncation(self) -> None:
        threshold = self.config["retriever"]["generationCache"]["t5gemma2"][
            "disableAboveInputTokens"
        ]
        self.assertEqual(
            seq2seq_retriever.generation_cache_kwargs(threshold, threshold),
            {},
        )
        self.assertEqual(
            seq2seq_retriever.generation_cache_kwargs(threshold + 1, threshold),
            {"use_cache": False},
        )
        self.assertEqual(
            seq2seq_retriever.generation_cache_kwargs(100000, 0),
            {},
        )

    def test_retriever_candidate_checkpoint_is_complete_and_hash_guarded(self) -> None:
        rows = [minimal_row()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retriever_candidates.jsonl"
            candidate = "Revenue increased."
            runner.write_jsonl(path, [{
                "index": 0,
                "source": "S1",
                "run": 3,
                "seed": 2026073103,
                "raw": '{"RetFact":"Revenue increased."}',
                "candidate": candidate,
                "candidateSha256": runner.sha256_text(candidate),
            }])
            loaded = runner.load_retriever_candidate_checkpoint(path, rows, 3, 2026073103)
            self.assertEqual(
                loaded,
                (['{"RetFact":"Revenue increased."}'], [candidate]),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["candidateSha256"] = "bad"
            runner.write_jsonl(path, [payload])
            self.assertIsNone(
                runner.load_retriever_candidate_checkpoint(path, rows, 3, 2026073103)
            )

            payload["candidateSha256"] = runner.sha256_text(candidate)
            payload["seed"] = 2026073199
            runner.write_jsonl(path, [payload])
            self.assertIsNone(
                runner.load_retriever_candidate_checkpoint(path, rows, 3, 2026073103)
            )

    def test_resumed_t5gemma_runtime_preserves_seed_and_route(self) -> None:
        case = runner.MatrixCase(
            "6_t5gemma2_z", "finqa_t5gemma2_z", "zero-shot",
            "adapter-converter", 1, True,
        )
        runtime = runner.resumed_retriever_runtime(
            case,
            "t5gemma2",
            self.config,
            2026073101,
            Path("/tmp/candidates.jsonl"),
        )
        self.assertEqual(runtime["actualModel"], "google/t5gemma-2-1b-1b")
        self.assertTrue(runtime["raw"]["candidate_checkpoint_reused"])
        self.assertEqual(runtime["raw"]["run_seed"], 2026073101)
        self.assertEqual(runtime["raw"]["max_new_tokens"], 128)
        self.assertEqual(runtime["raw"]["batch_size"], 1)
        self.assertEqual(runtime["runtimeProfile"], "t5gemma2-canonical-batch1")

    def test_parallel_atomic_json_writes_leave_valid_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.json"
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(
                    lambda index: runner.write_json(path, {"index": index}),
                    range(100),
                ))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn(payload["index"], range(100))
            self.assertEqual(list(path.parent.glob("shared.json.tmp.*")), [])

    def test_retriever_verifier_expands_exact_thirty_cases(self) -> None:
        cases = retriever_verifier.expand_retriever_cases(self.config)
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({item["outputId"] for item in cases}), 30)
        self.assertEqual(
            {
                family: sum(
                    retriever_verifier.family_for_source(item["sourceId"]) == family
                    for item in cases
                )
                for family in ("flan", "mistral", "t5gemma2")
            },
            {"flan": 10, "mistral": 10, "t5gemma2": 10},
        )


    def test_coordinators_gate_migration_and_full_evaluation(self) -> None:
        priority = (Path(__file__).resolve().parent / "experiment_6_narrative2_retriever_priority.sh").read_text()
        local_queue = (Path(__file__).resolve().parent / "experiment_6_narrative2_local_queue.sh").read_text()
        self.assertLess(
            priority.index("--require-complete --apply"),
            priority.index("write_state verification running"),
        )
        self.assertIn(".plannedRuns == 300", priority)
        self.assertIn('wait_for_rc "$runtime_dir/formal_gpt4.rc"', local_queue)
        self.assertIn(
            'GENERATOR_RESPONSE_FORMAT="$llama_response_format"', local_queue
        )
        self.assertIn(
            'GENERATOR_RESPONSE_SCHEMA_PATH="$llama_response_schema"', local_queue
        )
        self.assertIn(
            'GENERATOR_RESPONSE_FORMAT="$mistral_response_format"', local_queue
        )
        self.assertEqual(local_queue.count("--smoke-only --no-resume"), 1)
        self.assertIn(
            "--smoke-only --limit 2 --no-resume --source-id llama4",
            local_queue,
        )
        self.assertIn("verify_smoke_case 6_llama_z llama4 2", local_queue)
        self.assertIn("EXPERIMENT6_REUSE_LLAMA_SERVER", local_queue)
        self.assertIn("verify_smoke_case 6_mistral4_z mistral4", local_queue)
        self.assertIn(
            '.response.requestResponseFormat == "json_schema"', local_queue
        )
        for variable in ("controls_rc", "gpt55_rc", "gpt53_rc", "gpt4_rc"):
            self.assertIn(f'"${variable}" != "0"', local_queue)
        self.assertIn(".coverage.officialCaseRunsComplete == 540", local_queue)
        self.assertIn(".coverage.controlCaseRunsComplete == 40", local_queue)
        self.assertIn(".coverage.manifestCount == 580", local_queue)
        self.assertLess(
            local_queue.index("completed_ready_for_ranking"),
            local_queue.index("dist/evaluate_narrative2_fixed_v2.py"),
        )
        fixed_coordinator = (
            runner.REPO_ROOT
            / "dist"
            / "experiment_6_narrative2_fixed_v2_full.sh"
        ).read_text()
        self.assertIn("9_cases,90_case_runs,7650_predictions", fixed_coordinator)
        self.assertIn(
            "protocol=narrative2-fixed-python-v2", fixed_coordinator
        )
        self.assertIn(
            "development_partial_no_ranking", fixed_coordinator
        )
        self.assertLess(
            fixed_coordinator.index("part1_evaluation completed"),
            fixed_coordinator.index("remaining_matrix scheduling"),
        )



class ReferenceAlignedV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = runner.load_config(CONFIG_PATH)
        cls.cases = runner.expand_matrix(cls.config)

    def test_method_revision_and_ontology_are_versioned(self) -> None:
        method = evaluator_v5.method_metadata()
        expected_ontology_sha = evaluator_v5.v4.sha256_text(json.dumps(
            evaluator_v5.TREND_CLASS_ALIASES,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ))
        self.assertEqual(
            evaluator_v5.PROTOCOL,
            "narrative2-reference-aligned-hybrid-v5.1",
        )
        self.assertEqual(
            method["revision"],
            "evidence-gated-strict-json-schema-percent-dual-20260812",
        )
        self.assertEqual(method["trendOntologySha256"], expected_ontology_sha)
        self.assertEqual(len(method["methodCompatibilitySha256"]), 64)
        self.assertIn("evidenceValid=true", method["semanticEvidenceGate"])
        self.assertEqual(
            method["primaryScoreRole"],
            "diagnostic_only_not_formal_ranking",
        )
        self.assertEqual(
            method["thirdAdjudicationRole"],
            "audit_only_not_score_mutation",
        )

    def test_markdown_report_exposes_method_identity_and_role(self) -> None:
        rendered = evaluator_v5.markdown_report({
            "protocol": evaluator_v5.PROTOCOL,
            "status": "completed",
            "method": evaluator_v5.method_metadata(),
            "completedCases": 0,
            "formalPredictions": 0,
            "judge": {"model": "gpt-5.5", "reasoningEffort": "medium"},
            "cases": [],
        })
        self.assertIn("hybrid-v5.1", rendered)
        self.assertIn(evaluator_v5.METHOD_REVISION, rendered)
        self.assertIn(evaluator_v5.TREND_ONTOLOGY_SHA256, rendered)
        self.assertIn("diagnostic_only_not_formal_ranking", rendered)
        self.assertIn("evidenceValid=true", rendered)

    def test_evaluation_contract_locks_judge_and_audit_policy(self) -> None:
        evaluation = {
            "fields": list(evaluator_v5.FIELDS),
            "hardFields": ["DataName", "Position"],
            "semanticFields": ["ObjectName", "Trend", "Text"],
            "numericField": "Num",
            "failurePolicy": "rejected-zero",
            "judge": {
                "model": "gpt-5.5",
                "reasoningEffort": "medium",
                "minimumConfidence": 0.8,
                "requireEvidenceSpans": True,
                "blindCaseAndModel": True,
                "randomizeAB": True,
                "requestTimeoutSeconds": 300,
                "maxAttempts": 3,
                "retryDelaysSeconds": [5, 15],
            },
            "audit": {
                "sampleRate": 0.1,
                "swapAB": True,
                "thirdAdjudicationOnDisagreement": True,
            },
        }
        evaluator_v5.validate_evaluation_contract(evaluation)
        for section, name, invalid in (
            ("judge", "model", "gpt-4.1"),
            ("judge", "reasoningEffort", "low"),
            ("judge", "minimumConfidence", 0.7),
            ("audit", "sampleRate", 0.2),
        ):
            with self.subTest(section=section, name=name):
                changed = copy.deepcopy(evaluation)
                changed[section][name] = invalid
                with self.assertRaisesRegex(
                    evaluator_v5.ProtocolError,
                    "evaluation contract mismatch",
                ):
                    evaluator_v5.validate_evaluation_contract(changed)

    def test_route_comparison_rejects_incompatible_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text(json.dumps({
                "protocol": "narrative2-reference-aligned-hybrid-v5",
                "method": {"methodCompatibilitySha256": "wrong"},
                "cases": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                evaluator_v5.ProtocolError, "protocol mismatch"
            ):
                evaluator_v5.comparison_with([], report_path)
            compatible = {
                "protocol": evaluator_v5.PROTOCOL,
                "method": evaluator_v5.method_metadata(),
                "cases": [],
            }
            report_path.write_text(
                json.dumps(compatible), encoding="utf-8"
            )
            result = evaluator_v5.comparison_with([], report_path)
            self.assertEqual(result["overlappingCases"], 0)

    def test_write_tables_emits_field_and_ablation_scorecards(self) -> None:
        statistic = {"mean": 0.5, "sampleSd": 0.1, "min": 0.4, "max": 0.6}
        field_statistics = {
            field: {
                name: dict(statistic)
                for name in ("precision", "recall", "f1")
            }
            for field in evaluator_v5.FIELDS
        }
        pooled_field = {
            "tp": 1, "fp": 1, "fn": 1,
            "precision": 0.5, "recall": 0.5, "f1": 0.5,
        }
        stage = {
            "fields": field_statistics,
            "macro": {
                name: dict(statistic)
                for name in ("precision", "recall", "f1")
            },
            "micro": {
                name: dict(statistic)
                for name in ("precision", "recall", "f1")
            },
            "pooled": {
                "fields": {
                    field: dict(pooled_field)
                    for field in evaluator_v5.FIELDS
                },
                "micro": dict(pooled_field),
            },
        }
        case = {
            "outputId": "case",
            "sourceId": "source",
            "declaredRoute": "direct-binding",
            "effectiveRoute": "direct-binding",
            "strictSchemaValidity": dict(statistic),
            "coverage": {
                "goldBindings": 2, "predictedBindings": 2,
                "matchedBindings": 1, "anchorPrecision": 0.5,
                "anchorRecall": 0.5,
            },
            "primary": stage,
            "conditionalContent": {"f1": dict(statistic)},
            "ablations": {
                "frozen_hybrid_v4": None,
                "semantic_gpt55_medium": stage,
            },
            "runResults": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = evaluator_v5.write_tables(Path(directory), [case])
            field_rows = list(csv.DictReader(
                Path(paths["fieldScorecard"]).read_text(
                    encoding="utf-8"
                ).splitlines(),
                delimiter="\t",
            ))
            ablation_rows = list(csv.DictReader(
                Path(paths["ablationScorecard"]).read_text(
                    encoding="utf-8"
                ).splitlines(),
                delimiter="\t",
            ))
        self.assertEqual(len(field_rows), len(evaluator_v5.FIELDS))
        self.assertEqual({row["field"] for row in field_rows}, set(evaluator_v5.FIELDS))
        self.assertEqual(len(ablation_rows), 2)
        self.assertEqual(
            {row["available"] for row in ablation_rows}, {"True", "False"}
        )

    @staticmethod
    def binding(
        data_name: str = "Revenue",
        row: int = 0,
        column: int = 1,
        *,
        object_name: object = None,
        trend: object = "increase",
        num: object = None,
        text: object = "Revenue increased to 185 in 2024.",
    ) -> dict[str, object]:
        return {
            "ObjectName": (
                ["Revenue"] if object_name is None else object_name
            ),
            "DataName": data_name,
            "Position": [{
                "Begin": [row, column],
                "End": [row, column],
            }],
            "Trend": trend,
            "Num": [185] if num is None else num,
            "Text": text,
        }

    def test_alignment_handles_permutation_missing_extra_and_duplicate_anchor(self) -> None:
        gold = [
            self.binding("A", 0, 0),
            self.binding("B", 1, 1),
        ]
        prediction = [
            self.binding("B", 1, 1),
            self.binding("A", 0, 0),
            self.binding("A", 0, 0),
            self.binding("C", 2, 2),
        ]
        alignment = evaluator_v5.align_bindings(gold, prediction)
        self.assertEqual(
            alignment["matches"],
            [
                {"goldIndex": 1, "predictionIndex": 0},
                {"goldIndex": 0, "predictionIndex": 1},
            ],
        )
        self.assertEqual(alignment["unmatchedGold"], [])
        self.assertEqual(alignment["unmatchedPrediction"], [2, 3])

    def test_econ044_partial_binding_keeps_five_fields_tp(self) -> None:
        correct = self.binding(
            "Robocalls",
            0,
            1,
            object_name=["Robocalls"],
            trend="increase",
            num=[185],
            text="Robocalls increased to 185 in January 2014.",
        )
        omitted = self.binding(
            "Robocalls",
            1,
            1,
            object_name=["Robocalls"],
            trend="increase",
            num=[360],
            text="Robocalls later increased to 360.",
        )
        predicted = copy.deepcopy(correct)
        predicted["ObjectName"] = ["January 2014"]
        alignment = evaluator_v5.align_bindings(
            [correct, omitted], [predicted]
        )
        match = alignment["matches"][0]
        plan = evaluator_v5.build_semantic_plan(
            "", correct, predicted, 0
        )
        passes = evaluator_v5.semantic_passes(
            correct, predicted, plan, {}, 0.8
        )
        counts = evaluator_v5.zero_counts()
        evaluator_v5.add_aligned_pair(counts, passes, predicted)
        evaluator_v5.add_unmatched_gold(counts)
        self.assertEqual(counts["ObjectName"], {"tp": 0, "fp": 1, "fn": 2})
        for field in ("DataName", "Position", "Trend", "Num", "Text"):
            self.assertEqual(
                counts[field], {"tp": 1, "fp": 0, "fn": 1},
                field,
            )
        self.assertEqual(match, {"goldIndex": 0, "predictionIndex": 0})

    def test_invalid_single_field_does_not_cascade(self) -> None:
        gold = self.binding()
        prediction = self.binding(num=["185"])
        plan = evaluator_v5.build_semantic_plan(
            "", gold, prediction, 0
        )
        passes = evaluator_v5.semantic_passes(
            gold, prediction, plan, {}, 0.8
        )
        counts = evaluator_v5.zero_counts()
        evaluator_v5.add_aligned_pair(counts, passes, prediction)
        self.assertEqual(counts["Num"], {"tp": 0, "fp": 0, "fn": 1})
        for field in ("ObjectName", "DataName", "Position", "Trend", "Text"):
            self.assertEqual(
                counts[field], {"tp": 1, "fp": 0, "fn": 0},
                field,
            )

    def test_object_coreference_trend_classes_and_data_name_substitution(self) -> None:
        gold = self.binding(
            data_name="Value",
            object_name=["United States"],
            trend="inched up",
        )
        prediction = self.binding(
            data_name="Value",
            object_name=["the country"],
            trend="upgrade trend",
        )
        plan = evaluator_v5.build_semantic_plan(
            "", gold, prediction, 0
        )
        judgment = {
            plan["objectDecisionId"]: {
                "equivalent": True,
                "matchedPairs": [{"goldIndex": 0, "predictionIndex": 0}],
                "confidence": 0.85,
                "evidenceSpan": "the country",
                "evidenceValid": True,
            }
        }
        self.assertTrue(
            evaluator_v5.object_semantic_equal(plan, judgment, 0.8)
        )
        self.assertEqual(
            evaluator_v5.trend_class("expanded"), "increase"
        )
        self.assertEqual(
            evaluator_v5.trend_class("plunges"), "decrease"
        )
        substitution = self.binding(
            data_name="Value",
            object_name=["Value"],
            trend="increase",
        )
        blocked = evaluator_v5.build_semantic_plan(
            "", gold, substitution, 0
        )
        self.assertTrue(blocked["objectDataNameSubstitution"])
        self.assertFalse(
            evaluator_v5.object_semantic_equal(blocked, judgment, 0.8)
        )


    def test_semantic_judge_requires_verbatim_evidence_in_source(self) -> None:
        judgment = {
            "equivalent": True,
            "matchedPairs": [],
            "confidence": 0.95,
            "evidenceSpan": "invented evidence",
            "evidenceValid": False,
        }
        self.assertFalse(
            evaluator_v5.decision_accepts(judgment, 0.8, "Text")
        )
        judgment["evidenceValid"] = True
        self.assertTrue(
            evaluator_v5.decision_accepts(judgment, 0.8, "Text")
        )

    def test_num_main_requires_finite_json_number_array(self) -> None:
        for invalid in (None, "None", 12, ["None"], [True], [float("nan")]):
            with self.subTest(invalid=invalid):
                self.assertFalse(evaluator_v5.field_type_valid("Num", invalid))
                self.assertFalse(evaluator_v5.strict_numeric_equal([], invalid))
        self.assertTrue(evaluator_v5.field_type_valid("Num", []))
        self.assertTrue(evaluator_v5.field_type_valid("Num", [12, 13.5]))
        self.assertTrue(evaluator_v5.strict_numeric_equal([], []))
        schema_invalid_absence = self.binding(num=[])
        schema_invalid_absence["Num"] = None
        passed, schema_invalid = evaluator_v5.numeric_sensitivity_equal(
            self.binding(num=[]),
            schema_invalid_absence,
            0.01,
            allow_units=True,
        )
        self.assertTrue(passed)
        self.assertTrue(schema_invalid)

    def test_prediction_result_must_be_array_for_schema_validity(self) -> None:
        self.assertEqual(evaluator_v5.prediction_schema_errors([]), [])
        self.assertEqual(
            evaluator_v5.prediction_schema_errors({}),
            [{"predictionIndex": None, "errors": ["result_not_array"]}],
        )

    def test_trend_absence_requires_json_string_schema(self) -> None:
        gold = self.binding(trend="None")
        for invalid in (None, []):
            with self.subTest(invalid=invalid):
                prediction = self.binding(trend=invalid)
                self.assertFalse(
                    evaluator_v5.field_type_valid("Trend", invalid)
                )
                plan = evaluator_v5.build_semantic_plan(
                    "", gold, prediction, 0
                )
                self.assertFalse(plan["trendDeterministic"])
                self.assertFalse(
                    evaluator_v5.deterministic_passes(
                        gold, prediction, normalized=True
                    )["Trend"]
                )
                self.assertFalse(
                    evaluator_v5.semantic_passes(
                        gold, prediction, plan, {}, 0.8
                    )["Trend"]
                )

    def test_numeric_unit_sensitivity_recovers_but_schema_stays_invalid(self) -> None:
        gold = self.binding(
            num=[1_000_000],
            text="Revenue was one million dollars.",
        )
        prediction = self.binding(
            num=["1 million"],
            text="Revenue was 1 million dollars.",
        )
        passed, schema_invalid = evaluator_v5.numeric_sensitivity_equal(
            gold, prediction, 0.01, allow_units=True
        )
        self.assertTrue(passed)
        self.assertTrue(schema_invalid)
        self.assertFalse(
            evaluator_v5.field_type_valid("Num", prediction["Num"])
        )
        self.assertFalse(
            evaluator_v5.numeric_sensitivity_equal(
                gold, prediction, 0.01, allow_units=False
            )[0]
        )

    def test_numeric_percent_sensitivity_supports_ratio_and_point_conventions(self) -> None:
        prediction = self.binding(
            num=["12 percent"], text="Revenue increased by 12%."
        )
        for gold_num in ([0.12], [12]):
            with self.subTest(gold_num=gold_num):
                gold = self.binding(
                    num=gold_num, text="Revenue increased by 12%."
                )
                passed, schema_invalid = evaluator_v5.numeric_sensitivity_equal(
                    gold, prediction, 0.01, allow_units=True
                )
                self.assertTrue(passed)
                self.assertTrue(schema_invalid)

    def test_text_token_f1_preserves_numbers_dates_percent_and_polarity(self) -> None:
        detail = evaluator_v5.token_f1(
            "Revenue did not fall by -5% on 2024-01-31.",
            "Revenue fell by 5% in 2024.",
        )
        self.assertIn("-5%", detail["missingProtectedTokens"])
        self.assertIn("2024-01-31", detail["missingProtectedTokens"])
        self.assertLess(detail["f1"], 1.0)

    def test_zero_binding_identity_and_tp_fp_fn_recompute(self) -> None:
        alignment = evaluator_v5.align_bindings([], [])
        self.assertEqual(
            alignment,
            {
                "matches": [],
                "unmatchedGold": [],
                "unmatchedPrediction": [],
            },
        )
        scored = evaluator_v5.fields_metrics(
            evaluator_v5.zero_counts()
        )
        self.assertEqual(scored["macro"]["precision"], 1.0)
        self.assertEqual(scored["micro"]["f1"], 1.0)

    def test_tp_fp_fn_macro_and_pooled_micro_recompute(self) -> None:
        counts = evaluator_v5.zero_counts()
        prediction = self.binding()
        passes = {field: True for field in evaluator_v5.FIELDS}
        passes["Text"] = False
        evaluator_v5.add_aligned_pair(counts, passes, prediction)
        evaluator_v5.add_unmatched_gold(counts)
        evaluator_v5.add_unmatched_prediction(counts, prediction)
        evaluator_v5.add_unmatched_prediction(counts, prediction)
        for field in evaluator_v5.FIELDS:
            expected = (
                {"tp": 0, "fp": 3, "fn": 2}
                if field == "Text"
                else {"tp": 1, "fp": 2, "fn": 1}
            )
            self.assertEqual(counts[field], expected, field)
        scored = evaluator_v5.fields_metrics(counts)
        manual_field_f1 = [
            evaluator_v5.metric(**counts[field])["f1"]
            for field in evaluator_v5.FIELDS
        ]
        self.assertAlmostEqual(
            scored["macro"]["f1"],
            sum(manual_field_f1) / len(manual_field_f1),
        )
        pooled = {
            name: sum(counts[field][name] for field in evaluator_v5.FIELDS)
            for name in ("tp", "fp", "fn")
        }
        self.assertEqual(
            scored["micro"], evaluator_v5.metric(**pooled)
        )

    def test_conditional_content_is_undefined_without_matched_bindings(self) -> None:
        runs = [{
            "run": run,
            "conditionalContent": {
                "defined": False,
                "macro": {"f1": 1.0},
            },
        } for run in range(1, 11)]
        summary = evaluator_v5.summarize_conditional_runs(
            runs,
            lambda item: item["conditionalContent"]["macro"]["f1"],
        )
        self.assertIsNone(summary["mean"])
        self.assertIsNone(summary["top1"])
        self.assertEqual(summary["eligibleRuns"], 0)
        self.assertEqual(summary["totalRuns"], 10)

    def test_direct_diagnostic_run_bypasses_converter_and_adapter(self) -> None:
        case = next(
            case for case in self.cases
            if case.output_id == "6_mistral_base_d"
        )
        valid = json.dumps({
            "result": [self.binding()],
            "reason": "direct",
        })
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
            captured.update({
                "legacyCase": legacy_case,
                "csvPath": csv_path,
                "promptMode": prompt_mode,
                "args": args,
                "useAdapter": use_adapter,
                "familyOverride": family_override,
                "rawSuffix": raw_suffix,
            })
            return [valid], {
                "family": "mistral",
                "actual_engine": case.source_id,
                "adapter_dir": None,
                "use_adapter": False,
                "structured_output": args.structured_output,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_tokens,
            }

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with patch.object(
                runner.legacy,
                "run_retriever_case",
                side_effect=fake_native,
            ), patch.object(
                runner,
                "run_converter",
                side_effect=AssertionError("converter must not run"),
            ):
                manifest = runner.run_case_once(
                    case=case,
                    rows=[minimal_row()],
                    run_number=1,
                    run_seed=2026073101,
                    output_root=output_root,
                    config=self.config,
                    resume=False,
                    base_route_mode="direct-diagnostic",
                )
            predictions = evaluator_v5.read_jsonl(
                output_root
                / "cases"
                / case.output_id
                / "run_01"
                / "predictions.jsonl"
            )
        self.assertEqual(manifest["effectiveRoute"], "direct-diagnostic-native")
        self.assertEqual(manifest["baseRouteMode"], "direct-diagnostic")
        self.assertIsNone(manifest["converterModel"])
        self.assertIsNone(manifest["adapter"])
        self.assertFalse(captured["useAdapter"])
        self.assertEqual(captured["familyOverride"], "mistral")
        self.assertEqual(captured["rawSuffix"], ".direct_diagnostic")
        namespace = captured["args"]
        self.assertEqual(namespace.structured_output, "off")
        self.assertFalse(namespace.append_label_descriptions)
        self.assertEqual(predictions[0]["result"][0]["Num"], [185])
        self.assertEqual(
            runner.effective_route(case, "historical"),
            "retriever-converter",
        )

    def test_native_mistral_command_disables_historical_suffix_only_when_requested(self) -> None:
        case = runner.legacy.MatrixCase(
            "case", "mistral_v0_3", "narrative_dynamic_shot"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "input.csv"
            csv_path.write_text(
                "input,Rel_Fact,Source\nprompt,__BLINDED__,S1\n",
                encoding="utf-8",
            )
            (root / "raw").mkdir()
            namespace = runner.native_direct_namespace(
                root, self.config, 1, "mistral"
            )
            self.assertEqual(namespace.t5gemma_cache_safe_input_tokens, 4096)
            captured: list[str] = []

            def fake_run(command, log_path, timeout, cuda):
                del log_path, timeout, cuda
                captured.extend(command)
                output = Path(command[command.index("--output-txt") + 1])
                output.write_text(
                    'True: x Pred: {"result":[],"reason":"none"}\n',
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with patch.object(
                runner.legacy, "run_command", side_effect=fake_run
            ):
                predictions, runtime = runner.legacy.run_retriever_case(
                    case,
                    csv_path,
                    "dynamic-shot",
                    namespace,
                    use_adapter=False,
                )
        self.assertEqual(len(predictions), 1)
        self.assertEqual(
            captured[
                captured.index("--append-label-descriptions") + 1
            ],
            "false",
        )
        self.assertEqual(
            captured[captured.index("--structured-output") + 1],
            "off",
        )
        self.assertEqual(runtime["adapter_dir"], None)

    def test_v5_relocates_historical_generation_artifacts_by_manifest_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "cases" / "case_a" / "run_01"
            run_dir.mkdir(parents=True)
            prediction_path = run_dir / "predictions.jsonl"
            candidate_path = run_dir / "retriever_candidates.jsonl"
            prediction_path.write_text(
                json.dumps({"source": "S1", "result": []}) + "\n",
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps({"source": "S1", "candidate": "fact"}) + "\n",
                encoding="utf-8",
            )
            auxiliary_names = (
                "rawResponse", "prompts", "runtime", "formatReport"
            )
            for name in auxiliary_names:
                (run_dir / f"{name}.jsonl").write_text(
                    "safe\n", encoding="utf-8"
                )
            manifest = {
                "outputId": "case_a",
                "run": 1,
                "expectedRows": 1,
                "files": {
                    "predictions": "/tmp/previous/predictions.jsonl",
                    "retrieverCandidates": (
                        "/tmp/previous/retriever_candidates.jsonl"
                    ),
                },
                "hashes": {
                    "predictions": evaluator_v5.v4.sha256_file(
                        prediction_path
                    ),
                    "retrieverCandidates": evaluator_v5.v4.sha256_file(
                        candidate_path
                    ),
                },
            }
            for name in auxiliary_names:
                artifact_path = run_dir / f"{name}.jsonl"
                manifest["files"][name] = f"/tmp/previous/{artifact_path.name}"
                manifest["hashes"][name] = evaluator_v5.v4.sha256_file(
                    artifact_path
                )
            predictions, candidates = evaluator_v5.load_prediction_records(
                manifest, [{"source": "S1"}], root
            )
            contract = evaluator_v5.validate_run_contract(
                manifest, predictions, [{"source": "S1"}], root
            )
        self.assertEqual(predictions[0]["source"], "S1")
        self.assertEqual(candidates["S1"]["candidate"], "fact")
        self.assertEqual(contract["status"], "passed")

if __name__ == "__main__":
    unittest.main()
