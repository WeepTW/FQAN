"""Unified runtime entry point for the FQAN research workflow.

The entry point stays thin: it validates workflow contracts, resolves artifact
routes, records stage plans, and delegates heavy work to existing scripts only
when --execute is explicitly set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from new_full_finqa_run import resolve_engine
from result_organization import build_match_plan, validate_plan as validate_match_plan
from retriever_json_schema import schema_required


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent
ARGS_PATH = WORKSPACE_ROOT / "src" / "args.json"
EXPERIMENT_ROOT = REPO_ROOT / "Experiment"


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

PROMPT_ALIASES = {
    "raw": "raw",
    "raw_finqa": "raw",
    "raw-finqa": "raw",
    "finqa_raw": "raw",
    "finqa-raw": "raw",
    "original": "original",
    "zero-shot": "zero-shot",
    "zero_shot": "zero-shot",
    "many-shot": "many-shot",
    "many_shot": "many-shot",
    "dynamic-shot": "dynamic-shot",
    "dynamic_shot": "dynamic-shot",
}

PROMPT_CONTRACT_KEYS = {
    "raw": ["raw", "original"],
    "original": ["original"],
    "zero-shot": ["new_prompt_zero_shot"],
    "many-shot": ["new_prompt_many_shot", "new_prompt_few_shot"],
    "dynamic-shot": ["new_prompt_dynamic_shot"],
}

PROMPT_RELFACT_CSVS = {
    "raw": WORKSPACE_ROOT / "data" / "src" / "FINDER" / "finqa_test_rel_fact_instruction.csv",
    "original": WORKSPACE_ROOT / "data" / "finqa_original" / "finqa_test_rel_fact_instruction.csv",
    "zero-shot": WORKSPACE_ROOT / "data" / "finqa_zero_shot" / "finqa_test_rel_fact_instruction.csv",
    "many-shot": WORKSPACE_ROOT / "data" / "finqa_many_shot" / "finqa_test_rel_fact_instruction.csv",
    "dynamic-shot": WORKSPACE_ROOT / "data" / "finqa_dynamic_shot" / "finqa_test_rel_fact_instruction.csv",
}
MATCH_EMBED_BATCH_SIZE = int(os.environ.get("MATCH_EMBED_BATCH_SIZE", "256"))
DATA_JSON = first_existing_path(
    REPO_ROOT / "Data_Target_Module" / "Finqa" / "finqa_test_with_table_text.json",
    WORKSPACE_ROOT / "src" / "code" / "Data" / "Data_Target_Module" / "Finqa" / "finqa_test_with_table_text.json",
    WORKSPACE_ROOT / "src" / "code" / "Data-FINDER" / "Data_Target_Module" / "Finqa" / "finqa_test_with_table_text.json",
)

PROMPT_SUFFIXES = {
    "raw": "r",
    "original": "o",
    "zero-shot": "z",
    "many-shot": "m",
    "dynamic-shot": "d",
}

CANONICAL_RETRIEVER_EXPERIMENT_PREFIXES = {
    "flan_t5_large": "finqa_flan",
    "mistral_v0_3": "finqa_mistral",
    "t5gemma_2_1b_1b": "finqa_t5gemma2",
}

LEGACY_RETRIEVER_EXPERIMENT_PREFIXES = {
    "mistral_v0_3": ("finqa_Mistral",),
}

RETRIEVER_EXPERIMENT_SCRIPTS = {
    "flan_t5_large": REPO_ROOT / "dist" / "experiment_2_flan_retriever.sh",
    "mistral_v0_3": REPO_ROOT / "dist" / "experiment_1_mistral_retriever.sh",
    "t5gemma_2_1b_1b": REPO_ROOT / "dist" / "experiment_3_t5gemma_retriever.sh",
}

RETRIEVER_ALIASES = {
    "flan": "flan_t5_large",
    "flan_t5_large": "flan_t5_large",
    "mistral": "mistral_v0_3",
    "mistral_v0_3": "mistral_v0_3",
    "t5gemma": "t5gemma_2_1b_1b",
    "t5gemma_2_1b_1b": "t5gemma_2_1b_1b",
    "t5gemma-2-1b-1b": "t5gemma_2_1b_1b",
    "apollo": "apollo",
}

GENERATOR_ALIASES = {
    "deepseek": "deepseek_r1_qwen32b",
    "deepseek_r1_qwen32b": "deepseek_r1_qwen32b",
    "deepseek-r1-qwen32b": "deepseek_r1_qwen32b",
    "mistral4": "mistral4",
    "qwen3_6": "qwen3_6",
    "llama3_3": "llama3_3",
    "llama-3.3": "llama3_3",
    "llama3.3": "llama3_3",
    "llama3_3_70b": "llama3_3",
    "llama4": "llama4",
    "llama4_scout": "llama4",
    "llama-4-scout-17b-16e-instruct": "llama4",
    "qwythos": "qwythos9b",
    "qwythos9b": "qwythos9b",
    "qwythos-9b": "qwythos9b",
    "gpt-4.1": "gpt4_1",
    "gpt4.1": "gpt4_1",
    "gpt4_1": "gpt4_1",
    "gpt-4": "gpt4_1",
    "gpt4": "gpt4_1",
    "gpt5_3_codexS": "gpt5_3_codexS",
    "gpt-5.5": "gpt5_5",
    "gpt5_5": "gpt5_5",
}

MATCHED_OUTPUT_FALLBACKS = {
    "flan_t5_large": REPO_ROOT
    / "Data_Target_Module"
    / "lora_flan_retriever"
    / "output"
    / "best_matched_with_retrieved_facts_and_questions.json",
    "mistral_v0_3": REPO_ROOT
    / "Data_Target_Module"
    / "mistral_retriever"
    / "output"
    / "best_matched_with_retrieved_facts_and_questions_mistral.json",
    "t5gemma_2_1b_1b": REPO_ROOT
    / "Data_Target_Module"
    / "t5gemma_retriever"
    / "output"
    / "best_matched_with_retrieved_facts_and_questions_t5gemma.json",
    "apollo": REPO_ROOT
    / "Data_Target_Module"
    / "Apollo"
    / "output"
    / "best_matched_with_retrieved_facts_and_questions_apollo.json",
}

RAW_PREDICTION_FALLBACKS = {
    "flan_t5_large": REPO_ROOT
    / "Data_Target_Module"
    / "lora_flan_retriever"
    / "lora_flan_large_corrected_prediction_finqa_rel_fact.txt",
    "mistral_v0_3": REPO_ROOT
    / "Data_Target_Module"
    / "mistral_retriever"
    / "mistral_finqa_rel_fact_file.txt",
    "t5gemma_2_1b_1b": REPO_ROOT / "Data_Target_Module" / "t5gemma_retriever" / "t5gemma_finqa_rel_fact_file.txt",
    "apollo": REPO_ROOT / "Data_Target_Module" / "Apollo" / "finqa_apollo_rel_fact_file.txt",
}


@dataclass
class StageRecord:
    stage: str
    status: str
    command: list[str] | None = None
    cwd: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    message: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize(value: str, aliases: dict[str, str], label: str) -> str:
    key = value.strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported {label}: {value}")
    return aliases[key]


def as_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def prompt_data_dir(prompt_type: str) -> Path:
    normalized = normalize(prompt_type, PROMPT_ALIASES, "prompt type")
    if normalized == "raw":
        return WORKSPACE_ROOT / "data" / "src" / "FINDER"
    if normalized == "original":
        return WORKSPACE_ROOT / "data" / "finqa_original"
    if normalized == "zero-shot":
        return WORKSPACE_ROOT / "data" / "finqa_zero_shot"
    if normalized == "many-shot":
        return WORKSPACE_ROOT / "data" / "finqa_many_shot"
    if normalized == "dynamic-shot":
        return WORKSPACE_ROOT / "data" / "finqa_dynamic_shot"
    raise ValueError(f"Unsupported prompt type: {prompt_type}")


def prompt_csv(prompt_type: str, split: str) -> Path:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unsupported prompt CSV split: {split}")
    return prompt_data_dir(prompt_type) / f"finqa_{split}_rel_fact_instruction.csv"


def materialized_match_data_json(output_json: Path, split: str) -> Path:
    return output_json.with_name(f"finqa_{split}_retriever_data.json")


def build_match_data_command(relfact_csv: Path, data_json: Path) -> list[str]:
    return [
        "python",
        "-B",
        str(REPO_ROOT / "dist" / "build_retriever_few_data_json.py"),
        "--input-csv",
        str(relfact_csv),
        "--output-json",
        str(data_json),
    ]


def canonical_retriever_expt_id(retriever_model: str, prompt_type: str) -> str | None:
    prefix = CANONICAL_RETRIEVER_EXPERIMENT_PREFIXES.get(retriever_model)
    suffix = PROMPT_SUFFIXES.get(prompt_type)
    if not prefix or not suffix:
        return None
    return f"{prefix}_{suffix}"


def canonical_retriever_output_dir(retriever_model: str, prompt_type: str) -> Path | None:
    expt_id = canonical_retriever_expt_id(retriever_model, prompt_type)
    if expt_id is None:
        return None
    return EXPERIMENT_ROOT / expt_id / "retriever" / "outputs"


def legacy_retriever_output_dirs(retriever_model: str, prompt_type: str) -> list[Path]:
    suffix = PROMPT_SUFFIXES.get(prompt_type)
    if not suffix:
        return []
    return [
        EXPERIMENT_ROOT / f"{prefix}_{suffix}" / "retriever" / "outputs"
        for prefix in LEGACY_RETRIEVER_EXPERIMENT_PREFIXES.get(retriever_model, ())
    ]


def canonical_raw_prediction_path(retriever_model: str, prompt_type: str) -> Path | None:
    output_dir = canonical_retriever_output_dir(retriever_model, prompt_type)
    if output_dir is None:
        return None
    return output_dir / "predictions.txt"


def canonical_matched_output_path(retriever_model: str, prompt_type: str) -> Path | None:
    output_dir = canonical_retriever_output_dir(retriever_model, prompt_type)
    if output_dir is None:
        return None
    return output_dir / "best_matched_with_retrieved_facts_and_questions.json"


class Pipeline:
    def __init__(
        self,
        dataset: str,
        expt_id: str,
        parameters: Path = ARGS_PATH,
        convfinqa_enabled: bool = False,
    ) -> None:
        if dataset != "finqa" and not convfinqa_enabled:
            raise ValueError("Only FinQA is active by default; ConvFinQA requires explicit enablement.")
        self.dataset = dataset
        self.expt_id = expt_id
        self.parameters_path = parameters
        self.convfinqa_enabled = convfinqa_enabled
        self.parameters = self.load_parameters(parameters)
        self.experiment_dir = EXPERIMENT_ROOT / expt_id

    @staticmethod
    def load_parameters(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def pipeline_contracts(self) -> dict[str, Any]:
        return self.parameters.get("pipeline_contracts", {})

    def validate_contract(self, split: str = "test") -> list[str]:
        errors = []
        if self.dataset != "finqa":
            errors.append("Only FinQA is enabled in the current thesis workflow.")
        if not self.parameters_path.exists():
            errors.append(f"Missing parameter ledger: {self.parameters_path}")
        contracts = self.pipeline_contracts()
        if not contracts:
            errors.append("docs/args.json has no pipeline_contracts section.")
            return errors
        if "retriever_models" not in contracts:
            errors.append("pipeline_contracts.retriever_models is missing.")
        if "generator_models" not in contracts:
            errors.append("pipeline_contracts.generator_models is missing.")
        if split not in {"train", "dev", "test"}:
            errors.append(f"Unsupported split: {split}.")
        default_paths = contracts.get("default_artifact_paths", {})
        if self.dataset not in default_paths:
            errors.append(f"default_artifact_paths has no dataset entry for {self.dataset}.")
        elif split == "test" and split not in default_paths.get(self.dataset, {}):
            errors.append(f"default_artifact_paths.{self.dataset} has no split entry for {split}.")
        return errors

    def manifest_path(self) -> Path:
        return self.experiment_dir / "run_manifest.json"

    def write_manifest(self, records: list[StageRecord]) -> Path:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": utc_now(),
            "dataset": self.dataset,
            "expt_id": self.expt_id,
            "parameters": str(self.parameters_path),
            "convfinqa_enabled": self.convfinqa_enabled,
            "records": [record.__dict__ for record in records],
        }
        path = self.manifest_path()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def _run(self, command: list[str], cwd: Path) -> StageRecord:
        record = StageRecord(stage="subprocess", status="running", command=command, cwd=str(cwd))
        record.started_at = utc_now()
        proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
        record.finished_at = utc_now()
        record.returncode = proc.returncode
        record.stdout = proc.stdout[-4000:]
        record.stderr = proc.stderr[-4000:]
        record.status = "completed" if proc.returncode == 0 else "failed"
        return record

    def _record_stub(
        self,
        stage: str,
        status: str,
        command: list[str] | None = None,
        cwd: Path | None = None,
        message: str | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> StageRecord:
        now = utc_now()
        return StageRecord(
            stage=stage,
            status=status,
            command=command,
            cwd=str(cwd) if cwd else None,
            started_at=now,
            finished_at=now,
            message=message,
            artifacts=artifacts or {},
        )

    def split_retriever_output_dir(self, retriever_model: str, prompt_type: str, split: str) -> Path:
        return self.experiment_dir / "retriever" / split / "outputs" / retriever_model / prompt_type

    def raw_prediction_candidates(
        self,
        retriever_model: str,
        split: str,
        prompt_type: str = "original",
    ) -> list[Path]:
        contracts = self.pipeline_contracts()
        normalized_prompt = normalize(prompt_type, PROMPT_ALIASES, "prompt type")
        candidates = (
            contracts.get("raw_prediction_artifacts", {})
            .get(self.dataset, {})
            .get(split, {})
            .get(retriever_model, [])
        )
        paths: list[Path] = []
        if split != "test":
            local = self.split_retriever_output_dir(retriever_model, normalized_prompt, split) / "predictions.txt"
            paths.append(local)
            if retriever_model == "t5gemma_2_1b_1b":
                paths.extend([local.with_name("predictions.jsonl"), local.with_name("predictions.json")])
            for candidate in candidates:
                path = as_repo_path(candidate)
                if path not in paths:
                    paths.append(path)
            return paths
        canonical = canonical_raw_prediction_path(retriever_model, normalized_prompt)
        if canonical is not None:
            paths.append(canonical)
            if retriever_model == "t5gemma_2_1b_1b":
                for alternate in (canonical.with_name("predictions.jsonl"), canonical.with_name("predictions.json")):
                    if alternate not in paths:
                        paths.append(alternate)
        for legacy_dir in legacy_retriever_output_dirs(retriever_model, normalized_prompt):
            legacy = legacy_dir / "predictions.txt"
            if legacy not in paths:
                paths.append(legacy)
        for candidate in candidates:
            path = as_repo_path(candidate)
            if path not in paths:
                paths.append(path)
        fallback = RAW_PREDICTION_FALLBACKS.get(retriever_model)
        if fallback and fallback not in paths:
            paths.append(fallback)
        return paths

    def matched_output_candidates(self, retriever_model: str, prompt_type: str, split: str) -> list[Path]:
        contracts = self.pipeline_contracts()
        normalized_prompt = normalize(prompt_type, PROMPT_ALIASES, "prompt type")
        prompt_keys = PROMPT_CONTRACT_KEYS[normalized_prompt]
        prompt_section = (
            contracts.get("default_artifact_paths", {})
            .get(self.dataset, {})
            .get(split, {})
            .get(retriever_model, {})
        )
        paths: list[Path] = []
        if split != "test":
            paths.append(
                self.split_retriever_output_dir(retriever_model, normalized_prompt, split)
                / "best_matched_with_retrieved_facts_and_questions.json"
            )
            for prompt_key in prompt_keys:
                for candidate in prompt_section.get(prompt_key, []):
                    path = as_repo_path(candidate)
                    if path not in paths:
                        paths.append(path)
            return paths
        canonical = canonical_matched_output_path(retriever_model, normalized_prompt)
        if canonical is not None:
            paths.append(canonical)
        for legacy_dir in legacy_retriever_output_dirs(retriever_model, normalized_prompt):
            legacy = legacy_dir / "best_matched_with_retrieved_facts_and_questions.json"
            if legacy not in paths:
                paths.append(legacy)
        for prompt_key in prompt_keys:
            for candidate in prompt_section.get(prompt_key, []):
                path = as_repo_path(candidate)
                if path not in paths:
                    paths.append(path)
        fallback = MATCHED_OUTPUT_FALLBACKS.get(retriever_model)
        if fallback and fallback not in paths:
            paths.append(fallback)
        return paths

    def raw_prediction_path(self, retriever_model: str, split: str, prompt_type: str = "original") -> Path:
        candidates = self.raw_prediction_candidates(retriever_model, split, prompt_type)
        if not candidates:
            raise ValueError(f"No raw prediction route is registered for {retriever_model} on {split}.")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def matched_output_path(self, retriever_model: str, prompt_type: str, split: str) -> Path:
        candidates = self.matched_output_candidates(retriever_model, prompt_type, split)
        if not candidates:
            raise ValueError(
                f"No matched output route is registered for {retriever_model}/{prompt_type} on {split}."
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def generator_output_path(self, generator_model: str) -> Path:
        return self.experiment_dir / f"{self.dataset}_{generator_model}_generated.jsonl"

    def retrieval(
        self,
        model: str,
        prompt_type: str,
        operation: str,
        split: str,
        execute: bool = False,
    ) -> StageRecord:
        retriever = normalize(model, RETRIEVER_ALIASES, "retriever model")
        prompt = normalize(prompt_type, PROMPT_ALIASES, "prompt type")
        raw_prediction = self.raw_prediction_path(retriever, split, prompt)
        artifacts = {
            "split": split,
            "prompt_type": prompt,
            "train_csv": str(prompt_csv(prompt, "train")),
            "dev_csv": str(prompt_csv(prompt, "dev")),
            "test_csv": str(prompt_csv(prompt, "test")),
            "raw_prediction": str(raw_prediction),
            "raw_prediction_candidates": [
                str(path) for path in self.raw_prediction_candidates(retriever, split, prompt)
            ],
        }
        if retriever == "apollo":
            return self._record_stub(
                stage="retrieval",
                status="artifact_only",
                message="APOLLO is a RetFact artifact route in this workflow; no repo-local training command is registered.",
                artifacts=artifacts,
            )
        script = RETRIEVER_EXPERIMENT_SCRIPTS.get(retriever)
        if script is None:
            raise ValueError(f"Unsupported retriever: {retriever}")
        if operation == "train":
            command = [
                "env",
                f"PROMPT_MODES={prompt}",
                "RUN_TRAIN=1",
                "RUN_INFER=0",
                "RUN_MATCH=0",
                "bash",
                str(script),
            ]
        else:
            command = [
                "env",
                f"PROMPT_MODES={prompt}",
                "RUN_TRAIN=0",
                "RUN_INFER=1",
                "RUN_MATCH=0",
                "RUN_PREFLIGHT=0",
                "bash",
                str(script),
            ]
        cwd = REPO_ROOT

        if not execute:
            return self._record_stub(
                stage="retrieval",
                status="planned",
                command=command,
                cwd=cwd,
                message=(
                    "Dry run only. Retriever execution delegates to the canonical experiment_1/2/3 scripts; "
                    "schema assembly and artifact paths follow the same prompt-mode contract."
                ),
                artifacts=artifacts,
            )

        record = self._run(command, cwd)
        record.stage = "retrieval"
        record.message = (
            "Executed retriever route. Prompt type remains a contract-level selector for downstream artifact routing."
        )
        record.artifacts = artifacts
        return record

    def match(
        self,
        retriever_model: str,
        prompt_type: str,
        split: str,
        execute: bool = False,
    ) -> StageRecord:
        retriever = normalize(retriever_model, RETRIEVER_ALIASES, "retriever model")
        prompt = normalize(prompt_type, PROMPT_ALIASES, "prompt type")
        input_txt = self.raw_prediction_path(retriever, split, prompt)
        output_json = self.matched_output_path(retriever, prompt, split)
        relfact_csv = prompt_csv(prompt, split)
        data_json = DATA_JSON
        materialize_data_json = False
        if split != "test" or not DATA_JSON.exists():
            data_json = materialized_match_data_json(output_json, split)
            materialize_data_json = True
        materialize_command = build_match_data_command(relfact_csv, data_json)
        plan = build_match_plan(
            dataset=self.dataset,
            retriever_model=retriever,
            prompt_mode=prompt,
            input_txt=input_txt,
            data_json=data_json,
            relfact_csv=relfact_csv,
            output_json=output_json,
            embedding_batch_size=MATCH_EMBED_BATCH_SIZE,
        )
        command = [
            "python",
            "result_organization.py",
            "match",
            "--dataset",
            self.dataset,
            "--retriever-model",
            retriever,
            "--prompt-mode",
            prompt,
            "--input-txt",
            str(input_txt),
            "--data-json",
            str(data_json),
            "--relfact-csv",
            str(relfact_csv),
            "--embedding-batch-size",
            str(MATCH_EMBED_BATCH_SIZE),
            "--output-json",
            str(output_json),
        ]
        if schema_required(prompt):
            command.append("--require-valid-schema")
        artifacts = {
            "plan": plan.to_dict(),
            "split": split,
            "prompt_type": prompt,
            "data_json_materialized_from_csv": materialize_data_json,
            "data_json_materialization_command": materialize_command if materialize_data_json else None,
            "raw_prediction_candidates": [
                str(path) for path in self.raw_prediction_candidates(retriever, split, prompt)
            ],
            "matched_output_candidates": [
                str(path) for path in self.matched_output_candidates(retriever, prompt, split)
            ],
        }
        if execute and materialize_data_json:
            materialize_record = self._run(materialize_command, REPO_ROOT)
            artifacts["data_json_materialization_record"] = materialize_record.__dict__
            if materialize_record.status != "completed":
                materialize_record.stage = "match"
                materialize_record.message = "Failed to materialize split-specific FinQA data JSON for matching."
                materialize_record.artifacts = artifacts
                return materialize_record
        errors = validate_match_plan(plan)
        if not execute:
            allowed_dry_run_missing = {f"Missing raw prediction file: {input_txt}"}
            if materialize_data_json:
                allowed_dry_run_missing.add(f"Missing FinQA table/text file: {data_json}")
            errors = [error for error in errors if error not in allowed_dry_run_missing]
        if errors:
            return self._record_stub(
                stage="match",
                status="blocked",
                command=command,
                cwd=REPO_ROOT,
                message="; ".join(errors),
                artifacts=artifacts,
            )
        if not execute:
            return self._record_stub(
                stage="match",
                status="planned",
                command=command,
                cwd=REPO_ROOT,
                message="Dry run only. Add --execute to write the matched artifact.",
                artifacts=artifacts,
            )
        command.append("--execute")
        record = self._run(command, REPO_ROOT)
        record.stage = "match"
        record.artifacts = artifacts
        return record

    def generator(
        self,
        generator_model: str,
        input_json: Path,
        execute: bool = False,
        max_tokens: int = 512,
    ) -> StageRecord:
        generator = normalize(generator_model, GENERATOR_ALIASES, "generator model")
        credential_purpose = "execute" if execute else "test"
        config = resolve_engine(generator, credential_purpose=credential_purpose)
        output_jsonl = self.generator_output_path(generator)
        command = [
            "python",
            "new_full_finqa_run.py",
            "--engine",
            generator,
            "--input-json",
            str(input_json),
            "--output-jsonl",
            str(output_jsonl),
            "--max-tokens",
            str(max_tokens),
        ]
        artifacts = {
            "input_json": str(input_json),
            "input_exists": input_json.exists(),
            "output_jsonl": str(output_jsonl),
            "engine": config.to_public_dict(),
            "credential_purpose": credential_purpose,
        }
        if not input_json.exists():
            return self._record_stub(
                stage="generator",
                status="blocked",
                command=command,
                cwd=REPO_ROOT,
                message=f"Matched generator input is missing: {input_json}",
                artifacts=artifacts,
            )
        if not execute:
            message = (
                "Dry run only. Generator route and credentials were validated."
                if config.available
                else f"Dry run only. Missing credentials: {config.missing_credentials}"
            )
            return self._record_stub(
                stage="generator",
                status="planned",
                command=command,
                cwd=REPO_ROOT,
                message=message,
                artifacts=artifacts,
            )
        if not config.available:
            return self._record_stub(
                stage="generator",
                status="blocked",
                command=command,
                cwd=REPO_ROOT,
                message=f"Missing credentials: {config.missing_credentials}",
                artifacts=artifacts,
            )
        command.extend(["--credential-purpose", "execute"])
        command.append("--execute")
        record = self._run(command, REPO_ROOT)
        record.stage = "generator"
        record.artifacts = artifacts
        return record

    def narrative(self) -> StageRecord:
        finflier_root = REPO_ROOT / "FinFlier" / "system"
        if not finflier_root.exists():
            return self._record_stub(
                stage="narrative",
                status="planned",
                message="FinFlier/system is absent in the current checkout; narrative/schema routing remains planned.",
            )
        return self._record_stub(
            stage="narrative",
            status="available_unvalidated",
            message="FinFlier/system exists but no runtime smoke test was executed.",
            artifacts={"finflier_system": str(finflier_root)},
        )

    def binding_eval(
        self,
        gold_jsonl: Path | None = None,
        pred_jsonl: Path | None = None,
        metrics_json: Path | None = None,
        status_json: Path | None = None,
        narrative_route: str = "narrative_original",
        execute: bool = False,
        require_data: bool = False,
    ) -> StageRecord:
        gold_path = gold_jsonl or WORKSPACE_ROOT / "data" / "financial_narratives" / "gold" / f"{self.expt_id}.jsonl"
        pred_path = pred_jsonl or self.experiment_dir / "binding_eval_predictions" / f"{self.expt_id}.jsonl"
        metrics_path = metrics_json or self.experiment_dir / "binding_eval" / "metrics.json"
        status_path = status_json or self.experiment_dir / "binding_eval" / "status.json"
        command = [
            "python",
            "dist/evaluate_data_binding.py",
            "--experiment-id",
            self.expt_id,
            "--source-id",
            self.expt_id,
            "--narrative-route",
            narrative_route,
            "--gold-jsonl",
            str(gold_path),
            "--pred-jsonl",
            str(pred_path),
            "--metrics-json",
            str(metrics_path),
            "--status-json",
            str(status_path),
            "--vocabulary-types",
            "subject",
            "trend",
            "numerical",
        ]
        if require_data:
            command.append("--require-data")
        artifacts = {
            "gold_jsonl": str(gold_path),
            "gold_exists": gold_path.exists(),
            "pred_jsonl": str(pred_path),
            "pred_exists": pred_path.exists(),
            "metrics_json": str(metrics_path),
            "status_json": str(status_path),
            "vocabulary_types": ["subject", "trend", "numerical"],
        }
        missing = [str(path) for path in (gold_path, pred_path) if not path.exists()]
        if not execute:
            if missing:
                return self._record_stub(
                    stage="binding_eval",
                    status="runtime_blocked",
                    command=command,
                    cwd=REPO_ROOT,
                    message="Financial narrative gold/prediction JSONL is not ready: " + "; ".join(missing),
                    artifacts=artifacts,
                )
            return self._record_stub(
                stage="binding_eval",
                status="planned",
                command=command,
                cwd=REPO_ROOT,
                message="Dry run only. Add --execute to compute FinFlier-style data-binding metrics.",
                artifacts=artifacts,
            )

        record = self._run(command, REPO_ROOT)
        record.stage = "binding_eval"
        record.artifacts = artifacts
        if status_path.exists():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                record.status = payload.get("status") or record.status
                record.message = payload.get("next_step") or payload.get("failure_category")
                record.artifacts["failure_category"] = payload.get("failure_category")
                record.artifacts["metrics"] = payload.get("metrics")
            except Exception as exc:
                record.message = f"Data-binding status JSON could not be parsed: {exc}"
        return record

    def full_plan(
        self,
        retriever_model: str,
        prompt_type: str,
        generator_model: str,
        split: str,
        execute: bool = False,
        generator_max_tokens: int = 512,
    ) -> list[StageRecord]:
        retriever = normalize(retriever_model, RETRIEVER_ALIASES, "retriever model")
        prompt = normalize(prompt_type, PROMPT_ALIASES, "prompt type")
        input_json = self.matched_output_path(retriever, prompt, split)
        return [
            self.retrieval(retriever_model, prompt_type, operation="infer", split=split, execute=execute),
            self.match(retriever_model, prompt_type, split=split, execute=execute),
            self.generator(generator_model, input_json=input_json, execute=execute, max_tokens=generator_max_tokens),
            self.narrative(),
        ]


def parse_args() -> argparse.Namespace:
    stage_choices = ["validate", "retrieval", "match", "generator", "narrative", "binding_eval", "full"]
    parser = argparse.ArgumentParser(description="Unified FQAN pipeline entry point.")
    parser.add_argument("stage_pos", nargs="?", choices=stage_choices)
    parser.add_argument("--stage", dest="stage", choices=stage_choices)
    parser.add_argument("--dataset", default="finqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--expt-id", default="dry_run")
    parser.add_argument("--parameters", type=Path, default=ARGS_PATH)
    parser.add_argument("--convfinqa-enabled", action="store_true")
    parser.add_argument("--retriever-model", default="mistral_v0_3")
    parser.add_argument("--prompt-type", default="original")
    parser.add_argument("--retrieval-operation", choices=["train", "infer"], default="infer")
    parser.add_argument("--generator-model", default="mistral4")
    parser.add_argument("--generator-max-tokens", type=int, default=512)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--binding-gold-jsonl", type=Path)
    parser.add_argument("--binding-pred-jsonl", type=Path)
    parser.add_argument("--binding-metrics-json", type=Path)
    parser.add_argument("--binding-status-json", type=Path)
    parser.add_argument("--narrative-route", default="narrative_original")
    parser.add_argument("--require-binding-data", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.stage and args.stage_pos and args.stage != args.stage_pos:
        parser.error("positional stage and --stage disagree")
    args.stage = args.stage or args.stage_pos
    if not args.stage:
        parser.error("stage is required, either as a positional argument or --stage")
    return args


def main() -> None:
    args = parse_args()
    pipeline = Pipeline(
        dataset=args.dataset,
        expt_id=args.expt_id,
        parameters=args.parameters,
        convfinqa_enabled=args.convfinqa_enabled,
    )

    if args.stage == "validate":
        errors = pipeline.validate_contract(split=args.split)
        records = [
            StageRecord(
                stage="validate",
                status="failed" if errors else "completed",
                started_at=utc_now(),
                finished_at=utc_now(),
                message="; ".join(errors) if errors else "Pipeline contract and parameter ledger are present.",
            )
        ]
    elif args.stage == "retrieval":
        records = [
            pipeline.retrieval(
                model=args.retriever_model,
                prompt_type=args.prompt_type,
                operation=args.retrieval_operation,
                split=args.split,
                execute=args.execute,
            )
        ]
    elif args.stage == "match":
        records = [pipeline.match(args.retriever_model, args.prompt_type, split=args.split, execute=args.execute)]
    elif args.stage == "generator":
        input_json = args.input_json or pipeline.matched_output_path(
            normalize(args.retriever_model, RETRIEVER_ALIASES, "retriever model"),
            normalize(args.prompt_type, PROMPT_ALIASES, "prompt type"),
            args.split,
        )
        records = [
            pipeline.generator(
                args.generator_model,
                input_json=input_json,
                execute=args.execute,
                max_tokens=args.generator_max_tokens,
            )
        ]
    elif args.stage == "narrative":
        records = [pipeline.narrative()]
    elif args.stage == "binding_eval":
        records = [
            pipeline.binding_eval(
                gold_jsonl=args.binding_gold_jsonl,
                pred_jsonl=args.binding_pred_jsonl,
                metrics_json=args.binding_metrics_json,
                status_json=args.binding_status_json,
                narrative_route=args.narrative_route,
                execute=args.execute,
                require_data=args.require_binding_data,
            )
        ]
    else:
        records = pipeline.full_plan(
            retriever_model=args.retriever_model,
            prompt_type=args.prompt_type,
            generator_model=args.generator_model,
            split=args.split,
            execute=args.execute,
            generator_max_tokens=args.generator_max_tokens,
        )

    manifest = None
    if args.write_manifest:
        manifest = pipeline.write_manifest(records)
    payload = {
        "time": utc_now(),
        "dataset": pipeline.dataset,
        "split": args.split,
        "expt_id": pipeline.expt_id,
        "stage": args.stage,
        "execute": args.execute,
        "manifest": str(manifest) if manifest else None,
        "records": [record.__dict__ for record in records],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if any(record.status in {"failed", "blocked"} for record in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
