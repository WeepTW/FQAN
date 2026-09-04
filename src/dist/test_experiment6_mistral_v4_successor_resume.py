#!/usr/bin/env python3
"""Static trust-boundary tests for Mistral v4 successor resume behavior."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class SuccessorResumeTests(unittest.TestCase):
    def test_fresh_and_resume_modes_are_manifest_conditioned(self) -> None:
        source = (REPO / "dist/run_experiment6_mistral_base_md_chat_template_v4_successor.sh").read_text(encoding="utf-8")
        self.assertIn("formal_resume_args=(--no-resume)", source)
        self.assertIn("$GEN_ROOT/manifests", source)
        self.assertIn('formal_resume_args=()', source)
        self.assertIn('resume_mode="checkpoint-resume"', source)
        self.assertIn('"${formal_resume_args[@]}"', source)

    def test_inference_loads_and_skips_completed_checkpoint_rows(self) -> None:
        source = (REPO / ".external/FINDER/Retriever Codes/Mistral/mistral_direct_binding_chat_inference.py").read_text(encoding="utf-8")
        self.assertLess(source.index("load_checkpoint(row_checkpoint"), source.index("AutoModelForCausalLM.from_pretrained"))
        self.assertIn('if checkpoint_records[int(record["output_index"])] is None', source)
        self.assertIn("write_checkpoint(row_checkpoint, checkpoint_records)", source)

    def test_core_resume_checks_fingerprint_before_accepting_manifest(self) -> None:
        source = (REPO / "dist/run_experiment6_narrative2_generation.py").read_text(encoding="utf-8")
        self.assertIn("resume compatibility fingerprint mismatch", source)
        self.assertIn("resume=not args.no_resume", source)

    def test_finalizer_builds_evaluator_only_judge_bundle_before_text_stage(self) -> None:
        source = (REPO / "dist/finalize_experiment6_mistral_base_md_chat_template.sh").read_text(encoding="utf-8")
        builder = source.index("build_experiment6_judge_examples_v4.py")
        text_stage = source.index("evaluate_narrative2_reference_aligned_v5.py")
        self.assertLess(builder, text_stage)
        self.assertIn('--output-dir "$BIND_ROOT/judge_examples"', source)
        self.assertNotIn("rg ", source)
        self.assertIn("grep -Eq", source)
        self.assertIn("grep -h", source)
        self.assertIn("CONDA_ENV=fnqa", source)
        self.assertIn('PY=(conda run --no-capture-output -n "$CONDA_ENV" python -B)', source)
        materializer = (REPO / "dist/materialize_experiment6_mistral_chat_repaired_projection.py").read_text(encoding="utf-8")
        self.assertNotIn("build_experiment6_judge_examples_v4", materializer)

    def test_canonical_record_occurs_only_after_completion_audit(self) -> None:
        source = (REPO / "dist/run_experiment6_mistral_base_md_chat_template_v4_successor.sh").read_text(encoding="utf-8")
        audit = source.index("audit_experiment6_mistral_base_md_v4_completion.py")
        recorder = source.index("record_experiment6_mistral_base_md_v4_completion.py")
        completed = source.index('event "queue=complete"')
        self.assertLess(audit, recorder)
        self.assertLess(recorder, completed)
        finalizer = (REPO / "dist/finalize_experiment6_mistral_base_md_chat_template.sh").read_text(encoding="utf-8")
        self.assertNotIn("COMPLETE_LOG", finalizer)

    def test_postprocessing_uses_explicit_diagnostic_projection_contract(self) -> None:
        finalizer = (REPO / "dist/finalize_experiment6_mistral_base_md_chat_template.sh").read_text(encoding="utf-8")
        self.assertIn("evaluate_experiment6_binding_candidates_v1.py", finalizer)
        self.assertIn("--scope mistral-base-md", finalizer)
        self.assertIn("experiment6_mistral_base_md_evaluation_v6_1.json", finalizer)
        self.assertIn("--mistral-chat-projection", finalizer)
        self.assertNotIn("evaluate_narrative2_reference_aligned_v6_1.py formal", finalizer)

        materializer = (REPO / "dist/materialize_experiment6_mistral_chat_repaired_projection.py").read_text(encoding="utf-8")
        self.assertIn("\"official\": False", materializer)
        candidate = (REPO / "dist/evaluate_experiment6_binding_candidates_v1.py").read_text(encoding="utf-8")
        self.assertIn("MISTRAL_CHAT_PROJECTION_PROTOCOL", candidate)
        self.assertIn("sourceGenerationManifest", candidate)
        self.assertIn("mistral-base-md", candidate)
        semantic = (REPO / "dist/evaluate_narrative2_reference_aligned_v5.py").read_text(encoding="utf-8")
        self.assertIn("--mistral-chat-projection", semantic)
        self.assertIn("invalid Mistral diagnostic projection manifest", semantic)
        combine = (REPO / "dist/combine_experiment6_v610_with_text_semantic.py").read_text(encoding="utf-8")
        self.assertIn("wrapped_v610", combine)
        self.assertIn("v610.get(\"scope\") == \"mistral-base-md\"", combine)


if __name__ == "__main__":
    unittest.main()
