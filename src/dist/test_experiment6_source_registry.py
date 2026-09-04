#!/usr/bin/env python3
"""Regression tests for migration-aware Experiment 6 source identities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment6_paths import PathContractError, discover_paths
from experiment6_registry import RegistryError, load_registry


class PathContractTests(unittest.TestCase):
    def test_legacy_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "FQAN"
            repo = workspace / "FINDER-Mistral"
            for path in (repo / "Experiment", repo / "scripts", workspace / "src", workspace / "data"):
                path.mkdir(parents=True, exist_ok=True)
            (repo / "AGENTS.md").write_text("test\n", encoding="utf-8")
            paths = discover_paths(repo)
            self.assertEqual(paths.layout, "fqan-legacy-v1")
            self.assertEqual(paths.dist, repo / "scripts")
            self.assertEqual(paths.instructions, workspace / "src" / "instructions")
            self.assertEqual(paths.models, workspace / "Models")

    def test_fnqa_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "FQAN"
            repo = workspace / "src"
            for path in (repo / "Experiment", repo / "dist", workspace / "docs", workspace / "data"):
                path.mkdir(parents=True, exist_ok=True)
            (repo / "AGENTS.md").write_text("test\n", encoding="utf-8")
            paths = discover_paths(repo)
            self.assertEqual(paths.layout, "fnqa-v1")
            self.assertEqual(paths.dist, repo / "dist")
            self.assertEqual(paths.instructions, workspace / "docs" / "instructions")
            self.assertEqual(paths.models, workspace / "utils" / "models")

    def test_locator_cannot_escape(self) -> None:
        paths = discover_paths()
        with self.assertRaises(PathContractError):
            paths.resolve("data", "../gold.json")
        with self.assertRaises(PathContractError):
            paths.resolve("repo", "/tmp/outside")


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry()

    def test_all_registered_sources_and_hashes_resolve(self) -> None:
        for source_id, source in self.registry.sources.items():
            resolved = self.registry.resolve_source(source_id, source["formalRoute"])
            self.assertEqual(resolved["declaredRoute"], source["formalRoute"])

    def test_generation_cannot_resolve_evaluation_assets_by_stage(self) -> None:
        generation = self.registry.generation_assets()
        evaluation = self.registry.evaluation_assets()
        self.assertNotIn("goldTargets", generation)
        self.assertNotIn("judgeExampleWorkbook", generation)
        self.assertIn("goldTargets", evaluation)

    def test_direct_base_has_no_adapter(self) -> None:
        for source_id in ("flan_t5_large", "mistral_v0_3", "t5gemma_2_1b_1b"):
            resolved = self.registry.resolve_source(source_id, "direct-binding")
            self.assertEqual(resolved["kind"], "base")
            self.assertNotIn("adapter", resolved)

    def test_route_mismatch_is_blocked(self) -> None:
        with self.assertRaises(RegistryError):
            self.registry.resolve_source("mistral_v0_3", "adapter-converter")
        with self.assertRaises(RegistryError):
            self.registry.resolve_source("finqa_mistral_d", "direct-binding")

    def test_unknown_source_is_blocked(self) -> None:
        with self.assertRaises(RegistryError):
            self.registry.resolve_source("invented_source", "direct-binding")

    def test_implementation_hash_changes_compatibility_fingerprint(self) -> None:
        cases = [
            {
                "outputId": "6_flan_base_z",
                "sourceId": "flan_t5_large",
                "promptMode": "zero-shot",
                "route": "direct-binding",
            }
        ]
        common = {
            "execution_mode": "formal;rows=85",
            "generation_config_sha256": "config",
            "prompt_policy_version": "prompt",
            "input_workbook_sha256": "input",
        }
        first = self.registry.compatibility_snapshot(
            cases,
            **common,
            implementation_sha256s={"runner": "first"},
        )
        second = self.registry.compatibility_snapshot(
            cases,
            **common,
            implementation_sha256s={"runner": "second"},
        )
        self.assertNotEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
