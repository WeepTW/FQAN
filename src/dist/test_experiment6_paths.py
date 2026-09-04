#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment6_paths import PathContractError, discover_paths


def make_repo(workspace: Path, repo_name: str, dist_name: str) -> Path:
    repo = workspace / repo_name
    (repo / "Experiment").mkdir(parents=True)
    (repo / dist_name).mkdir()
    (repo / "README.md").write_text("test\n", encoding="utf-8")
    (workspace / "data").mkdir()
    return repo


class PathContractTests(unittest.TestCase):
    def test_legacy_layout(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw) / "FQAN"
            repo = make_repo(workspace, "FINDER-Mistral", "scripts")
            (workspace / "src" / "doc").mkdir(parents=True)
            paths = discover_paths(repo)
            self.assertEqual(paths.layout, "legacy-finder-v1")
            self.assertEqual(paths.conda_env, "fnqa")
            self.assertEqual(paths.asset, workspace / "src" / "doc")

    def test_transition_and_final_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            transitional = Path(raw) / "FQAN"
            repo = make_repo(transitional, "FINDER-Mistral", "dist")
            (transitional / "docs" / "asset").mkdir(parents=True)
            paths = discover_paths(repo)
            self.assertEqual(paths.layout, "fnqa-transition-v1")
            self.assertEqual(paths.dist, repo / "dist")

        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw) / "FQAN"
            repo = make_repo(workspace, "src", "dist")
            (workspace / "docs" / "asset").mkdir(parents=True)
            (workspace / "utils" / "models").mkdir(parents=True)
            paths = discover_paths(repo)
            self.assertEqual(paths.layout, "fnqa-v1")
            self.assertEqual(paths.models, workspace / "utils" / "models")
            self.assertEqual(paths.environment()["CONDA_ENV"], "fnqa")
            self.assertEqual(paths.code, workspace / "docs" / "code")
            external = repo / ".external" / "FINDER"
            external.mkdir(parents=True)
            paths = discover_paths(repo)
            self.assertEqual(paths.code, external)

    def test_locator_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw) / "FQAN"
            repo = make_repo(workspace, "src", "dist")
            (workspace / "docs" / "asset").mkdir(parents=True)
            paths = discover_paths(repo)
            with self.assertRaises(PathContractError):
                paths.resolve("data", "../secret")


if __name__ == "__main__":
    unittest.main()
