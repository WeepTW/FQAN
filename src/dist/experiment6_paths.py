#!/usr/bin/env python3
"""Migration-aware logical paths for the FQAN workspace.

The resolver supports both the current FQAN/src layout and the legacy
FINDER-Mistral layout.  Configuration files name a logical root plus a
relative path; they never need a username or an absolute workspace path.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class PathContractError(RuntimeError):
    """Raised when a logical path escapes or contradicts the repo layout."""


@dataclass(frozen=True)
class Experiment6Paths:
    layout: str
    workspace: Path
    repo: Path
    dist: Path
    docs: Path
    asset: Path
    instructions: Path
    data: Path
    log: Path
    code: Path
    utils: Path
    models: Path

    @property
    def conda_env(self) -> str:
        return "fnqa"

    @property
    def hf_home(self) -> Path:
        return self.models / ".cache" / "huggingface"

    def roots(self) -> Mapping[str, Path]:
        return {
            "workspace": self.workspace,
            "repo": self.repo,
            "dist": self.dist,
            "docs": self.docs,
            "asset": self.asset,
            "instructions": self.instructions,
            "data": self.data,
            "log": self.log,
            "code": self.code,
            "utils": self.utils,
            "models": self.models,
            "hf_home": self.hf_home,
        }

    def environment(self) -> Mapping[str, str]:
        return {
            "FQAN_ROOT": str(self.workspace),
            "FQAN_SRC_ROOT": str(self.repo),
            "FQAN_DIST_ROOT": str(self.dist),
            "FQAN_DOCS_ROOT": str(self.docs),
            "FQAN_ASSET_ROOT": str(self.asset),
            "FQAN_DATA_ROOT": str(self.data),
            "FQAN_LOG_ROOT": str(self.log),
            "FQAN_CODE_ROOT": str(self.code),
            "FQAN_UTILS_ROOT": str(self.utils),
            "FQAN_MODELS_ROOT": str(self.models),
            "HF_HOME": str(self.hf_home),
            "HF_HUB_CACHE": str(self.hf_home / "hub"),
            "TRANSFORMERS_CACHE": str(self.hf_home / "hub"),
            "CONDA_ENV": self.conda_env,
            "FQAN_LAYOUT": self.layout,
        }

    def resolve(self, root: str, relative_path: str = "") -> Path:
        try:
            base = self.roots()[root]
        except KeyError as exc:
            raise PathContractError(f"unknown logical root: {root}") from exc

        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise PathContractError(
                f"path must be relative and may not contain '..': {relative_path!r}"
            )
        target = (base / relative).resolve()
        resolved_base = base.resolve()
        if target != resolved_base and resolved_base not in target.parents:
            raise PathContractError(
                f"path escapes logical root {root}: {relative_path!r}"
            )
        return target

    def resolve_locator(self, locator: Mapping[str, object]) -> Path:
        root = locator.get("root")
        path = locator.get("path", "")
        if not isinstance(root, str) or not isinstance(path, str):
            raise PathContractError("locator requires string root and path")
        return self.resolve(root, path)


def _looks_like_repo(path: Path) -> bool:
    return (
        (path / "README.md").is_file()
        and (path / "Experiment").is_dir()
        and ((path / "scripts").is_dir() or (path / "dist").is_dir())
    )


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if _looks_like_repo(candidate):
            return candidate
    raise PathContractError(f"cannot locate Experiment 6 repository from {current}")


def discover_paths(repo_root: Path | None = None) -> Experiment6Paths:
    repo = (repo_root or find_repo_root()).resolve()
    if not _looks_like_repo(repo):
        raise PathContractError(f"not an Experiment 6 repository: {repo}")

    workspace = repo.parent
    if repo.name not in {"FINDER-Mistral", "src"}:
        raise PathContractError(f"unsupported implementation repo name: {repo.name}")
    if not (workspace / "data").is_dir():
        raise PathContractError(f"workspace data directory is missing: {workspace / 'data'}")

    # Accept every deliberate migration boundary, not arbitrary directory names.
    if (workspace / "docs").is_dir():
        docs = workspace / "docs"
        asset = docs / "asset"
        docs_layout = "docs"
    elif repo.name == "FINDER-Mistral" and (workspace / "src").is_dir():
        docs = workspace / "src"
        asset = docs / ("asset" if (docs / "asset").is_dir() else "doc")
        docs_layout = "legacy-src"
    else:
        raise PathContractError("documentation root is absent at a known migration boundary")

    dist = repo / ("dist" if (repo / "dist").is_dir() else "scripts")
    external_finder = repo / ".external" / "FINDER"
    code = external_finder if external_finder.is_dir() else docs / "code"
    utils = workspace / "utils"
    models = (
        utils / "models"
        if repo.name == "src" or (utils / "models").is_dir()
        else workspace / "Models"
    )
    layout = (
        "fnqa-v1"
        if repo.name == "src" and docs_layout == "docs" and dist.name == "dist"
        else "fnqa-transition-v1"
        if docs_layout == "docs" or dist.name == "dist"
        else "legacy-finder-v1"
    )
    return Experiment6Paths(
        layout=layout,
        workspace=workspace,
        repo=repo,
        dist=dist,
        docs=docs,
        asset=asset,
        instructions=docs / "instructions",
        data=workspace / "data",
        log=docs / "log",
        code=code,
        utils=utils,
        models=models,
    )


PATHS = discover_paths()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the FQAN path contract")
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    parser.add_argument("--output")
    args = parser.parse_args()
    values = dict(PATHS.environment())
    rendered = (
        json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else "".join(
            f"export {key}={shlex.quote(value)}\n" for key, value in values.items()
        )
    )
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(target)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
