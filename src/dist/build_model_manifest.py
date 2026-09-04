#!/usr/bin/env python3
"""Build a value-free inventory for canonical FQAN model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiment6_paths import PATHS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        base = Path(root)
        for name in files:
            candidate = base / name
            if not candidate.is_symlink():
                total += candidate.stat().st_size
    return total


def cache_repositories(cache_root: Path, consumers: dict[str, list[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not cache_root.is_dir():
        return rows
    for repo in sorted(cache_root.glob("models--*")):
        refs: dict[str, str] = {}
        refs_root = repo / "refs"
        if refs_root.is_dir():
            for ref in sorted(refs_root.rglob("*")):
                if ref.is_file() and not ref.is_symlink():
                    refs[str(ref.relative_to(refs_root))] = ref.read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
        snapshots_root = repo / "snapshots"
        revisions = sorted(p.name for p in snapshots_root.iterdir()) if snapshots_root.is_dir() else []
        blobs_root = repo / "blobs"
        blobs = [p for p in blobs_root.iterdir()] if blobs_root.is_dir() else []
        complete_blobs = [p for p in blobs if p.is_file() and not p.name.endswith(".incomplete")]
        rows.append(
            {
                "cache_id": repo.name,
                "locator": {"root": "HF_HOME", "path": str(repo.relative_to(PATHS.hf_home))},
                "source_model_id": repo.name.removeprefix("models--").replace("--", "/"),
                "refs": refs,
                "snapshot_revisions": revisions,
                "complete_blob_count": len(complete_blobs),
                "complete_blob_bytes": sum(p.stat().st_size for p in complete_blobs),
                "partial_blob_count": sum(p.name.endswith(".incomplete") for p in blobs),
                "license_status": "verify_upstream_model_card",
                "gated_status": "unknown",
                "consumers": consumers.get(repo.name, []),
            }
        )
    return rows


def registry_consumers() -> dict[str, list[str]]:
    registry_path = PATHS.repo / "config" / "experiment6_source_registry.json"
    if not registry_path.is_file():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    consumers: dict[str, list[str]] = {}
    for source_id, source in payload.get("sources", {}).items():
        model_id = source.get("baseModel")
        if not isinstance(model_id, str) or "/" not in model_id:
            continue
        cache_id = "models--" + model_id.replace("/", "--")
        consumers.setdefault(cache_id, []).append(source_id)
    return {key: sorted(value) for key, value in consumers.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PATHS.utils / "model_manifest.json")
    args = parser.parse_args()
    model_root = PATHS.models
    top_assets = []
    for path in sorted(model_root.iterdir()):
        if path.name == ".cache":
            continue
        top_assets.append(
            {
                "name": path.name,
                "kind": "symlink" if path.is_symlink() else "directory" if path.is_dir() else "file",
                "locator": {"root": "FQAN_MODELS_ROOT", "path": path.name},
                "bytes": directory_size(path) if path.is_dir() else path.stat().st_size,
                "sha256": sha256_file(path) if path.is_file() and not path.is_symlink() else None,
            }
        )
    consumers = registry_consumers()
    repositories = cache_repositories(PATHS.hf_home, consumers)
    repositories.extend(cache_repositories(PATHS.hf_home / "hub", consumers))
    manifest = {
        "schema_version": 1,
        "purpose": "canonical base-model/cache inventory; Experiment adapters remain under src/Experiment",
        "canonical_roots": {
            "models": {"root": "FQAN_UTILS_ROOT", "path": "models"},
            "huggingface": {"root": "FQAN_MODELS_ROOT", "path": ".cache/huggingface"},
        },
        "top_level_assets": top_assets,
        "huggingface_repositories": sorted(
            repositories, key=lambda row: str(row["locator"]["path"])
        ),
        "quarantine": {"root": "FQAN_UTILS_ROOT", "path": "quarantine/hf_cache_merge_20260811"},
        "content_hash_inventory": {
            "root": "FQAN_LOG_ROOT",
            "path": "20260810T124329Z_fnqa_pre_migration_baseline/model_files_sha256_combined.tsv",
        },
        "revision_blob_graph": {
            "root": "FQAN_LOG_ROOT",
            "path": "20260810T124329Z_fnqa_pre_migration_baseline/hf_repo_revision_blob_graph.tsv",
        },
        "credential_policy": "HF_TOKEN is environment-only; credential quarantine is excluded from active runtime resolution",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
