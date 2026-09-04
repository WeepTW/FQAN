#!/usr/bin/env python3
"""Validated source identities and compatibility fingerprints for Experiment 6."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiment6_paths import PATHS, Experiment6Paths


class RegistryError(RuntimeError):
    """Raised when a registry identity, route, path, or digest is invalid."""


FORMAL_ROUTES = frozenset({"adapter-converter", "direct-binding", "converter-control"})
GENERATION_STAGES = frozenset({"generation", "provenance-only"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SourceRegistry:
    path: Path
    file_sha256: str
    document: Mapping[str, Any]
    paths: Experiment6Paths

    @property
    def sources(self) -> Mapping[str, Mapping[str, Any]]:
        return self.document["sources"]

    def generation_assets(self) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        for name, raw in self.document["assets"].items():
            if raw.get("stage") not in GENERATION_STAGES:
                continue
            resolved[name] = self._verify_file(name, raw)
        return resolved

    def evaluation_assets(self) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        for name, raw in self.document["assets"].items():
            if raw.get("stage") != "evaluation":
                continue
            resolved[name] = self._verify_file(name, raw)
        return resolved

    def source(self, source_id: str) -> Mapping[str, Any]:
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise RegistryError(f"unknown source_id: {source_id}") from exc

    def resolve_source(
        self,
        source_id: str,
        declared_route: str,
        *,
        require_formal_route: bool = True,
    ) -> dict[str, Any]:
        if declared_route not in FORMAL_ROUTES:
            raise RegistryError(f"unsupported declared route: {declared_route}")
        source = dict(self.source(source_id))
        formal_route = source.get("formalRoute")
        if require_formal_route and declared_route != formal_route:
            raise RegistryError(
                f"source {source_id} requires route {formal_route}, got {declared_route}"
            )

        kind = source.get("kind")
        if declared_route == "adapter-converter" and kind != "adapter":
            raise RegistryError(
                f"adapter-converter source must have kind=adapter: {source_id}"
            )
        if declared_route == "converter-control" and kind != "control":
            raise RegistryError(
                f"converter-control source must have kind=control: {source_id}"
            )
        if declared_route == "direct-binding" and kind in {"adapter", "control"}:
            raise RegistryError(
                f"direct-binding source may not load {kind}: {source_id}"
            )

        result: dict[str, Any] = {
            "sourceId": source_id,
            "kind": kind,
            "family": source.get("family"),
            "baseModel": source.get("baseModel"),
            "formalRoute": formal_route,
            "declaredRoute": declared_route,
        }
        for key in (
            "actualModelRequired",
            "reasoningEffort",
            "allowSubstitution",
            "legacyDiagnosticOnly",
        ):
            if key in source:
                result[key] = source[key]

        if kind == "adapter":
            adapter = source.get("adapter")
            if not isinstance(adapter, Mapping):
                raise RegistryError(f"adapter record missing for {source_id}")
            adapter_root = self.paths.resolve_locator(adapter)
            if not adapter_root.is_dir():
                raise RegistryError(f"adapter directory missing for {source_id}: {adapter_root}")
            config = self._verify_named_file(
                f"{source_id}.adapter_config",
                adapter_root / "adapter_config.json",
                adapter.get("configSha256"),
            )
            weights = self._verify_named_file(
                f"{source_id}.adapter_weights",
                adapter_root / "adapter_model.safetensors",
                adapter.get("weightsSha256"),
            )
            prompt_mode = source.get("promptMode")
            training = self._resolve_training_data(prompt_mode)
            result["adapter"] = {
                "locator": {"root": adapter.get("root"), "path": adapter.get("path")},
                "resolvedPath": str(adapter_root),
                "config": config,
                "weights": weights,
            }
            result["trainingPromptMode"] = prompt_mode
            result["trainingData"] = training
        return result

    def compatibility_snapshot(
        self,
        cases: Iterable[Mapping[str, str]],
        *,
        execution_mode: str,
        generation_config_sha256: str,
        prompt_policy_version: str,
        input_workbook_sha256: str,
        implementation_sha256s: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        source_records: dict[str, dict[str, Any]] = {}
        case_records: list[dict[str, str]] = []
        for case in cases:
            source_id = case["sourceId"]
            route = case["route"]
            resolved = self.resolve_source(source_id, route)
            source_records[source_id] = _portable_source_identity(resolved)
            case_records.append(
                {
                    "outputId": case["outputId"],
                    "sourceId": source_id,
                    "promptMode": case["promptMode"],
                    "route": route,
                }
            )
        material = {
            "protocol": "experiment6-compatibility-fingerprint-v1",
            "executionMode": execution_mode,
            "registrySha256": self.file_sha256,
            "generationConfigSha256": generation_config_sha256,
            "promptPolicyVersion": prompt_policy_version,
            "inputWorkbookSha256": input_workbook_sha256,
            "implementationSha256s": dict(sorted((implementation_sha256s or {}).items())),
            "generationAssets": {
                name: _portable_file_identity(asset)
                for name, asset in sorted(self.generation_assets().items())
            },
            "sources": dict(sorted(source_records.items())),
            "cases": sorted(case_records, key=lambda item: item["outputId"]),
        }
        return {"sha256": canonical_sha256(material), "material": material}

    def _resolve_training_data(self, prompt_mode: object) -> dict[str, dict[str, Any]]:
        if not isinstance(prompt_mode, str):
            raise RegistryError("adapter promptMode must be a string")
        training_sets = self.document["trainingData"]
        try:
            raw_set = training_sets[prompt_mode]
        except KeyError as exc:
            raise RegistryError(f"training data missing for {prompt_mode}") from exc
        return {
            split: self._verify_file(f"trainingData.{prompt_mode}.{split}", locator)
            for split, locator in sorted(raw_set.items())
        }

    def _verify_file(self, name: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        path = self.paths.resolve_locator(raw)
        verified = self._verify_named_file(name, path, raw.get("sha256"))
        verified["locator"] = {"root": raw.get("root"), "path": raw.get("path")}
        if "sheet" in raw:
            verified["sheet"] = raw["sheet"]
        if "stage" in raw:
            verified["stage"] = raw["stage"]
        return verified

    @staticmethod
    def _verify_named_file(name: str, path: Path, expected_sha256: object) -> dict[str, Any]:
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise RegistryError(f"invalid expected SHA-256 for {name}")
        if not path.is_file():
            raise RegistryError(f"registered file missing for {name}: {path}")
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise RegistryError(
                f"registered SHA-256 mismatch for {name}: expected "
                f"{expected_sha256}, got {actual}"
            )
        return {"resolvedPath": str(path), "sha256": actual, "bytes": path.stat().st_size}


def _portable_file_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "locator": value.get("locator"),
        "sha256": value.get("sha256"),
        "bytes": value.get("bytes"),
    }
    for key in ("sheet", "stage"):
        if key in value:
            result[key] = value[key]
    return result


def _portable_source_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: raw for key, raw in value.items() if key != "adapter" and key != "trainingData"}
    adapter = value.get("adapter")
    if isinstance(adapter, Mapping):
        result["adapter"] = {
            "locator": adapter.get("locator"),
            "config": _portable_file_identity(adapter["config"]),
            "weights": _portable_file_identity(adapter["weights"]),
        }
    training = value.get("trainingData")
    if isinstance(training, Mapping):
        result["trainingData"] = {
            split: _portable_file_identity(record)
            for split, record in sorted(training.items())
        }
    return result


def load_registry(
    path: Path | None = None,
    *,
    paths: Experiment6Paths = PATHS,
) -> SourceRegistry:
    registry_path = (path or paths.resolve("repo", "config/experiment6_source_registry.json")).resolve()
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot load source registry {registry_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise RegistryError("source registry must be a JSON object")
    if document.get("protocol") != "experiment6-source-registry-v1":
        raise RegistryError("unsupported source registry protocol")
    for key in ("assets", "trainingData", "sources"):
        if not isinstance(document.get(key), dict):
            raise RegistryError(f"source registry requires object {key}")
    return SourceRegistry(
        path=registry_path,
        file_sha256=sha256_file(registry_path),
        document=document,
        paths=paths,
    )
