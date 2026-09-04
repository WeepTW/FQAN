"""Atomic, input-bound row checkpoints shared by retriever inference routes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def checkpoint_path(output_path: Path) -> Path:
    return Path(str(output_path) + ".checkpoint.jsonl")


def load_checkpoint(
    path: Path,
    expected_input_hashes: Sequence[str],
) -> list[dict[str, Any] | None]:
    records: list[dict[str, Any] | None] = [None] * len(expected_input_hashes)
    if not path.is_file():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid retriever checkpoint JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(item, dict):
            raise RuntimeError(
                f"retriever checkpoint item must be an object at {path}:{line_number}"
            )
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RuntimeError(f"invalid retriever checkpoint index at {path}:{line_number}")
        if not 0 <= index < len(records):
            raise RuntimeError(f"retriever checkpoint index out of range: {index}")
        if records[index] is not None:
            raise RuntimeError(f"duplicate retriever checkpoint index: {index}")
        actual_hash = item.get("inputSha256")
        expected_hash = expected_input_hashes[index]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"retriever checkpoint input hash mismatch at index {index}: "
                f"{actual_hash!r} != {expected_hash!r}"
            )
        records[index] = item
    return records


def write_checkpoint(
    path: Path,
    records: Sequence[Mapping[str, Any] | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                if record is not None:
                    handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
