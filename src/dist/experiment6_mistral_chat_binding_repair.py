#!/usr/bin/env python3
"""Gold-free mechanical repair for isolated Mistral chat Binding outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


BINDING_KEYS = frozenset(
    {"ObjectName", "DataName", "Position", "Trend", "Num", "Text"}
)
EXAMPLE_MARKER = re.compile(r"\[EXAMPLE\s+\d+\]", re.IGNORECASE)


def balanced_json_objects(raw: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    text = str(raw or "")
    for start, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        objects.append(value)
                    break
    return objects


def repair_unique_binding(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {"available": False, "reason": "empty"}
    if "## Output examples" in text:
        return {"available": False, "reason": "prompt_echo_guard"}
    candidates = [
        value
        for value in balanced_json_objects(text)
        if isinstance(value, Mapping) and frozenset(value) == BINDING_KEYS
    ]
    unique = {
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False): value
        for value in candidates
    }
    if len(EXAMPLE_MARKER.findall(text)) > 1 and len(unique) != 1:
        return {
            "available": False,
            "reason": "prompt_echo_guard",
            "candidateCount": len(unique),
        }
    if len(unique) != 1:
        return {
            "available": False,
            "reason": "binding_object_not_unique",
            "candidateCount": len(unique),
        }
    binding = next(iter(unique.values()))
    return {
        "available": True,
        "method": "unique-six-field-binding-object-v1",
        "payload": {"result": [binding], "reason": "mechanically wrapped"},
    }


if __name__ == "__main__":
    raise SystemExit("import this module; it is not a standalone mutator")
