"""Minimal FinQA answer normalization and execution-accuracy helpers.

This FQAN-owned module keeps the public runner independent from the unlicensed
FINDER utility bundle while preserving its answer-comparison behavior.
"""

from __future__ import annotations

from math import isclose
from typing import Any


def _decimal_places(value: float) -> int:
    text = str(value)
    return len(text.rsplit(".", 1)[1]) if "." in text else 5


def finqa_equal(
    prediction: bool | float | str | None,
    reference: float | str,
    include_percentage: bool = False,
    is_close: float = False,
) -> bool:
    """Compare a prediction with a FinQA reference answer."""
    if prediction is None:
        return False
    if type(prediction) is bool:
        return reference == ("yes" if prediction else "no")
    if isinstance(reference, str) or isinstance(prediction, str):
        return prediction == reference

    candidates = (reference / 100, reference, reference * 100) if include_percentage else (reference,)
    for candidate in candidates:
        try:
            if is_close and isclose(candidate, prediction, rel_tol=0.001):
                return True
            precision = min(_decimal_places(prediction), _decimal_places(candidate))
            if round(prediction, precision) == round(candidate, precision):
                return True
        except (ArithmeticError, TypeError, ValueError):
            continue
    return False


def floatify_ans(answer: Any) -> Any:
    """Normalize common executor outputs before FinQA comparison."""
    if answer is None:
        return None
    if type(answer) is dict:
        answer = next(iter(answer.values()))
    elif type(answer) is bool:
        return answer
    elif isinstance(answer, (list, tuple)):
        if not answer:
            return None
        answer = answer[0]

    try:
        return float(answer)
    except (TypeError, ValueError):
        return str(answer)
