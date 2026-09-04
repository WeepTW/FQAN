#!/usr/bin/env python3
"""Numerical equivalence checks for memory-bounded T5 encoder attention."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Attention

RETRIEVER_ROOT = Path(__file__).resolve().parents[1] / ".external/FINDER/Retriever Codes"
sys.path.insert(0, str(RETRIEVER_ROOT))

from seq2seq_retriever import (  # noqa: E402
    ORIGINAL_T5_ATTENTION_FORWARD,
    T5ChunkedPositionBias,
    configure_t5_chunked_attention,
)


def main() -> None:
    torch.manual_seed(7)
    config = T5Config(
        d_model=32,
        d_kv=8,
        num_heads=4,
        dropout_rate=0.0,
        is_decoder=False,
    )
    first = T5Attention(config, has_relative_attention_bias=True, layer_idx=0)
    second = T5Attention(config, has_relative_attention_bias=False, layer_idx=1)
    hidden = torch.randn(1, 37, 32)
    mask = torch.zeros(1, 1, 1, 37)

    expected_first, expected_bias = ORIGINAL_T5_ATTENTION_FORWARD(
        first, hidden, mask=mask
    )
    expected_second, _ = ORIGINAL_T5_ATTENTION_FORWARD(
        second, expected_first, mask=mask, position_bias=expected_bias
    )

    configure_t5_chunked_attention(8)
    actual_first, lazy_bias = first(hidden, mask=mask)
    actual_second, reused_bias = second(
        actual_first, mask=mask, position_bias=lazy_bias
    )

    assert isinstance(lazy_bias, T5ChunkedPositionBias)
    assert reused_bias is lazy_bias
    torch.testing.assert_close(actual_first, expected_first, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_second, expected_second, rtol=1e-5, atol=1e-6)
    print("t5_chunked_attention_equivalence=pass")


if __name__ == "__main__":
    main()
