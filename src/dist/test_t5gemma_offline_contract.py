#!/usr/bin/env python3
"""Regression tests for offline T5Gemma cache routing."""

from __future__ import annotations

import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / ".external/FINDER/Retriever Codes" / "t5gemma-2" / "t5gemma-2_train.py"


class T5GemmaOfflineContractTests(unittest.TestCase):
    def test_hf_home_maps_to_hub_cache_without_token_gate(self) -> None:
        environment = {"HF_HOME": "/tmp/fnqa-hf-contract", "HF_TOKEN": ""}
        with patch.dict(os.environ, environment, clear=False):
            os.environ.pop("HF_HUB_CACHE", None)
            namespace = runpy.run_path(str(SCRIPT), run_name="t5gemma_contract_test")
        self.assertEqual(namespace["DEFAULT_CACHE_DIR"], "/tmp/fnqa-hf-contract/hub")
        self.assertNotIn("require_training_params", namespace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
