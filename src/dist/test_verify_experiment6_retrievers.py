#!/usr/bin/env python3
"""Regression tests for Experiment 6 retriever verification summaries."""

from __future__ import annotations

import unittest

import verify_experiment6_retrievers as verifier


class OptionalFmeanTests(unittest.TestCase):
    def test_empty_values_return_none(self) -> None:
        self.assertIsNone(verifier.optional_fmean([]))

    def test_nonempty_values_return_mean(self) -> None:
        self.assertEqual(verifier.optional_fmean([0.5, 1.0]), 0.75)

    def test_missing_metric_has_explicit_marker(self) -> None:
        self.assertEqual(verifier.format_optional(None), "—")

    def test_metric_format_is_stable(self) -> None:
        self.assertEqual(verifier.format_optional(0.75), "0.750000")


if __name__ == "__main__":
    unittest.main()
