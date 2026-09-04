#!/usr/bin/env python3
"""Regression tests for relaxed Experiment 6 Binding materialization v3."""

from __future__ import annotations

import unittest

from materialize_experiment6_bindings_relaxed_v3 import (
    BINDING_KEYS,
    normalize_binding,
    payload_bindings,
    recover_unavailable_raw,
    relaxed_binding_valid,
)


class RelaxedBindingV3Test(unittest.TestCase):
    def test_missing_fields_are_blank(self) -> None:
        binding, operations, _ = normalize_binding({"DataName": "Revenue"})
        self.assertEqual(tuple(binding), BINDING_KEYS)
        self.assertEqual(binding["ObjectName"], [])
        self.assertEqual(binding["Position"], [])
        self.assertEqual(binding["Trend"], "")
        self.assertEqual(binding["Num"], [])
        self.assertEqual(binding["Text"], "")
        self.assertIn("fill-missing:ObjectName", operations)
        self.assertEqual(relaxed_binding_valid(binding), (True, "valid"))

    def test_num_strings_and_units_are_retained(self) -> None:
        source = {
            "ObjectName": "Robocalls",
            "DataName": "Robocalls",
            "Position": {"Begin": [10, 1], "End": [10, 1]},
            "Trend": None,
            "Num": "185 million per month",
            "Text": ["Robocalls", "185 million"],
        }
        binding, operations, quality = normalize_binding(source)
        self.assertEqual(binding["ObjectName"], ["Robocalls"])
        self.assertEqual(binding["Num"], ["185 million per month"])
        self.assertEqual(binding["Text"], '["Robocalls","185 million"]')
        self.assertIn("wrap-scalar:Num", operations)
        self.assertIn("Num-retains-nonnumeric-content", quality)

    def test_position_pair_and_aliases_are_normalized(self) -> None:
        pair, _, _ = normalize_binding({"Position": [4, 2]})
        self.assertEqual(pair["Position"], [{"Begin": [4, 2], "End": [4, 2]}])
        alias, _, _ = normalize_binding({"Position": {"Start": [1, 3], "Bound": [1, 4]}})
        self.assertEqual(alias["Position"], [{"Begin": [1, 3], "End": [1, 3]}])

    def test_nonobject_payload_is_represented_without_guessing_text(self) -> None:
        bindings, _, quality, status = payload_bindings([0, 12])
        self.assertEqual(status, "relaxed_payload_recovered")
        self.assertEqual(bindings[0]["Position"], [{"Begin": [0, 12], "End": [0, 12]}])
        self.assertEqual(bindings[0]["Text"], "")
        self.assertIn("no-binding-object", quality)

    def test_reason_only_payload_becomes_empty_result(self) -> None:
        bindings, _, quality, status = payload_bindings({"reason": "unbound"})
        self.assertEqual(bindings, [])
        self.assertEqual(status, "relaxed_payload_empty")
        self.assertIn("no-binding-structure", quality)

    def test_known_flan_malformed_shape_is_recovered(self) -> None:
        raw = '["objectName":["..."],"DataName":"Revenue","Position":["Begin":[0,1],"End":[0,1]],"Trend":"None","Num":["$5m"],"Text":"Revenue was $5m"]'
        bindings, _, quality, status, marker = recover_unavailable_raw(raw)
        self.assertEqual(status, "raw_flan_shape_recovered")
        self.assertEqual(marker, "known-flan-shape")
        self.assertEqual(bindings[0]["DataName"], "Revenue")
        self.assertEqual(bindings[0]["Num"], ["$5m"])
        self.assertIn("known-flan-malformed-shape", quality)

    def test_mistral_missing_brace_is_recovered(self) -> None:
        raw = '## Output {"result":[{"ObjectName":["A"],"DataName":"Revenue","Position":[{"Begin":[0,1],"End":[0,1]}],"Trend":"None","Num":[5],"Text":"A was 5"],"reason":"Success"}'
        bindings, operations, _, status, marker = recover_unavailable_raw(raw)
        self.assertEqual(status, "raw_json_recovered")
        self.assertEqual(marker, "leading-output")
        self.assertEqual(bindings[0]["Text"], "A was 5")
        self.assertIn("insert-missing-binding-closing-brace", operations)

    def test_begin_end_pseudo_answer_is_recovered(self) -> None:
        raw = '[BEGIN] {ObjectName:[Country], DataName:"GDP", Position:{Begin:[2,1],End:[2,1]}, Trend:"increase", Num:["4.2%"], Text:"GDP rose 4.2%."} [END]'
        bindings, _, _, status, marker = recover_unavailable_raw(raw)
        self.assertEqual(status, "raw_pseudo_recovered")
        self.assertEqual(marker, "begin-end")
        self.assertEqual(bindings[0]["DataName"], "GDP")
        self.assertEqual(bindings[0]["Text"], "GDP rose 4.2%.")

    def test_empty_explicit_binding_is_not_promoted(self) -> None:
        raw = '## Output content {"result":[0,0],"reason":"..."}'
        bindings, _, _, status, _ = recover_unavailable_raw(raw)
        self.assertIsNone(bindings)
        self.assertEqual(status, "unavailable_no_binding_structure")

    def test_example_contamination_is_not_promoted(self) -> None:
        raw = '[BEGIN] {ObjectName:["WTI"], Position:{Begin:[0,0],End:[0,0]}, Trend:"None", Text:"value ## Example [EXAMPLE 01]"} [END]'
        bindings, _, _, status, _ = recover_unavailable_raw(raw)
        self.assertIsNone(bindings)
        self.assertEqual(status, "unavailable_no_binding_structure")

    def test_prompt_echo_is_not_promoted_to_answer(self) -> None:
        raw = '# Financial data-text binding\n## Binding coordinate contract\nReturn exactly {"result":[{"ObjectName":["..."]}]}\n## Chart data (lossless compact form)\n__row__,GDP'
        bindings, _, _, status, marker = recover_unavailable_raw(raw)
        self.assertIsNone(bindings)
        self.assertEqual(status, "unavailable_prompt_echo")
        self.assertIsNone(marker)

    def test_empty_and_degenerate_are_unavailable(self) -> None:
        self.assertEqual(recover_unavailable_raw("")[3], "unavailable_empty")
        self.assertEqual(recover_unavailable_raw("[##_rowOC__CO_____ [ [ [ [ [ [ [ [ [ [ [")[3], "unavailable_degenerate_tokens")


if __name__ == "__main__":
    unittest.main()
