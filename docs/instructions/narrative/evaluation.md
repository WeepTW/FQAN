# Narrative Evaluation

## Evaluation Layers

Narrative evaluation keeps three views separate:

1. Strict generation compliance records whether the original model response
   satisfies the declared JSON and field schema.
2. Deterministic materialization recovers readable rows and bindings without
   changing the original response.
3. Reference-aligned evaluation compares materialized bindings with the fixed
   targets distributed for evaluation.

A higher materialized score does not establish that the original response was
schema-valid. Strict and materialized results must therefore be reported
separately.

## Alignment and Field Comparison

Bindings are aligned one-to-one within the same source using `(DataName,
Position)` as the anchor. `DataName` is compared after trimming and case
folding. `Position` preserves JSON types, array lengths, order, and values;
object-key order is irrelevant. When predictions repeat an anchor, the first
match is retained and the remaining predictions are unmatched.

After alignment:

| Field | Comparison |
| --- | --- |
| `ObjectName` | Normalized text followed by one-to-one exact or approved coreference comparison |
| `Trend` | Versioned category normalization preserving direction, period, baseline, and scope |
| `Num` | Finite JSON-number arrays compared one-to-one under the declared tolerance |
| `DataName` | Anchor comparison |
| `Position` | Anchor comparison |
| `Text` | Exact normalized proposition or an explicitly enabled semantic comparison |

If semantic comparison is unavailable, `Text` is reported as `NA` and omitted
from the five-field macro score.

## Counting

For a matched anchor, a correct field contributes one true positive. A present
but incorrect field contributes one false positive and one false negative. A
missing target field contributes one false negative. Unmatched predictions add
false positives for valid present fields; unmatched target bindings add false
negatives. True negatives are not defined.

Each field reports precision, recall, and F1. Formal summaries pool all planned
runs for each case. Supplementary top-ranked selections must not replace the
complete-run result.

## Evaluation Integrity

Targets, comparison examples, and repaired artifacts are evaluator-only.
Comparisons must be identity-blinded, versioned, and auditable. An unavailable
evaluation service is reported as a runtime blocker rather than replaced by an
unplanned method.
