# Narrative2 Evaluation

## Evaluation layers

Experiment 6 preserves three distinct views:

1. Strict generation compliance records whether the raw model response satisfies
   the declared JSON and field schema.
2. Repaired-v4 materialization performs deterministic, gold-free recovery of
   readable rows and bindings. It never changes the raw response and is not a
   claim that the original output was schema-valid.
3. Reference-aligned v6.1 evaluates the materialized bindings against frozen
   gold targets.

The strict and repaired views must be reported separately. A higher repaired
score does not prove that a repair rule or prompt is scientifically better.

## Alignment and field comparison

Bindings are aligned one-to-one within the same source by the hard anchor
`(DataName, Position)`. `DataName` is compared after trimming and case folding.
`Position` retains JSON types, array lengths, order, and values; object-key order
is irrelevant. Duplicate predicted anchors keep the first match and count the
remaining predictions as unmatched.

After anchor alignment:

| Field | Comparison |
| --- | --- |
| `ObjectName` | Unicode and whitespace normalization, followed by one-to-one exact or approved coreference comparison |
| `Trend` | Versioned trend-category normalization, preserving direction, period, baseline, and scope |
| `Num` | Finite JSON-number arrays with one-to-one numerical comparison under the declared tolerance |
| `DataName` | Hard anchor comparison |
| `Position` | Hard anchor comparison |
| `Text` | Exact normalized proposition or an explicitly executed semantic judge |

Without a Text judge, `Text` remains `NA` and is excluded from the five-field
macro score.

## Counting

For a matched anchor, a correct field contributes one true positive. A present
but wrong field contributes one false positive and one false negative. A
missing gold field contributes one false negative. Unmatched predictions add
false positives for valid present fields; unmatched gold bindings add false
negatives. Experiment 6 does not define true negatives.

Each field reports Precision, Recall, and F1. The primary table pools all ten
runs for each case. Top-1 and top-3 are supplementary run selections and must
not replace the ten-run result.

## Evaluation integrity

Gold, judge examples, and repaired artifacts are evaluator-only. Judges must be
identity-blinded, versioned, and auditable. Unsupported services are recorded as
runtime blockers; no substitute model or partial ranking is allowed.
