# Narrative2 Python annotation evaluator

This bundle evaluates model-produced FinFlier bindings against the frozen
`narrative2.xlsx` target while using `narrative1.xlsx` only for Trend/Num
presence.

## Requirements

- Python 3.10 or newer.
- The evaluator itself uses only the Python standard library.

## Commands

```bash
python evaluate_narrative2_annotations.py self-test
python evaluate_narrative2_annotations.py compare-batch --targets gold_targets.json --predictions predictions.jsonl --output evaluation.json
python evaluate_narrative2_annotations.py compare-row --targets gold_targets.json --source Econ_002 --prediction row.json --output result.json
python evaluate_narrative2_annotations.py compare-binding --targets gold_targets.json --source Econ_002 --binding-index 0 --prediction binding.json --output result.json
```

`compare-batch` requires strict JSONL with one row result per line. Unknown
sources, duplicate `(source, run)` keys, malformed bindings, and binding-count
mismatches are rejected before scoring.
