# Narrative2 Workflow

This directory is the concise documentation layer for Experiment 6. Runtime
assets and archived analyses live under `src/narrative`; they are not stored
beside these instructions.

## Scope

Experiment 6 maps a financial narrative to chart bindings with six fields:
`ObjectName`, `DataName`, `Position`, `Trend`, `Num`, and `Text`.

Two dimensions describe a case:

- Type: `No-adaptor` or `Fine-tuned`.
- Input: a registered prompt `type` or the versioned `FinFlier` prompt.

The existing 38-case matrix contains `Fine-tuned × type` and
`No-adaptor × type`. The supplementary comparison adds three
`No-adaptor × FinFlier` cases for FLAN, Mistral, and T5Gemma2.

## Generation flow

1. Resolve the sealed 85-row input and model identity through the source
   registry.
2. Verify all source, prompt, model, and configuration hashes.
3. Build the declared input without reading gold or repaired predictions.
4. Run the declared route and checkpoint each source.
5. Freeze raw responses and strict predictions before materialization or
   evaluation.

A `Fine-tuned` route produces RetFact candidates with its registered adapter and
uses the registered converter. A `No-adaptor` route loads the registered base
model directly and must not load an adapter or converter.

## Input policy

A `type` input is one of the registered `original`, `zero-shot`, `many-shot`, or
`dynamic-shot` prompts. A `FinFlier` input is extracted from the reference
`default_prompt` and its original special-pattern dispatch. It is versioned and
hashed separately from `original`; historical `6_FinFlier_*` outputs are not
renamed or reused as FinFlier-prompt results.

No input may contain `result`, `reason`, gold targets, judge decisions, or repair
outputs from Narrative2. Native-tokenizer preflight must include the completion
budget. Truncation and model substitution are prohibited.

## Runtime assets

- Generation bundle: `src/narrative/assets/generation/`
- Evaluation bundle: `src/narrative/assets/evaluation/`
- FinFlier reference material: `src/narrative/source_material/`
- Archived temporary analyses: `src/narrative/results/`

Generation code may resolve only generation-stage assets. Evaluation assets are
available only after prediction hashes are frozen.
