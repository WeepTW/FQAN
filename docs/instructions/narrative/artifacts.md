# Narrative2 Artifacts

## Active layout

Runtime inputs and evaluation assets are stored under `src/narrative/assets`.
Derived analyses formerly held in workspace `tmp/` are preserved under
`src/narrative/results/legacy_tmp_20260815/`. Source documents that informed the
method are stored under `src/narrative/source_material/`.

The relocation changes paths but not asset identity. Pre- and post-relocation
SHA-256 inventories must agree before the old instruction payload is replaced.

## Run outputs

Every case/run writes to a fresh `src/Experiment/<experiment-root>/` subtree and
preserves:

- prompt and input provenance;
- raw model responses and strict predictions;
- row checkpoints and runtime status;
- model, adapter, converter, seed, and route identity;
- format and materialization reports;
- evaluation metrics and rejected records;
- a SHA-256 inventory.

A compatibility fingerprint covers data bytes, prompt bytes, source registry,
model or adapter identity, route, generation code, and evaluator version. A run
may resume only when its fingerprint matches.

## Completion gates

The existing formal matrix completes at 38 cases × 10 runs × 85 sources. The
supplementary FinFlier comparison completes independently at 3 cases × 10 runs
× 85 sources. Its manifests must show `direct-binding`, no adapter, no
converter, no generation cache, 85 unique sources, and one consistent
fingerprint.

Historical predictions, repaired candidates, and score tables remain immutable.
Archived temporary analyses may support audit and reproduction, but they are
not imported into a fresh generation root or a formal ranking.
