# Narrative Artifacts

Public narrative evaluation inputs are stored in
`src/narrative/assets/evaluation/`. They include the evaluation prompt, target
annotations, controlled vocabulary, and a prediction template.

Experiment commands create their outputs locally beneath `src/Experiment/`.
These generated directories are ignored by Git and are not part of the release.

## Expected Run Records

A complete run should preserve:

- the selected input and prompt mode;
- the raw model response and parsed prediction;
- the model and experiment configuration;
- evaluation metrics and rejected records;
- a completion status that distinguishes successful and incomplete runs.

Raw responses must remain unchanged. Any deterministic repair or
materialization step should write a separate artifact so that format compliance
and downstream evaluation can be reported independently.
