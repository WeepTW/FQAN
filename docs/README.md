# Documentation

Start with the repository [`README.md`](../README.md). It gives the supported
sequence for installing requirements, creating the environment, loading data,
installing models, and running experiments.

## Guides

- [`asset/folder_structure.md`](asset/folder_structure.md) explains the public
  repository layout.
- [`asset/system_workflow.md`](asset/system_workflow.md) summarizes the research
  workflow and its stable entry points.
- [`asset/new_prompt.txt`](asset/new_prompt.txt) records the public retriever and
  data-binding prompt contract.
- [`instructions/narrative/artifacts.md`](instructions/narrative/artifacts.md)
  describes narrative inputs and locally generated outputs.
- [`instructions/narrative/evaluation.md`](instructions/narrative/evaluation.md)
  defines the narrative evaluation method.

`args.json` is the portable runtime contract read by the public pipeline.
Experiment settings are stored under `../src/config/`.
