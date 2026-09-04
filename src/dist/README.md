# FQAN command guide

Run commands from the repository root. The supported conda environment is `fnqa`.

## Complete preparation sequence

```bash
bash src/dist/reproduce.sh requirements
bash src/dist/reproduce.sh environment
bash src/dist/reproduce.sh upstreams
bash src/dist/reproduce.sh data public
bash src/dist/reproduce.sh models smoke
bash src/dist/reproduce.sh experiments smoke
bash src/dist/reproduce.sh experiments dry-run
```

The `upstreams` stage checks out pinned FINDER and FinFlier revisions under `src/.external/`. Their source is not committed to FQAN. The public data stage resolves Git LFS files. The smoke model profile downloads only the small matching model needed for lightweight checks.

## Formal entrypoints

| Experiment | Command | Non-training check |
| --- | --- | --- |
| 0 | `bash src/dist/reproduce.sh experiments 0` | `SETUP_DOWNLOAD_MODE=none SETUP_ARTIFACTS_REQUIRED=0 SETUP_STRICT=0 bash src/dist/experiment_setup.sh` |
| 1 | `bash src/dist/reproduce.sh experiments 1` | `RUN_TRAIN=0 RUN_INFER=0 RUN_MATCH=0 bash src/dist/experiment_1_mistral_retriever.sh` |
| 2 | `bash src/dist/reproduce.sh experiments 2` | `RUN_TRAIN=0 RUN_INFER=0 RUN_MATCH=0 bash src/dist/experiment_2_flan_retriever.sh` |
| 3 | `bash src/dist/reproduce.sh experiments 3` | `RUN_TRAIN=0 RUN_INFER=0 RUN_MATCH=0 bash src/dist/experiment_3_t5gemma_retriever.sh` |
| 4 | `bash src/dist/reproduce.sh experiments 4` | `LOSS_EXPLORATORY_ACK=1 RUN_TRAIN=0 RUN_INFER=0 RUN_MATCH=0 bash src/dist/experiment_4_retriever_loss_diagram.sh` |
| 5 | `bash src/dist/reproduce.sh experiments 5` | `RUN_EXECUTE=0 bash src/dist/experiment_5_qwen_few10_smoke.sh` |
| 6 | `bash src/dist/reproduce.sh experiments 6 ACTION` | `bash src/dist/reproduce.sh experiments 6 preflight` |
| 7 | `bash src/dist/reproduce.sh experiments 7` | `PUBLIC_PREFLIGHT_ONLY=1 PREFLIGHT_ONLY=1 WAIT_BEFORE_START_SECONDS=0 bash src/dist/experiment_7_formal_tmux_run.sh` |

Experiment 6 actions are `auto`, `start`, `resume`, `status`, `evaluate`, `report`, `preflight`, and `smoke`.

Formal routes stop when required data, models, services, or programs are unavailable. Some exact FINDER-derived research changes cannot be redistributed until compatible permission exists; the affected route reports the missing program instead of silently using a different implementation.

## Public checks

```bash
conda run -n fnqa python -B src/dist/test_experiment6_paths.py
conda run -n fnqa python -B src/dist/test_finqa_target_execution.py
```

A formal result should record its configuration, seed, prompt mode, model identity, and effective route. Generated outputs and checkpoints belong under `src/Experiment/` and must not be committed.
