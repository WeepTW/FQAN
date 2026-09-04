# Model installation and local runtime assets

The public `utils/` surface contains this guide and `model_manifest.json`. Model weights, Hugging Face caches, GGUF files, tokenizers copied from third parties, checkpoints, and local runtimes are not published or backed up.

## Profiles

- `smoke`: matching encoder only; suitable for release validation.
- `retrievers`: matching encoder plus FLAN-T5, Mistral, and T5Gemma 2 retriever bases.
- `formal_generators`: the large generator models used by later experiment routes.

Inspect the manifest before a large download:

```bash
conda run -n fnqa python -B src/dist/install_models.py --profile smoke --dry-run
conda run -n fnqa python -B src/dist/install_models.py --profile smoke
```

For a gated profile, accept the model provider's terms first and export `HF_TOKEN` only in the active shell. The token must not be written to the repository, conda configuration, README files, or the manifest.

```bash
export HF_TOKEN="<HF_TOKEN>"
conda run -n fnqa python -B src/dist/install_models.py --profile retrievers
unset HF_TOKEN
```

Downloads use the pinned revisions in `model_manifest.json` and the shared cache below `utils/models/.cache/huggingface`. To verify an existing cache without network access:

```bash
conda run -n fnqa python -B src/dist/install_models.py --profile smoke --local-files-only
```

Every model remains governed by its upstream license and model card. Review those terms before use or redistribution.
