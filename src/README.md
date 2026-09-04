# FQAN source

This directory contains the program layer used by the public research workflow. Run supported operations from the repository root through `src/dist/reproduce.sh`.

## Main areas

- `dist/` contains setup, validation, data, model, and Experiment 0-7 entrypoints.
- `config/` contains public model, route, evaluation, and path settings.
- `narrative/` contains FQAN narrative prompts, schemas, and evaluation code.
- `FINDER/` and `FinFlier/` contain attribution and local installation guidance. Complete upstream trees are installed under the ignored `src/.external/` directory.
- `Experiment/` is the local result root. GitHub keeps only its README.

Top-level Python modules connect retrieval, answer generation, numerical evaluation, narrative binding, and export.

## Start here

```bash
bash src/dist/reproduce.sh requirements
bash src/dist/reproduce.sh environment
bash src/dist/reproduce.sh upstreams
bash src/dist/reproduce.sh data public
bash src/dist/reproduce.sh models smoke
bash src/dist/reproduce.sh experiments smoke
bash src/dist/reproduce.sh experiments dry-run
```

See `dist/README.md` for formal experiment commands.

## Safety and licensing

Credentials are read only from environment variables. Generated Python execution remains disabled unless `FQAN_ALLOW_GENERATED_CODE_EXECUTION=1` is explicitly set in an isolated environment.

FQAN-authored source is covered by the root MIT License. Downloaded FINDER and FinFlier files retain third-party rights and are not part of the licensed FQAN repository.
