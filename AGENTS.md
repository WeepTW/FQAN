# FQAN Agent Instructions

Use this file together with the root `README.md` when an automated coding agent prepares or runs FQAN.

## Required order

Run the public workflow one command at a time from the repository root:

```bash
bash src/dist/reproduce.sh requirements
bash src/dist/reproduce.sh environment
bash src/dist/reproduce.sh upstreams
bash src/dist/reproduce.sh data public
bash src/dist/reproduce.sh models smoke
bash src/dist/reproduce.sh experiments smoke
bash src/dist/reproduce.sh experiments dry-run
```

Stop on a nonzero exit status. Report the failed command and its error without exposing environment-variable values.

## Boundaries

- Use only the conda environment `fnqa`.
- Treat `README.md`, `src/dist/README.md`, and the JSON files under `src/config/` as the public operating instructions.
- Keep FINDER and FinFlier downloads under `src/.external/`.
- Keep model files and caches under `utils/models/`.
- Keep generated results under `src/Experiment/`.
- Never commit `src/.external/`, `utils/models/`, Experiment outputs, logs, caches, local backup files, or editor settings.
- Read credentials only from environment variables. Never print or persist their values.
- Do not substitute a model, dataset, prompt mode, or route when a required dependency is missing.
- Keep generated-code execution disabled unless the researcher explicitly enables it in an isolated environment.

## Release rules

The public repository uses one root `LICENSE`. Markdown files are limited to `README.md` and `AGENTS.md`. Before proposing a release, scan filenames and file contents for credentials, personal home paths, email addresses, private hosts, and numeric IP addresses. Report findings by filename and issue type without printing any secret value.

FINDER and FinFlier remain external downloads because their pinned upstream revisions do not provide repository-level redistribution licenses. Do not copy their source into a release unless written redistribution permission or a compatible upstream license is documented.
