# FQAN: Financial Question Answering with Narrative Visualization

FQAN is a research implementation that connects financial question answering with evidence-grounded narrative visualization. It retrieves relevant text and table facts, produces an answer, and binds narrative statements to chart or interface objects. The project supports explainable financial interfaces; it is not a trading or investment-advice system.

## Reproduce the public release

The verified environment is Ubuntu with Python 3.10 in a conda environment named `fnqa`. Formal model runs require a compatible NVIDIA GPU. Access to gated models and paid language-model services remains the researcher's responsibility.

From a clean clone, run these stages in order:

```bash
git clone https://github.com/WeepTW/FQAN.git
cd FQAN
git lfs install

bash src/dist/reproduce.sh requirements
bash src/dist/reproduce.sh environment
bash src/dist/reproduce.sh upstreams
bash src/dist/reproduce.sh data public
bash src/dist/reproduce.sh models smoke
bash src/dist/reproduce.sh experiments smoke
bash src/dist/reproduce.sh experiments dry-run
```

The stages perform the following work:

1. Check that Git, Git LFS, and conda are available.
2. Create or reuse the `fnqa` environment and install the published requirements.
3. Download the pinned FINDER and FinFlier source trees into the ignored `src/.external/` directory.
4. Download Git LFS data and verify the public dataset layout.
5. Install a small model profile used for smoke testing.
6. Run lightweight public tests.
7. Validate the Experiment 0-7 commands without training, paid calls, or generated-code execution.

A missing dataset, model agreement, service credential, upstream file, or compatible GPU is reported as a blocker. The workflow does not silently replace a model, dataset, or experiment route.

An automated coding agent should read this README and `AGENTS.md`, then run the commands above one at a time. It should stop and report the command and error when a stage fails.

## Run experiments

After the smoke test and dry run pass, start a formal route with:

```bash
bash src/dist/reproduce.sh experiments EXPERIMENT_ID
```

Replace `EXPERIMENT_ID` with a number from `0` to `7`.

| ID | Purpose |
| --- | --- |
| 0 | Validate the research setup and required assets. |
| 1 | Train or evaluate the Mistral retriever route. |
| 2 | Train or evaluate the FLAN-T5 retriever route. |
| 3 | Train or evaluate the T5Gemma retriever route. |
| 4 | Compare retriever training loss. |
| 5 | Check generator routing on a small sample. |
| 6 | Generate and evaluate narrative-data bindings. |
| 7 | Generate and execute FinQA answer programs. |

Experiment 6 accepts an optional action after the ID: `auto`, `start`, `resume`, `status`, `evaluate`, `report`, `preflight`, or `smoke`.

```bash
bash src/dist/reproduce.sh experiments 6 preflight
```

Formal runs can require additional datasets, large model downloads, provider approval, or API credentials. The command guide in `src/dist/README.md` lists the direct entrypoints and non-training checks.

## FINDER and FinFlier

FQAN depends on FINDER and uses FinFlier as the visualization reference. Both projects are represented in this repository by pinned source entries, integration code, and component READMEs. Their complete source trees are downloaded locally by the `upstreams` stage.

The pinned upstream repositories did not contain a repository-level license when this release was prepared. Their code is therefore not copied into FQAN and is not covered by the FQAN MIT License. This is necessary for FQAN itself to remain a commercially usable open-source release. Review the upstream terms before using those downloads.

The original FINDER scripts do not contain every later FQAN research modification. Routes that require an unpublished derivative file stop with a clear missing-file error. Exact reproduction of those routes requires permission from the upstream rightsholder or a clean-room open-source replacement. The public smoke and dry-run stages remain usable without claiming that missing code is available.

## Data

Files under `data/` use Git LFS. The release includes redistributable FinQA-derived files with public-report email redaction and FQAN-authored transformations, annotations, and test fixtures. Data whose redistribution rights are uncertain is not included. Follow `data/README.md` for sources, terms, and placement of separately obtained files.

Some FinQA records quote public annual reports and may contain public corporate contact details from those reports. They are research source material, not FQAN credentials or participant data.

Participant responses, private research logs, references, model weights, checkpoints, caches, generated experiment results, and local backup instructions are not published.

## Repository layout

- `src/` contains the FQAN program, configuration, tests, and stable commands.
- `src/FINDER/` and `src/FinFlier/` explain the pinned external components.
- `src/Experiment/` contains only a README; local commands create ignored result directories there.
- `data/` contains public data and preparation utilities.
- `questionnaire/` contains the bilingual study instrument, not participant responses.
- `utils/` contains the model manifest and download guidance, not model files.
- `docs/args.json` is the small portable runtime contract used by the pipeline.

## Credentials and generated code

Credentials must be exported only in the active shell. Never write a token, password, private endpoint, host address, or account name into this repository.

Experiment 7 can execute short Python programs produced by a language model to recover numerical answers. Execution is disabled by default. Enable it only inside an isolated research environment:

```bash
export FQAN_ALLOW_GENERATED_CODE_EXECUTION=1
```

Do not enable this setting on a machine that contains unrelated credentials or sensitive files. Smoke tests, prompt inspection, and formal dry runs do not require it.

## Research scope

FQAN evaluates evidence retrieval, narrative-data binding, and FinQA execution accuracy. It does not establish improved investment decisions, user trust, or financial outcomes. Missing artifacts are reported as unavailable rather than scored as failures.

Generated outputs must be reviewed before use in a publication. Preserve the model identity, prompt mode, dataset split, seed, and effective route with reported results.

## Contributing and security

Contributions should remain focused, include a test for behavioral changes, and avoid committing generated results or external assets. By submitting a contribution, you agree that your FQAN-authored contribution is released under the root MIT License.

Report suspected credential exposure or code-execution vulnerabilities through GitHub's private security-advisory feature. Do not post secrets or private research data in a public issue.

## Citation and license

Until a paper DOI is available, cite the FQAN repository name, release commit, and repository URL. FQAN-authored code, documentation, questionnaires, annotations, and derived data are released under the commercially permissive MIT License in `LICENSE`. Included FinQA files retain their original MIT notice. Downloaded dependencies and upstream projects retain their own terms.
