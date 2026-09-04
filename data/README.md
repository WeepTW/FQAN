# FQAN research data

This directory contains redistributable FinQA material, FQAN-authored transformations, narrative annotations, and lightweight test fixtures. CSV, JSON, JSONL, and XLSX files are stored with Git LFS.

## Load and check the data

Run from the repository root:

```bash
git lfs install
git lfs pull
conda run -n fnqa python -B src/dist/check_data.py --profile public
```

If a file begins with the Git LFS pointer header instead of dataset content, run `git lfs pull` again.

## Included material

| Path | Status and terms |
| --- | --- |
| `src/FinQA/` | FinQA-derived files with public-report email redaction, distributed under MIT; the original copyright notice is preserved in the root `LICENSE`. |
| `finqa_original/`, `finqa_zero_shot/`, `finqa_many_shot/`, `finqa_dynamic_shot/` | FQAN-authored prompt layouts and transformations released under MIT. |
| `src/narratives/` | FQAN-authored narrative annotations released under MIT. |
| `financial_narratives/`, `narratives_gold/`, `testing/` | FQAN-authored research fixtures released under MIT. |
| Python and shell files in this directory | FQAN data-preparation and evaluation utilities released under MIT. |

FinQA records can quote public annual reports and may contain public corporate contact strings. They are source-document content, not FQAN credentials or participant records.

## Material obtained separately

Formal FINDER routes can require these files:

```text
data/src/FINDER/finqa_train_rel_fact_instruction.csv
data/src/FINDER/finqa_dev_rel_fact_instruction.csv
data/src/FINDER/finqa_test_rel_fact_instruction.csv
```

Obtain them from the official FINDER source, confirm that your use is permitted, and place them at the paths above. Then run:

```bash
conda run -n fnqa python -B src/dist/check_data.py --profile formal
```

FinFlier corpora, paired data, third-party chart collections, market workbooks, and participant responses are not published. Obtain required third-party material from its official source and follow its terms. Do not commit it unless redistribution permission is documented.

## Citation

Cite the original FinQA work when using FinQA data. Cite FQAN when using its derived prompts, annotations, or evaluation material.
