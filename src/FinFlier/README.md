# FinFlier dependency

FQAN uses FinFlier as the reference interface for narrative-data bindings. FQAN-owned export and evaluation code is published in `src/finflier_export.py` and `src/narrative/`.

The complete FinFlier source is not copied into this repository because the pinned upstream revision does not provide a repository-level redistribution license. Install the reviewed revision for local research use with:

```bash
bash src/dist/reproduce.sh upstreams
```

The installer checks out FinFlier at revision `3ffeb0963849e4010f21f67dd84ec0a461e1a878` under `src/.external/FinFlier/`. That directory is ignored by Git.

Upstream repository: <https://github.com/CatherineHao/FinFlier>

Review the upstream terms and cite the original work. The FQAN MIT License does not grant rights to the downloaded FinFlier code, corpus, paired data, media, or generated assets.
