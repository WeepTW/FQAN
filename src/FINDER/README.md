# FINDER dependency

FQAN uses FINDER for financial evidence retrieval and in-context selection.

The complete FINDER source is not copied into this repository because the pinned upstream revision does not provide a repository-level redistribution license. Install the reviewed revision for local research use with:

```bash
bash src/dist/reproduce.sh upstreams
```

The installer checks out FINDER at revision `68b85b56d48a4294e35f706ad425a8002bdcdb93` under `src/.external/FINDER/`. That directory is ignored by Git.

Upstream repository: <https://github.com/subhendukhatuya/FINDER_POT_Financial_Numeric_Reasoning>

Review the upstream terms and cite the original work. The FQAN MIT License does not grant rights to the downloaded FINDER code. Some later FQAN retriever changes are not in the upstream revision and cannot be redistributed without compatible permission; affected formal routes report this as a blocker.
