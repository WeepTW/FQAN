# Public environment variable examples

Set only the variables required by the model route you intend to run. Values belong in the active shell or a local ignored environment file; never commit a real credential or private endpoint.

## Hugging Face access

```bash
export HF_TOKEN="<HF_TOKEN>"
```

`HF_TOKEN` is needed only for gated models after their provider terms have been accepted.

## OpenAI-compatible services

```bash
export OPENAI_API_KEY="<OPENAI_API_KEY>"
export OPENAI_BASE_URL="<OPENAI_BASE_URL>"  # optional compatible service
```

For Azure OpenAI, use the route-specific variables reported by the experiment preflight:

```bash
export AZURE_OPENAI_API_KEY="<AZURE_OPENAI_API_KEY>"
export AZURE_OPENAI_ENDPOINT="<AZURE_OPENAI_ENDPOINT>"
export AZURE_OPENAI_DEPLOYMENT="<AZURE_OPENAI_DEPLOYMENT>"
```

## Local model services

Local OpenAI-compatible endpoints should bind to `localhost` unless the operator has configured authentication and network controls.

```bash
export VLLM_BASE_URL="http://localhost:<PORT>/v1"
export VLLM_API_KEY="EMPTY"
```

## Generated-code execution

Experiment 7 generated-code execution is disabled by default. Enable it only inside an isolated research environment:

```bash
export FQAN_ALLOW_GENERATED_CODE_EXECUTION=1
```

Unset credentials after the run. Private NAS variables are intentionally not documented in the public release.
