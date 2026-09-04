"""Check Hugging Face gated-model access without logging credentials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import get_token, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe config/tokenizer access for a Hugging Face model.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE") or (str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else os.environ.get("TRANSFORMERS_CACHE")))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--require-weights",
        action="store_true",
        help="Also verify the model.safetensors weight file is available in cache or downloadable.",
    )
    return parser.parse_args()


def _special_token_content(value):
    if isinstance(value, dict):
        return value.get("content")
    return value


def load_tokenizer_json_fallback(model_id: str, kwargs: dict):
    tokenizer_file = hf_hub_download(model_id, "tokenizer.json", **kwargs)
    special_tokens = {}
    try:
        special_tokens_file = hf_hub_download(model_id, "special_tokens_map.json", **kwargs)
    except Exception:
        special_tokens_file = None
    if special_tokens_file:
        data = json.loads(Path(special_tokens_file).read_text(encoding="utf-8"))
        for key in ("bos_token", "eos_token", "unk_token", "pad_token", "mask_token"):
            value = _special_token_content(data.get(key))
            if value:
                special_tokens[key] = value
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, **special_tokens)


def exception_chain_text(exc: BaseException) -> str:
    parts = []
    current: BaseException | None = exc
    while current is not None:
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return "\n".join(parts)


def validate_config_metadata_fallback(model_id: str, kwargs: dict) -> None:
    last_exc: BaseException | None = None
    for filename in ("config.json", "params.json"):
        try:
            path = hf_hub_download(model_id, filename, **kwargs)
            json.loads(Path(path).read_text(encoding="utf-8"))
            return
        except Exception as exc:  # noqa: BLE001 - preserve original access error context.
            last_exc = exc
    raise RuntimeError(f"Cannot access config.json or params.json for {model_id}.") from last_exc


def require_weight_file(model_id: str, kwargs: dict) -> None:
    last_exc: BaseException | None = None
    for filename in ("model.safetensors", "model.safetensors.index.json", "consolidated.safetensors.index.json"):
        try:
            hf_hub_download(model_id, filename, **kwargs)
            return
        except (HfHubHTTPError, LocalEntryNotFoundError) as exc:
            last_exc = exc
    message = exception_chain_text(last_exc) if last_exc else ""
    if "public gated repositories" in message or "403 Forbidden" in message:
        raise RuntimeError(
            f"Cannot download safetensors metadata for {model_id}. "
            "The active Hugging Face fine-grained token can read metadata but is not allowed "
            "to access public gated model files. Enable public gated repository access for "
            "that token, or export a read token that has accepted access to this model."
        ) from last_exc
    raise RuntimeError(
        f"Cannot download safetensors metadata for {model_id}. "
        "Verify network connectivity, local HF cache, and Hugging Face token permissions."
    ) from last_exc


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()
    kwargs = {
        "cache_dir": args.cache_dir,
        "token": token,
        "local_files_only": args.local_files_only,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        AutoConfig.from_pretrained(args.model_id, **kwargs)
    except (OSError, ValueError) as exc:
        message = str(exc)
        fallback_markers = ("Unrecognized model", "model_type", "couldn't connect", "couldn't find", "config.json")
        if not any(marker in message for marker in fallback_markers):
            raise
        validate_config_metadata_fallback(args.model_id, kwargs)
    try:
        AutoTokenizer.from_pretrained(args.model_id, **kwargs)
    except (OSError, ValueError) as exc:
        tokenizer_fallback_markers = ("Couldn't instantiate the backend tokenizer", "couldn't connect", "couldn't find", "config.json")
        if not any(marker in str(exc) for marker in tokenizer_fallback_markers):
            raise
        try:
            AutoTokenizer.from_pretrained(args.model_id, use_fast=False, **kwargs)
        except (OSError, ValueError) as fallback_exc:
            if not any(marker in str(fallback_exc) for marker in tokenizer_fallback_markers):
                raise
            load_tokenizer_json_fallback(args.model_id, kwargs)
    if args.require_weights:
        require_weight_file(args.model_id, kwargs)
    print(f"hf_model_access_ok model_id={args.model_id} hf_token_present={int(bool(token))}", flush=True)


if __name__ == "__main__":
    main()
