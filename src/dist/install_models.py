#!/usr/bin/env python3
"""Download a pinned FQAN model profile into the shared Hugging Face cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


# The standard HTTP downloader is slower but avoids Xet cache/log collisions
# observed when research agents run this command concurrently.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


WORKSPACE = Path(__file__).resolve().parents[2]
MANIFEST_PATH = WORKSPACE / "utils" / "model_manifest.json"
DEFAULT_HF_HOME = WORKSPACE / "utils" / "models" / ".cache" / "huggingface"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "retrievers", "formal_generators"), default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    names = manifest["profiles"][args.profile]
    models = manifest["models"]
    if any(models[name]["requires_token"] for name in names) and not os.environ.get("HF_TOKEN"):
        print("error: HF_TOKEN is required for at least one selected gated model", file=__import__("sys").stderr)
        return 2
    if args.dry_run:
        for name in names:
            print(f"planned: {name} ({models[name]['model_id']})")
        return 0

    from huggingface_hub import snapshot_download

    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)).expanduser()
    cache_dir = hf_home / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        item = models[name]
        try:
            snapshot_download(
                repo_id=item["model_id"],
                revision=item["revision"],
                cache_dir=cache_dir,
                token=os.environ.get("HF_TOKEN"),
                local_files_only=args.local_files_only,
            )
        except Exception as exc:
            print(f"error: {name} download blocked ({exc.__class__.__name__})", file=__import__("sys").stderr)
            return 2
        print(f"ready: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
