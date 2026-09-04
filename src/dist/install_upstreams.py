#!/usr/bin/env python3
"""Install pinned FINDER and FinFlier checkouts for local research use.

The upstream projects are downloaded into src/.external and remain untracked.
Researchers are responsible for reviewing and following the upstream terms.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


class InstallError(RuntimeError):
    pass


SRC_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = SRC_ROOT / "config" / "upstreams.json"
EXTERNAL_ROOT = SRC_ROOT / ".external"


def run_git(*args: str, cwd: Path | None = None) -> str:
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise InstallError("Git operation failed; inspect network access and the upstream repository")
    return result.stdout.strip()


def load_config() -> dict[str, object]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("components"), dict):
        raise InstallError("invalid upstream configuration")
    return payload


def install(name: str, item: dict[str, object], *, dry_run: bool) -> None:
    url = str(item["url"])
    revision = str(item["revision"])
    destination = (SRC_ROOT / str(item["destination"])).resolve()
    if destination.parent != EXTERNAL_ROOT.resolve():
        raise InstallError(f"unsafe destination configured for {name}")
    if dry_run:
        print(f"{name}: planned at pinned revision {revision}")
        return
    if destination.exists():
        if not (destination / ".git").is_dir():
            raise InstallError(f"{name}: destination exists but is not a Git checkout")
        actual = run_git("rev-parse", "HEAD", cwd=destination)
        dirty = run_git("status", "--porcelain", cwd=destination)
        if actual != revision or dirty:
            raise InstallError(f"{name}: existing checkout is not clean at the pinned revision")
        print(f"{name}: already ready")
        return

    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=EXTERNAL_ROOT) as raw:
        checkout = Path(raw) / "checkout"
        run_git("clone", "--no-checkout", url, str(checkout))
        run_git("checkout", "--detach", revision, cwd=checkout)
        actual = run_git("rev-parse", "HEAD", cwd=checkout)
        if actual != revision:
            raise InstallError(f"{name}: checked-out revision does not match the manifest")
        checkout.rename(destination)
    print(f"{name}: installed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("all", "finder", "finflier"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    components = load_config()["components"]
    assert isinstance(components, dict)
    names = sorted(components) if args.component == "all" else [args.component]
    for name in names:
        item = components[name]
        if not isinstance(item, dict):
            raise InstallError(f"invalid configuration for {name}")
        install(name, item, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InstallError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
