#!/usr/bin/env python3
"""Fetch the detector weights from this repo's GitHub release and verify them.

The weights are not git objects (repo hygiene: no model, document or database
file is tracked). ``data/models/manifest.json`` pins the release tag and the
sha256 + size of every file; this script puts each file at
``data/models/<name>`` and exits non-zero unless every file matches its pin.
The Dockerfile COPYs ``data/models/safety_world_v2.onnx`` from the build
context, so every ``docker build`` (deploy-aws.yml, docker-publish.yml, a dev
box) runs this first.

A file already on disk with the right checksum is kept (no download). A file
with the WRONG checksum is never used: it is re-downloaded, and if the fresh
copy still mismatches the script fails -- a corrupted or swapped asset stops
the build instead of shipping.

Download goes through the GitHub CLI (``gh release download``), which reads
``GH_TOKEN`` / ``GITHUB_TOKEN`` in CI and the developer's ``gh auth`` login
locally. The token is never printed.

Usage:
    python scripts/fetch_model.py            # fetch what is missing, verify all
    python scripts/fetch_model.py --check    # verify only, never download
    python scripts/fetch_model.py --image    # only files the image needs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "models" / "manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, entry: dict) -> str | None:
    """None when ``path`` matches the pin, else the reason it does not."""
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if "size" in entry and size != entry["size"]:
        return f"size {size} != pinned {entry['size']}"
    digest = sha256_of(path)
    if digest != entry["sha256"]:
        return f"sha256 {digest} != pinned {entry['sha256']}"
    return None


def download(repo: str, tag: str, name: str, dest_dir: Path) -> Path:
    gh = shutil.which("gh")
    if gh is None:
        raise SystemExit("fetch_model: the GitHub CLI (gh) is required to download release assets")
    subprocess.run(
        [gh, "release", "download", tag, "--repo", repo, "--pattern", name,
         "--dir", str(dest_dir), "--clobber"],
        check=True,
    )
    return dest_dir / name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify only, never download")
    ap.add_argument("--image", action="store_true", help="only files with required_for_image")
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = manifest_path.parent
    failures: list[str] = []
    for entry in manifest["files"]:
        if args.image and not entry.get("required_for_image"):
            continue
        target = out_dir / entry["name"]
        reason = verify(target, entry)
        if reason is None:
            print(f"ok       {entry['name']} (sha256 {entry['sha256'][:12]}...)")
            continue
        if args.check:
            failures.append(f"{entry['name']}: {reason}")
            continue
        print(f"fetch    {entry['name']} from release {manifest['release_tag']} ({reason})")
        with tempfile.TemporaryDirectory() as tmp:
            got = download(manifest["repo"], manifest["release_tag"], entry["name"], Path(tmp))
            reason = verify(got, entry)
            if reason is not None:
                failures.append(f"{entry['name']}: downloaded asset {reason}")
                continue
            if target.exists():
                target.unlink()
            shutil.move(str(got), str(target))
        print(f"verified {entry['name']} (sha256 {entry['sha256'][:12]}...)")
    if failures:
        for f in failures:
            print(f"::error::model checksum check failed -- {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
