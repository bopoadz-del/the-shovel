"""Decide whether a changed-file list can change what the Docker image does.

Docker Build and Publish skips its image build on a pull request whose every
changed file is a test or a document: neither changes what the built image
serves. Anything else -- product code, requirements, the Dockerfile, a
workflow, a script, the frontend -- builds.

Fail-closed: an empty list, or a list that cannot be read, builds the image.
Knowledge documents under docs/knowledge/ are copied into the image (see
.dockerignore) and seeded at boot, so they are product content, not docs.

Usage: git diff --name-only BASE HEAD | python scripts/ci_image_inputs.py
Prints ``image=true`` or ``image=false`` (a GITHUB_OUTPUT line).
"""

from __future__ import annotations

import sys

_KNOWLEDGE_PREFIX = "docs/knowledge/"
_NON_IMAGE_PREFIXES = ("tests/", "docs/")


def is_non_image_file(path: str) -> bool:
    """True when ``path`` is a test or a document outside the knowledge base."""
    p = path.strip().replace("\\", "/")
    if not p:
        return False
    if p.startswith(_KNOWLEDGE_PREFIX):
        return False
    if p.startswith(_NON_IMAGE_PREFIXES):
        return True
    return p.lower().endswith(".md")


def image_needed(paths: list[str]) -> bool:
    """Build unless there is at least one file and every file is non-image."""
    files = [p for p in (x.strip() for x in paths) if p]
    if not files:
        return True
    return not all(is_non_image_file(p) for p in files)


def main() -> int:
    paths = sys.stdin.read().splitlines()
    print(f"image={'true' if image_needed(paths) else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
