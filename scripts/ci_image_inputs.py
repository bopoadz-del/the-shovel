"""Decide whether a changed-file list can change what the Docker image does.

Docker Build and Publish skips its image build on a pull request whose every
changed file is outside the image:

* a test under ``tests/`` (copied in by ``COPY . .`` but never run or read
  by the serving app), or
* a file the repo's own ``.dockerignore`` keeps out of the build context.
  The patterns are read from that file at run time with Docker's matching
  rules (root-anchored, ``*`` does not cross ``/``, ``**`` does, a pattern
  that matches a directory covers everything under it, ``!`` re-includes,
  the last matching pattern wins). So a ``*.md`` rule excludes ``README.md``
  but not ``config/rules.md``, which the image does contain.

Files under ``.github/`` are kept out of the context but define how the image
is built and checked, so they always build.

Fail-closed: an empty list builds; a missing ``.dockerignore`` excludes
nothing, so every non-test file builds.

The file list must come from ``git diff --name-only --no-renames``: with
rename detection a file moved out of product code is listed only by its new
path, and the move would look like a test-only change.

Usage: git diff --name-only --no-renames BASE HEAD | python scripts/ci_image_inputs.py
Prints ``image=true`` or ``image=false`` (a GITHUB_OUTPUT line).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TEST_PREFIX = "tests/"
_BUILD_DEFINITION_PREFIX = ".github/"


def _norm(path: str) -> str:
    return path.strip().replace("\\", "/")


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """Docker .dockerignore pattern -> regex over a /-separated path."""
    out = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
            continue
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z")


def parse_dockerignore(text: str) -> list[tuple[re.Pattern[str], bool]]:
    """(regex, is_exception) per pattern line, in file order."""
    rules = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        exception = line.startswith("!")
        if exception:
            line = line[1:].strip()
        line = line.lstrip("/").rstrip("/")
        if line:
            rules.append((_pattern_regex(line), exception))
    return rules


def excluded_from_context(path: str, rules: list[tuple[re.Pattern[str], bool]]) -> bool:
    """True when the last pattern that matches ``path`` or one of its parent
    directories excludes it (Docker's MatchesOrParentMatches)."""
    parts = path.split("/")
    prefixes = ["/".join(parts[: k + 1]) for k in range(len(parts))]
    excluded = False
    for rx, exception in rules:
        if any(rx.match(p) for p in prefixes):
            excluded = not exception
    return excluded


def _load_rules(root: Path = _ROOT) -> list[tuple[re.Pattern[str], bool]]:
    """Patterns of ``root/.dockerignore``; an unreadable file excludes nothing
    (so every non-test file builds) and says so on stderr."""
    try:
        return parse_dockerignore((root / ".dockerignore").read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"ci_image_inputs: .dockerignore not read ({exc}); every non-test file builds",
              file=sys.stderr)
        return []


def is_non_image_file(path: str, rules: list[tuple[re.Pattern[str], bool]] | None = None) -> bool:
    """True when ``path`` cannot change what the built image serves."""
    p = _norm(path)
    if not p:
        return False
    if p.startswith(_TEST_PREFIX):
        return True
    if p.startswith(_BUILD_DEFINITION_PREFIX):
        return False
    return excluded_from_context(p, _load_rules() if rules is None else rules)


def image_needed(paths: list[str], rules: list[tuple[re.Pattern[str], bool]] | None = None) -> bool:
    """Build unless there is at least one file and every file is non-image."""
    files = [p for p in (_norm(x) for x in paths) if p]
    if not files:
        return True
    if rules is None:
        rules = _load_rules()
    return not all(is_non_image_file(p, rules) for p in files)


def main() -> int:
    paths = sys.stdin.read().splitlines()
    print(f"image={'true' if image_needed(paths) else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
