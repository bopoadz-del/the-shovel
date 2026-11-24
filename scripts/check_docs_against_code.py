#!/usr/bin/env python3
"""Check that what a contributor doc names in backticks still exists in the code.

A doc such as ``AGENTS.md`` tells the next agent "the stop flag lives in
``app/core/ingest_lifecycle``" or "set ``P1B_STALL_WARN_S``". When the code
moves on and the doc does not, the next agent follows a reference that no
longer exists. This check reads every inline code span in the docs it is given
and sorts each one into a claim it can test against the tracked tree:

* **repo path**: a token that contains ``/`` and whose first segment is an
  entry at the repo root (``app/...``, ``scripts/...``, ``.cursor/...``). It
  must exist as a file or directory, as ``<path>.py`` or as a package. A path
  git ignores (a build output, a run-state file) is a runtime artifact, not a
  claim about the code, so it is not checked. A token whose first segment is
  not a root entry (a model id, a unit such as ``AED/m2``, an eslint rule) is
  not a path.
* **source file name**: a bare file name with a source extension
  (``scan_secrets.py``, ``lint.yml``). Some tracked file must have that name.
* **symbol**: an identifier with an underscore (``_predispatch_file_tool``,
  ``DOC_EXTRACT_TIMEOUT_S``) or a CamelCase name (``RunLifecycle``),
  including the name on the left of ``NAME=value``. It must occur as a whole
  word in some tracked file that is not a markdown doc, so a doc never
  confirms itself.

A token with a glob ``*`` or a ``$`` variable is a pattern, not a name, and is
not checked. Anything that is none of the above (a command line, a URL, a
plain word) is not a claim this check can test.

Exit 0 when every claim resolves, 1 with ``doc:line: kind `token``` per stale
reference otherwise. Stdlib only.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Docs read when no doc is named on the command line: the files an agent
#: is told to read before it acts.
DEFAULT_DOCS = ("AGENTS.md", "CLAUDE.md")

#: File name extensions that mark a bare token as a source file name.
SOURCE_EXTENSIONS = frozenset(
    {"py", "yml", "yaml", "sh", "ts", "tsx", "js", "mjs", "toml", "cfg", "ini"}
)

#: Files above this size are not read for symbols (data dumps, lock files).
MAX_SCAN_BYTES = 2_000_000

_SPAN = re.compile(r"`([^`\n]+)`")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SNAKE = re.compile(r"_?[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+_?|_[A-Za-z][A-Za-z0-9]*")
_CAMEL = re.compile(r"[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+")
_ASSIGN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=\S*")
_FILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.([A-Za-z0-9]+)")


@dataclass(frozen=True)
class Claim:
    """One checkable reference a doc makes."""

    doc: str
    line: int
    kind: str  # "path" | "file" | "symbol"
    token: str


@dataclass(frozen=True)
class Tree:
    """What the repo holds, read once."""

    root: Path
    files: frozenset[str]  # tracked paths, posix, relative to root
    root_entries: frozenset[str]
    basenames: frozenset[str]
    dirs: frozenset[str]
    words: frozenset[str]

    @classmethod
    def read(cls, root: Path, files: list[str]) -> Tree:
        tracked = frozenset(f.replace("\\", "/") for f in files)
        dirs: set[str] = set()
        for f in tracked:
            parts = f.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                dirs.add("/".join(parts[:i]))
        words: set[str] = set()
        for f in tracked:
            if f.lower().endswith(".md"):
                continue
            p = root / f
            try:
                if p.stat().st_size > MAX_SCAN_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            words.update(_WORD.findall(text))
        return cls(
            root=root,
            files=tracked,
            root_entries=frozenset(f.split("/")[0] for f in tracked),
            basenames=frozenset(f.rsplit("/", 1)[-1] for f in tracked),
            dirs=frozenset(dirs),
            words=frozenset(words),
        )

    def has_path(self, token: str) -> bool:
        path = token.rstrip("/")
        return (
            path in self.files
            or path in self.dirs
            or f"{path}.py" in self.files
            or f"{path}/__init__.py" in self.files
        )


def claims_in(doc: str, text: str, tree: Tree) -> list[Claim]:
    """Every checkable claim the inline code spans of ``text`` make."""
    out: list[Claim] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for span in _SPAN.findall(line):
            claim = classify(span.strip(), tree)
            if claim is not None:
                kind, token = claim
                out.append(Claim(doc, lineno, kind, token))
    return out


def classify(token: str, tree: Tree) -> tuple[str, str] | None:
    """The kind of claim ``token`` makes, or None when it makes none."""
    if not token or "*" in token or "$" in token or " " in token:
        return None
    if "/" in token:
        first = token.split("/", 1)[0]
        if first and first in tree.root_entries:
            return ("path", token)
        return None
    assign = _ASSIGN.fullmatch(token)
    if assign:
        token = assign.group(1)
    name = _FILE_NAME.fullmatch(token)
    if name and name.group(1).lower() in SOURCE_EXTENSIONS:
        return ("file", token)
    if _SNAKE.fullmatch(token) or _CAMEL.fullmatch(token):
        return ("symbol", token)
    return None


def ignored(root: Path, paths: list[str]) -> set[str]:
    """The subset of ``paths`` that git ignores (runtime artifacts)."""
    if not paths:
        return set()
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", *(p.rstrip("/") for p in paths)],
        cwd=root,
        capture_output=True,
        check=False,
    )
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    return {line.strip() for line in lines if line.strip()}


def stale(claims: list[Claim], tree: Tree) -> list[Claim]:
    """The claims the tree does not back."""
    missing_paths = [c for c in claims if c.kind == "path" and not tree.has_path(c.token)]
    skip = ignored(tree.root, [c.token for c in missing_paths])
    out: list[Claim] = []
    for c in claims:
        if c.kind == "path":
            if not tree.has_path(c.token) and c.token.rstrip("/") not in skip:
                out.append(c)
        elif c.kind == "file":
            if c.token not in tree.basenames:
                out.append(c)
        elif c.token not in tree.words:
            out.append(c)
    return out


def tracked_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    )
    return [f for f in proc.stdout.decode("utf-8", "replace").split("\0") if f]


def run(root: Path, docs: list[str]) -> list[Claim]:
    tree = Tree.read(root, tracked_files(root))
    claims: list[Claim] = []
    for doc in docs:
        p = root / doc
        if not p.is_file():
            continue
        claims.extend(claims_in(doc, p.read_text(encoding="utf-8"), tree))
    return stale(claims, tree)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("docs", nargs="*", default=list(DEFAULT_DOCS))
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args(argv)
    bad = run(Path(args.root), args.docs)
    for c in bad:
        print(f"{c.doc}:{c.line}: {c.kind} `{c.token}` not found in the tracked tree")
    if bad:
        print(f"{len(bad)} stale reference(s)")
        return 1
    print("docs against code: every checkable reference resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
