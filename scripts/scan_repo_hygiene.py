#!/usr/bin/env python3
"""Fail the build when the repository carries documents, data or live names.

Documents never live in git. Codes and knowledge go in through the admin path
into the database; model weights are release assets pinned by sha256
(``data/models/manifest.json``, fetched by ``scripts/fetch_model.py``). Only
text seeds (``docs/knowledge/*.md`` etc.) ship with the code. This gate keeps
it that way. Over ``git ls-files`` it fails on:

  BINARY      a tracked image, PDF, office document, mail item, archive,
              database or model file -- by extension OR by content (magic
              bytes, so an extension-less ``data/file_<hash>`` still counts) --
              unless the path is in ``ALLOWED_BINARIES`` below.
  SIZE        a tracked file above ``MAX_BYTES`` (500 KB) unless the path is
              in ``ALLOWED_LARGE`` with its own ceiling.
  LIVE-NAME   a live document name, document-code-shaped token, document id
              or project id from the production database. CI cannot read the
              database, so the owner generates ``config/live_name_hashes.txt``
              (``scripts/gen_live_name_hashes.py``): SALTED SHA-256 digests of
              the normalized names -- never the names themselves. This scan
              hashes every word n-gram and raw token of every tracked text file
              (and every path) the same way and reports a digest hit as
              ``path:line`` only. ``--show`` prints the matched text locally.

Both allow-lists are reviewed lists of exact paths, never globs: a new binary
fixture is a deliberate, visible decision in a diff, and it must be a tiny
GENERATED test file, never a real document. An allow-list entry whose path is
no longer tracked is reported as stale (warning, so parallel cleanups can land
in either order) -- delete it.

Stdlib only. Exit 1 with a finding per line.

Usage:
    python scripts/scan_repo_hygiene.py           # CI check
    python scripts/scan_repo_hygiene.py --show    # also print matched live text (local only)
    python scripts/scan_repo_hygiene.py --write-baseline   # only after REMOVING names
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASH_FILE = "config/live_name_hashes.txt"
#: Grandfathered live-name hits ``{path: {digest: count}}`` -- digests only --
#: on the same contract as ``scripts/hardwiring_baseline.json``: it may only
#: SHRINK. A hit not in it, or a count above it, fails. It exists because the
#: identifier clean-up of tests/docs lands separately; regenerate it with
#: ``--write-baseline`` only after REMOVING names, never to admit one, and
#: delete the file once it is empty.
BASELINE_FILE = "config/live_name_baseline.json"
MAX_BYTES = 500 * 1000

# Extensions that are never source: images, PDF, office/mail, archives,
# databases, model weights, CAD.
BLOCKED_SUFFIXES = frozenset([
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".ico", ".svg",
    ".heic", ".heif", ".psd", ".pdf", ".ps", ".eps", ".doc", ".docx", ".docm", ".dot",
    ".dotx", ".xls", ".xlsx", ".xlsm", ".xlsb", ".xlt", ".ppt", ".pptx", ".pptm",
    ".odt", ".ods", ".odp", ".odg", ".rtf", ".pages", ".numbers", ".key", ".vsd",
    ".vsdx", ".mpp", ".msg", ".eml", ".pst", ".zip", ".tar", ".gz", ".tgz", ".bz2",
    ".xz", ".7z", ".rar", ".zst", ".jar", ".whl", ".iso", ".dmg", ".db", ".sqlite",
    ".sqlite3", ".db3", ".mdb", ".accdb", ".parquet", ".feather", ".npy", ".npz",
    ".pkl", ".pickle", ".joblib", ".onnx", ".pt", ".pth", ".ckpt", ".safetensors",
    ".h5", ".hdf5", ".tflite", ".pb", ".engine", ".gguf", ".bin", ".model", ".dwg",
    ".dxf", ".rvt", ".rfa", ".nwd", ".nwc",
])

# Content signatures, so a renamed or extension-less binary is still caught.
MAGIC = (
    (b"%PDF", "PDF"),
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"GIF87a", "GIF image"),
    (b"GIF89a", "GIF image"),
    (b"RIFF", "RIFF media (webp/wav/avi)"),
    (b"II*\x00", "TIFF image"),
    (b"MM\x00*", "TIFF image"),
    (b"PK\x03\x04", "zip container (docx/xlsx/pptx/zip)"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "OLE container (doc/xls/ppt/msg)"),
    (b"SQLite format 3\x00", "SQLite database"),
    (b"\x1f\x8b", "gzip archive"),
    (b"7z\xbc\xaf\x27\x1c", "7z archive"),
    (b"Rar!\x1a\x07", "RAR archive"),
    (b"BZh", "bzip2 archive"),
    (b"\xfd7zXZ\x00", "xz archive"),
)

#: Reviewed binary allow-list: exact paths, each a tiny GENERATED test input or
#: a UI asset. Never a real drawing, contract, photo or schedule.
ALLOWED_BINARIES: dict[str, str] = {
    "frontend/public/favicon.svg": "UI asset (302 B)",
    "tests/fixtures/formats/sample.doc": "generated Word 97 sample for the legacy-format extractor",
    "tests/fixtures/formats/sample.msg": "generated Outlook item for the .msg extractor",
    "tests/fixtures/formats/sample.ppt": "generated PowerPoint 97 sample for the legacy-format extractor",
    "tests/fixtures/formats/sample.rtf": "generated RTF sample for the extractor",
    "tests/fixtures/formats/sample.xls": "generated Excel 97 sample for the legacy-format extractor",
    # Being replaced by a synthetic fixture set in a parallel PR; delete these
    # entries when it lands (they then show up as stale).
    "tests/fixtures/drawing_tm_200.pdf": "pending replacement by a synthetic drawing",
    "tests/fixtures/drawing_tm_9900001.pdf": "pending replacement by a synthetic drawing",
    **{f"tests/fixtures/ingest_shard_sample/construction-3-001/docs/file_{n}.{ext}":
       "pending replacement by a synthetic ingest shard"
       for n, ext in ((1, "docx"), (2, "xlsx"), (4, "png"), (8, "pdf"), (9, "docx"),
                      (10, "xlsx"), (12, "png"), (16, "pdf"), (17, "docx"), (18, "xlsx"),
                      (20, "png"), (24, "pdf"), (25, "docx"), (26, "xlsx"), (28, "png"),
                      (32, "pdf"), (33, "docx"), (34, "xlsx"), (36, "png"), (40, "pdf"))},
    **{f"tests/fixtures/ingest_shard_sample/construction-3-001/drawings/dwg_{n}.pdf":
       "pending replacement by a synthetic ingest shard" for n in range(1, 21)},
}

#: Reviewed size allow-list: exact path -> its own ceiling in bytes.
ALLOWED_LARGE: dict[str, int] = {
    # Source module, not data. The ceiling stops it growing further; split it
    # rather than raising the number.
    "app/agents/runtime.py": 800 * 1000,
    # Pending replacement by synthetic fixtures in a parallel PR.
    "tests/fixtures/drawing_tm_200.pdf": 3_900_000,
    "tests/fixtures/ohdd_baseline_2013.xer": 1_700_000,
}

_WORD = re.compile(r"[a-z0-9]+")
_TOKEN = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?")


def normalize_words(text: str) -> list[str]:
    """Lower-case alphanumeric words: ``A-101_Ground Floor`` -> a 101 ground floor."""
    return _WORD.findall(text.lower())


def raw_tokens(text: str) -> list[str]:
    """Lower-case runs of ``[a-z0-9._-]`` (codes, ids, file names without spaces)."""
    return _TOKEN.findall(text.lower())


def digest(salt: str, kind: str, value: str, width: int) -> str:
    return hashlib.sha256(f"{salt}|{kind}|{value}".encode()).hexdigest()[:width]


class LiveHashes:
    """The committed digest set (see ``scripts/gen_live_name_hashes.py``).

    File format, one record per line, ``#`` comments ignored::

        salt <hex>
        width <hex chars kept per digest>
        A <digest>          first three words of a word-sequence name (anchor)
        W<n> <digest>       a whole n-word name sequence
        T <digest>          a raw token (code, id, file name without spaces)
    """

    def __init__(self, salt: str, width: int, anchors: set[str],
                 seqs: dict[int, set[str]], tokens: set[str]) -> None:
        self.salt, self.width = salt, width
        self.anchors, self.seqs, self.tokens = anchors, seqs, tokens

    @classmethod
    def load(cls, path: Path) -> LiveHashes | None:
        if not path.is_file():
            return None
        salt, width = "", 64
        anchors: set[str] = set()
        seqs: dict[int, set[str]] = {}
        tokens: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            kind, _, value = line.partition(" ")
            if kind == "salt":
                salt = value
            elif kind == "width":
                width = int(value)
            elif kind == "A":
                anchors.add(value)
            elif kind == "T":
                tokens.add(value)
            elif kind.startswith("W") and kind[1:].isdigit():
                seqs.setdefault(int(kind[1:]), set()).add(value)
        if not salt:
            raise SystemExit(f"{path}: no salt line")
        return cls(salt, width, anchors, seqs, tokens)

    def h(self, kind: str, value: str) -> str:
        return digest(self.salt, kind, value, self.width)

    def hits(self, text: str) -> list[tuple[str, str]]:
        """``(digest, matched text)`` for every live form in ``text`` (one line)."""
        found: list[tuple[str, str]] = []
        for tok in sorted(set(raw_tokens(text)) | set(normalize_words(text))):
            d = self.h("T", tok)
            if d in self.tokens:
                found.append((d, tok))
        if self.seqs:
            words = normalize_words(text)
            lengths = sorted(self.seqs, reverse=True)
            for i in range(len(words) - 2):
                if self.h("A", " ".join(words[i:i + 3])) not in self.anchors:
                    continue
                for n in lengths:
                    if i + n <= len(words):
                        seq = " ".join(words[i:i + n])
                        d = self.h(f"W{n}", seq)
                        if d in self.seqs[n]:
                            found.append((d, seq))
                            break
        return found


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True,
                         capture_output=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def sniff(head: bytes) -> str | None:
    for sig, label in MAGIC:
        if head.startswith(sig):
            return label
    return None


def binary_kind(rel: str, path: Path) -> str | None:
    """What makes ``path`` a non-source file, or None for text."""
    with open(path, "rb") as f:
        head = f.read(4096)
    suffix = Path(rel).suffix.lower()
    label = sniff(head)
    if label:
        return label
    if suffix in BLOCKED_SUFFIXES:
        return f"{suffix} file"
    if b"\x00" in head:
        return "binary file"
    return None


def live_hits(root: Path, files: list[str], hashes: LiveHashes) -> dict[str, list[tuple[int, str, str]]]:
    """``{path: [(line, digest, text), ...]}`` over text files; line 0 is the path itself."""
    out: dict[str, list[tuple[int, str, str]]] = {}
    for rel in sorted(set(files)):
        if rel in (HASH_FILE, BASELINE_FILE):
            continue
        path = root / rel
        if not path.is_file() or binary_kind(rel, path):
            continue
        found = [(0, d, t) for d, t in hashes.hits(rel)]
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            found += [(lineno, d, t) for d, t in hashes.hits(line)]
        if found:
            out[rel] = found
    return out


def hit_counts(hits: dict[str, list[tuple[int, str, str]]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for rel, found in sorted(hits.items()):
        per: dict[str, int] = {}
        for _, d, _ in found:
            per[d] = per.get(d, 0) + 1
        out[rel] = dict(sorted(per.items()))
    return out


def scan(root: Path, files: list[str], hashes: LiveHashes | None,
         allowed_binaries: dict[str, str] | None = None,
         allowed_large: dict[str, int] | None = None,
         show: bool = False,
         baseline: dict[str, dict[str, int]] | None = None) -> tuple[list[str], list[str]]:
    """Return ``(findings, warnings)`` for ``files`` (repo-relative, posix)."""
    allowed_binaries = ALLOWED_BINARIES if allowed_binaries is None else allowed_binaries
    allowed_large = ALLOWED_LARGE if allowed_large is None else allowed_large
    findings: list[str] = []
    warnings: list[str] = []
    present = set(files)
    for rel in sorted(present):
        path = root / rel
        if not path.is_file():
            continue  # deleted in the working tree / submodule
        size = path.stat().st_size
        ceiling = allowed_large.get(rel, MAX_BYTES)
        if size > ceiling:
            findings.append(f"SIZE {rel}: {size} bytes > {ceiling}")
        kind = binary_kind(rel, path)
        if kind and rel not in allowed_binaries:
            findings.append(f"BINARY {rel}: {kind} -- documents, data and models never live in git")
    if hashes is not None:
        baseline = baseline or {}
        hits = live_hits(root, files, hashes)
        counts = hit_counts(hits)
        for rel, found in hits.items():
            allowed = baseline.get(rel, {})
            over = {d for d, n in counts[rel].items() if n > allowed.get(d, 0)}
            for lineno, d, text in found:
                if d in over:
                    where = f"{rel}:{lineno}" if lineno else f"{rel} (the path itself)"
                    detail = f" [{text}]" if show else f" ({len(text)} chars)"
                    findings.append(f"LIVE-NAME {where}: live document name / code / id{detail}")
        for rel, per in sorted(baseline.items()):
            for d, n in sorted(per.items()):
                if counts.get(rel, {}).get(d, 0) < n:
                    warnings.append(f"live-name baseline can shrink ({rel} {d}): run --write-baseline")
    for rel in sorted(set(allowed_binaries) - present):
        warnings.append(f"stale ALLOWED_BINARIES entry (no longer tracked): {rel}")
    for rel in sorted(set(allowed_large) - present):
        warnings.append(f"stale ALLOWED_LARGE entry (no longer tracked): {rel}")
    return findings, warnings


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    show = "--show" in argv
    hashes = LiveHashes.load(ROOT / HASH_FILE)
    if hashes is None:
        print(f"::error::{HASH_FILE} missing -- the live-name check cannot run", file=sys.stderr)
        return 1
    files = tracked_files(ROOT)
    if "--write-baseline" in argv:
        counts = hit_counts(live_hits(ROOT, files, hashes))
        (ROOT / BASELINE_FILE).write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n",
                                          encoding="utf-8", newline="\n")
        print(f"wrote {BASELINE_FILE}: {sum(map(len, counts.values()))} entries in {len(counts)} files")
        return 0
    bpath = ROOT / BASELINE_FILE
    baseline = json.loads(bpath.read_text(encoding="utf-8")) if bpath.is_file() else {}
    findings, warnings = scan(ROOT, files, hashes, show=show, baseline=baseline)
    for w in warnings:
        print(f"::warning::{w}")
    for f in findings:
        print(f)
    if findings:
        print(f"\n{len(findings)} repo-hygiene finding(s). Documents and data go in "
              "through the admin path into the database; models are release assets "
              "(data/models/manifest.json). See scripts/scan_repo_hygiene.py.", file=sys.stderr)
        return 1
    print("repo hygiene: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
