#!/usr/bin/env python3
"""Owner-side: write ``config/live_name_hashes.txt`` from the live database.

CI cannot read the production database, so ``scripts/scan_hardwiring.py
--live-names`` only runs on the owner's machine. This script gives CI an
offline form of the same check without putting a single live name in git: it
reads the live document names, document ids and project ids (read-only), picks
the identifying forms with the SAME rules as ``scan_hardwiring.leakage_findings``
(distinctive name phrases, file-name forms, reference codes, 8-char document-id
prefixes, customer project ids), normalizes each one, and writes only SALTED,
truncated SHA-256 digests. ``scripts/scan_repo_hygiene.py`` hashes the repo's
text the same way and fails on a hit.

``DATABASE_URL`` is read from ``~/.thefork-backup/neon.env`` (or the
environment) and never printed. The salt is kept across regenerations so a
refresh diff shows only names that came or went; ``--new-salt`` rotates it.

Usage (owner machine, then commit the regenerated file):
    python scripts/gen_live_name_hashes.py
    python scripts/gen_live_name_hashes.py --env-file PATH --new-salt
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scan_hardwiring as sh
from scan_repo_hygiene import HASH_FILE, digest, normalize_words

WIDTH = 20  # hex chars kept per digest: 80 bits, collision-free at this scale
DEFAULT_ENV = Path.home() / ".thefork-backup" / "neon.env"

#: Where the PRODUCT authors document names itself: the seeded knowledge files
#: (their file names become document names in the general-knowledge project)
#: and the synthetic fixture generators (their output is uploaded by evals). A
#: live name found here came from the repo, not from a client, so it is not
#: hashed. Paths ending in "/" contribute their tracked file NAMES; files
#: contribute their text.
REPO_AUTHORED = ("docs/knowledge/", "scripts/synthetic_fixtures.py", "scripts/seed_fixtures.py")


def repo_authored_text(root: Path = ROOT) -> str:
    """Normalized ``" w1 w2 ... "`` of every name the repo itself authors."""
    parts: list[str] = []
    for src in REPO_AUTHORED:
        p = root / src
        if src.endswith("/"):
            if p.is_dir():
                parts += [f.name for f in sorted(p.rglob("*")) if f.is_file()]
        elif p.is_file():
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return " " + " | ".join(" ".join(normalize_words(x)) for x in parts) + " "


def read_database_url(env_file: Path) -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    if not env_file.is_file():
        raise SystemExit(f"no DATABASE_URL in the environment and no {env_file}")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        line = line.removeprefix("export ")
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{env_file} has no DATABASE_URL line")


def records(live: dict, system_ids: set[str],
            authored: str = " ") -> tuple[set[str], set[tuple[int, str]]]:
    """``(tokens, word_sequences)`` -- the normalized identifying forms.

    ``authored`` is ``repo_authored_text()``: a name whose words occur there
    was written by the product (seed / synthetic fixture), not by a client.
    """
    tokens: set[str] = set()
    seqs: set[tuple[int, str]] = set()

    def add_seq(words: list[str]) -> None:
        if len(words) >= 3:
            seqs.add((len(words), " ".join(words)))

    for name in live.get("documents", []):
        stem = os.path.splitext(name)[0]
        w = normalize_words(stem)
        if w and f" {' '.join(w)} " in authored:
            continue
        if len(w) >= 3 and len(" ".join(w)) >= 15:
            if sh._distinctive_name(w):
                add_seq(w)
            else:
                for form in sh._file_name_forms(name):
                    if any(ch.isspace() for ch in form):
                        add_seq(normalize_words(form))
                    else:
                        tokens.add(form.lower())
        tokens.update(c.lower() for c in sh._REF_CODE.findall(stem))
    for d in (str(x) for x in live.get("document_ids", [])):
        if len(d) >= 8:
            tokens.add(d[:8].lower())
    for p in live.get("projects", []):
        if len(p) >= 6 and p not in system_ids:
            tokens.add(p.lower())
    return tokens, seqs


def existing_salt(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("salt "):
            return line.split(" ", 1)[1].strip()
    return None


def render(salt: str, tokens: set[str], seqs: set[tuple[int, str]]) -> str:
    lines = [
        "# Salted SHA-256 digests of live document names, reference codes, document",
        "# ids and project ids -- never the names. Generated owner-side by",
        "# scripts/gen_live_name_hashes.py; checked by scripts/scan_repo_hygiene.py.",
        "# Do not edit by hand.",
        f"salt {salt}",
        f"width {WIDTH}",
    ]
    anchors = sorted({digest(salt, "A", " ".join(s.split()[:3]), WIDTH) for _, s in seqs})
    lines += [f"A {h}" for h in anchors]
    lines += sorted(f"W{n} {digest(salt, f'W{n}', s, WIDTH)}" for n, s in seqs)
    lines += sorted(f"T {digest(salt, 'T', t, WIDTH)}" for t in tokens)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-file", default=str(DEFAULT_ENV))
    ap.add_argument("--out", default=str(ROOT / HASH_FILE))
    ap.add_argument("--new-salt", action="store_true")
    args = ap.parse_args(argv)

    os.environ["DATABASE_URL"] = read_database_url(Path(args.env_file))
    live = sh.load_live_names()
    out = Path(args.out)
    salt = (None if args.new_salt else existing_salt(out)) or secrets.token_hex(16)
    tokens, seqs = records(live, sh.system_project_ids(ROOT), repo_authored_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(salt, tokens, seqs), encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT).as_posix()}: {len(seqs)} name sequences, "
          f"{len(tokens)} tokens (from {len(live.get('documents', []))} names, "
          f"{len(live.get('document_ids', []))} ids, {len(live.get('projects', []))} projects)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
