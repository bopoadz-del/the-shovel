#!/usr/bin/env python3
"""Quick static security scan for blocks.

Scans Python files under app/blocks/ (or a path you pass) for dangerous
patterns. Exits non-zero if any are found, so it can plug into a pre-commit
hook or CI step.

Usage:
    python scripts/security_scan.py
    python scripts/security_scan.py app/blocks/foo.py app/containers/

The list intentionally targets *unsafe* uses — we tolerate `subprocess` and
similar inside well-known sandboxed blocks via a per-file allowlist below.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

DANGEROUS_PATTERNS = [
    (r"\beval\s*\(", "eval()"),
    (r"\bexec\s*\(", "exec()"),
    (r"__import__\s*\(", "__import__()"),
    (r"\bos\.system\s*\(", "os.system()"),
    (r"shell\s*=\s*True", "subprocess(shell=True)"),
    (r"\bctypes\.", "ctypes (native code)"),
    (r"\bpickle\.loads\s*\(", "pickle.loads()"),
    (r"\bmarshal\.loads\s*\(", "marshal.loads()"),
]

# Files where the patterns are reviewed and acceptable.
# Keep this short — additions should be justified.
ALLOWLIST = {
    "app/blocks/sandbox.py",            # the whole point is controlled exec
    "app/blocks/code.py",               # code-execution block, runs in subprocess
    "app/blocks/formula_executor.py",   # restricted formula runner
    "app/blocks/google_drive.py",       # __import__("base64") — lazy stdlib import
    "app/blocks/async_processor.py",    # __import__() — dynamic task dispatch
}


def scan_file(path: Path, repo_root: Path) -> list[str]:
    rel = path.relative_to(repo_root).as_posix()
    if rel in ALLOWLIST:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    findings: list[str] = []
    for pattern, label in DANGEROUS_PATTERNS:
        if re.search(pattern, text):
            findings.append(f"{rel}: {label}")
    return findings


def iter_targets(args: Iterable[str], repo_root: Path) -> Iterable[Path]:
    if not args:
        yield from (repo_root / "app" / "blocks").rglob("*.py")
        return
    for arg in args:
        p = Path(arg)
        if not p.is_absolute():
            p = repo_root / p
        if p.is_dir():
            yield from p.rglob("*.py")
        elif p.suffix == ".py":
            yield p


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    findings: list[str] = []
    for target in iter_targets(sys.argv[1:], repo_root):
        findings.extend(scan_file(target, repo_root))
    if findings:
        print("Security scan: FAIL")
        for f in findings:
            print(f"  - {f}")
        return 1
    print("Security scan: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
