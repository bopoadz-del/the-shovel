#!/usr/bin/env python3
"""Read-only audit of the training/evaluation question corpus.

This script inspects the JSONL files under data/learning/ and produces a
markdown report highlighting:

* duplicate instructions
* contradictory labels/rules (PRC-951 example)
* versioned/duplicate shard sets
* rows whose source chunk is missing from the local vector store
* question categories with weak coverage

It does not modify any data.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
LEARNING_DIR = ROOT / "data" / "learning"
RAG_DB = ROOT / "data" / "rag" / "vectors.db"
FORK_DB = ROOT / "data" / "the_fork.db"


PRC_501_QUESTION_PATTERNS = [
    re.compile(r"\bAPPROVED\b", re.IGNORECASE),
    re.compile(r"\bPRC[-\s]?501\b", re.IGNORECASE),
    re.compile(r"design\s+(?:review|status|document)", re.IGNORECASE),
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  ⚠ parse error in {path}: {exc}", file=sys.stderr)
    return rows


def extract_source_doc_id(source: str) -> str | None:
    if not source:
        return None
    if source.startswith("drive_archive:"):
        parts = source.split(":")
        if len(parts) >= 2:
            return parts[1]
    return None


def get_local_chunk_set() -> Set[Tuple[str, int]] | None:
    """Load the set of (doc_id, chunk_index) present in the local vector DB."""
    db_path = RAG_DB if RAG_DB.exists() else (FORK_DB if FORK_DB.exists() else None)
    if not db_path:
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT doc_id, chunk_index FROM chunks WHERE project_id = 'drive_archive'"
        ).fetchall()
        conn.close()
        return {(r[0], int(r[1])) for r in rows}
    except Exception as exc:
        print(f"  ⚠ could not read vector store {db_path}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    if not LEARNING_DIR.exists():
        print(f"Learning directory not found: {LEARNING_DIR}", file=sys.stderr)
        return 1

    jsonl_files = sorted(LEARNING_DIR.glob("*.jsonl"))
    shard_files = sorted(LEARNING_DIR.glob("training_scenarios_v*_shard_*.jsonl"))

    report_lines: List[str] = [
        "# Training / Evaluation Data Quality Audit",
        "",
        f"**Generated:** {os.environ.get('DATE', 'now')}",
        f"**Scope:** read-only scan of `{LEARNING_DIR}`",
        "",
        "## 1. File inventory",
        "",
        "| file | rows |",
        "|---|---|",
    ]

    all_rows: List[Tuple[str, Dict[str, Any]]] = []
    for fp in jsonl_files:
        rows = load_jsonl(fp)
        report_lines.append(f"| `{fp.name}` | {len(rows)} |")
        for r in rows:
            all_rows.append((fp.name, r))

    report_lines.extend([
        "",
        f"**Total rows scanned:** {len(all_rows)}",
        "",
        "## 2. Duplicate instructions",
        "",
    ])

    instr_counter: Counter[str] = Counter()
    for _, row in all_rows:
        instr = (row.get("instruction") or "").strip()
        if instr:
            instr_counter[instr] += 1

    duplicates = [(instr, count) for instr, count in instr_counter.items() if count > 1]
    duplicates.sort(key=lambda x: -x[1])
    report_lines.append(f"**Duplicate instruction strings:** {len(duplicates)} ({sum(c for _, c in duplicates)} total redundant rows)")
    report_lines.append("")
    if duplicates:
        report_lines.extend([
            "| count | instruction preview |",
            "|---|---|",
        ])
        for instr, count in duplicates[:20]:
            preview = instr[:120].replace("|", "\\|")
            report_lines.append(f"| {count} | {preview}... |")
    else:
        report_lines.append("No duplicate instructions found.")

    report_lines.extend([
        "",
        "## 3. Versioned shard sprawl",
        "",
    ])
    if shard_files:
        report_lines.append(f"**Versioned shard files:** {len(shard_files)}")
        report_lines.append("")
        report_lines.extend(["| file | rows |", "|---|---|"])
        for fp in shard_files:
            rows = load_jsonl(fp)
            report_lines.append(f"| `{fp.name}` | {len(rows)} |")
        report_lines.append("")
        report_lines.append(
            "_Note: multiple overlapping versions make it hard to identify the "
            "canonical eval set. Consider retiring old shards and keeping one "
            "reference file._"
        )
    else:
        report_lines.append("No versioned shard files found.")

    report_lines.extend([
        "",
        "## 4. PRC-951 / APPROVED contradiction scan",
        "",
    ])

    prc_rows: List[Tuple[str, Dict[str, Any]]] = []
    for fname, row in all_rows:
        instr = (row.get("instruction") or "") + " " + (row.get("response") or "")
        if any(p.search(instr) for p in PRC_501_QUESTION_PATTERNS):
            prc_rows.append((fname, row))

    report_lines.append(f"**Rows touching PRC-951 / APPROVED / design status:** {len(prc_rows)}")
    report_lines.append("")

    yes_valid: List[Tuple[str, Dict[str, Any]]] = []
    no_valid: List[Tuple[str, Dict[str, Any]]] = []
    for fname, row in prc_rows:
        resp = (row.get("response") or "").lower()
        if re.search(r"\b(is\s+)?valid\b", resp) or re.search(r"\b(is\s+)?allowed\b", resp) or "yes" in resp:
            yes_valid.append((fname, row))
        if "forbidden" in resp or "not valid" in resp or "no," in resp or "prohibited" in resp:
            no_valid.append((fname, row))

    report_lines.append(f"- Rows implying APPROVED is valid/allowed: {len(yes_valid)}")
    report_lines.append(f"- Rows implying APPROVED is forbidden/not valid: {len(no_valid)}")
    report_lines.append("")
    if yes_valid and no_valid:
        report_lines.append(
            "**⚠ Contradiction detected:** the training set contains both "
            "'APPROVED is valid' and 'APPROVED is forbidden' labels for PRC-951. "
            "Reconcile against the authoritative source (construction_knowledge.py / "
            "procedures_db.json) before using this data for fine-tuning or evaluation."
        )
        report_lines.append("")
        report_lines.extend(["### Examples saying APPROVED is valid", ""])
        for fname, row in yes_valid[:5]:
            report_lines.append(f"- `{fname}`: {row.get('instruction', '')[:100]}...")
            report_lines.append(f"  response: {row.get('response', '')[:120]}...")
        report_lines.extend(["", "### Examples saying APPROVED is forbidden", ""])
        for fname, row in no_valid[:5]:
            report_lines.append(f"- `{fname}`: {row.get('instruction', '')[:100]}...")
            report_lines.append(f"  response: {row.get('response', '')[:120]}...")

    report_lines.extend([
        "",
        "## 5. Source chunk coverage",
        "",
    ])

    local_chunks = get_local_chunk_set()
    if local_chunks is None:
        report_lines.append("Local vector store not available; skipping source-coverage check.")
    else:
        report_lines.append(f"Local `drive_archive` chunks: {len(local_chunks):,}")
        report_lines.append("")
        missing_by_file: Dict[str, int] = defaultdict(int)
        total_drive_rows = 0
        missing_drive_rows = 0
        for fname, row in all_rows:
            source = row.get("source") or ""
            doc_id = extract_source_doc_id(source)
            if not doc_id:
                continue
            total_drive_rows += 1
            m = re.match(r"drive_archive:[a-f0-9]+:(\d+)", source)
            if m:
                chunk_idx = int(m.group(1))
                if (doc_id, chunk_idx) not in local_chunks:
                    missing_by_file[fname] += 1
                    missing_drive_rows += 1
        report_lines.append(
            f"**Rows sourced to `drive_archive`:** {total_drive_rows:,} | "
            f"**missing from local chunks:** {missing_drive_rows:,} "
            f"({missing_drive_rows / total_drive_rows * 100:.1f}%)"
        )
        report_lines.append("")
        if missing_by_file:
            report_lines.extend(["| file | missing source chunks |", "|---|---|"])
            for fname, count in sorted(missing_by_file.items(), key=lambda x: -x[1])[:20]:
                report_lines.append(f"| `{fname}` | {count} |")

    report_lines.extend([
        "",
        "## 6. Discipline / category coverage",
        "",
    ])

    discipline_counter: Counter[str] = Counter()
    for _, row in all_rows:
        discipline = row.get("discipline") or row.get("source", "").split(":")[0] or "unspecified"
        discipline_counter[discipline] += 1

    if discipline_counter:
        report_lines.extend(["| category | rows |", "|---|---|"])
        for disc, count in discipline_counter.most_common(20):
            report_lines.append(f"| {disc} | {count} |")
        low_coverage = [(d, c) for d, c in discipline_counter.items() if c < 10]
        if low_coverage:
            report_lines.append("")
            report_lines.append(
                f"**Low-coverage categories (<10 rows):** "
                f"{', '.join(d for d, _ in sorted(low_coverage))}"
            )

    report_lines.extend([
        "",
        "## 7. Recommendations",
        "",
        "1. Pick one canonical training file and archive the versioned shards.",
        "2. Deduplicate instructions before using the file for fine-tuning.",
        "3. Reconcile the PRC-951 / APPROVED contradiction against the procedure DB.",
        "4. For rows with missing source chunks, either re-index the missing docs or remove the rows.",
        "5. Back-fill low-coverage categories with curated examples.",
        "",
    ])

    report = "\n".join(report_lines)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
