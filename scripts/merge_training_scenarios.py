#!/usr/bin/env python3
"""Merge all training-scenario sources into a single JSONL with re-tagged
sources, then validate.

Inputs (any subset; missing files are skipped):

  data/learning/evm_scenarios.jsonl        -> source_label="evm_knowledge"
  data/learning/expert_scenarios.jsonl     -> source_label="procedure_knowledge"
  data/learning/knowledge_scenarios.jsonl  -> source_label="knowledge_functions"
  data/learning/training_scenarios_*.jsonl -> source_label="project_docs" or
                                              "boq_docs" depending on project_id

The merger:
  1. Loads every row from every input file.
  2. Re-tags the ``source`` field with the high-level label requested by the
     operator (the original generator's source tag is preserved in
     ``source_detail`` so traceability survives).
  3. Runs the same validation pipeline used by the LLM-driven generator
     (``scripts.generate_training_scenarios._validate_scenarios``): drops
     empty rows, drops short rows, drops near-duplicate (cosine 0.85+)
     instructions.
  4. Writes the kept rows to ``data/learning/training_scenarios_merged.jsonl``.
  5. Reports per-source counts before/after, drop reasons, and 3 sample
     rows per source.

CLI:

  python scripts/merge_training_scenarios.py \\
      --out data/learning/training_scenarios_merged.jsonl

  # Add an LLM-generated project-docs file with explicit labeling:
  python scripts/merge_training_scenarios.py \\
      --project-docs data/learning/training_scenarios_c0ac2b2d_*.jsonl \\
      --boq-docs data/learning/training_scenarios_3f6f28b2_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import re

from scripts.generate_training_scenarios import _validate_scenarios

DEFAULT_INPUTS: Dict[str, str] = {
    "evm_knowledge": "data/learning/evm_scenarios.jsonl",
    "procedure_knowledge": "data/learning/expert_scenarios.jsonl",
    "knowledge_functions": "data/learning/knowledge_scenarios.jsonl",
}


def _normalize_instruction(text: str) -> str:
    """Normalize an instruction for cross-source dedupe.

    Catches near-duplicates from different sources (e.g. the same BOQ row
    showing up in two generator runs with slightly different whitespace or
    truncation). Drops case, collapses internal whitespace, strips edge
    whitespace + trailing punctuation. Doesn't touch the response — that's
    the cosine pass's job.
    """
    s = text or ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip().rstrip(".?!,;: ")
    return s


def _dedupe_by_instruction(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    """First-pass dedupe by normalized instruction hash. Returns
    ``(kept, dropped_count)``. Earlier-pool rows win; later rows lose, so
    sources passed first to the merger (knowledge, procedures, evm)
    anchor the dataset and LLM-generated rows fill in around them."""
    seen: Dict[str, str] = {}
    kept: List[Dict[str, str]] = []
    dropped = 0
    for r in rows:
        norm = _normalize_instruction(r.get("instruction", ""))
        if not norm:
            kept.append(r)
            continue
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if digest in seen:
            dropped += 1
            continue
        seen[digest] = r.get("source", "?")
        kept.append(r)
    return kept, dropped


def _load_jsonl(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  warn: {path}:{i + 1} unparseable JSON ({exc})", file=sys.stderr)
    return rows


def _retag(rows: Iterable[Dict[str, str]], source_label: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        # Preserve the original generator's source under source_detail so the
        # fine-tune QA can trace any row back to its function / section.
        original = r.get("source", "")
        out.append({
            "instruction": r.get("instruction", ""),
            "response": r.get("response", ""),
            "source": source_label,
            "source_detail": original,
        })
    return out


def _summarise(label: str, before: int, kept: List[Dict[str, str]]) -> Dict[str, object]:
    return {
        "source": label,
        "before": before,
        "after": len(kept),
        "drop_count": before - len(kept),
        "drop_rate_pct": round((before - len(kept)) * 100 / before, 1) if before else 0.0,
        "samples": kept[:3],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/learning/training_scenarios_merged.jsonl")
    parser.add_argument("--project-docs", action="append", default=[],
                        help="Glob(s) matching LLM-generated project-doc JSONL files to tag "
                             "as 'project_docs'.")
    parser.add_argument("--boq-docs", action="append", default=[],
                        help="Glob(s) for BOQ-doc JSONL files tagged 'boq_docs'.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the dedupe/short-row validation pipeline.")
    args = parser.parse_args()

    # Combined pool — re-tag each input set so the merged JSONL carries the
    # high-level label requested in the spec.
    by_source_before: Dict[str, int] = {}
    by_source_kept: Dict[str, List[Dict[str, str]]] = {}
    pool: List[Dict[str, str]] = []

    for label, default_path in DEFAULT_INPUTS.items():
        rows = _load_jsonl(default_path)
        by_source_before[label] = len(rows)
        pool.extend(_retag(rows, label))

    for glob_pattern in args.project_docs:
        files = sorted(glob.glob(glob_pattern))
        rows: List[Dict[str, str]] = []
        for f in files:
            rows.extend(_load_jsonl(f))
        by_source_before["project_docs"] = by_source_before.get("project_docs", 0) + len(rows)
        pool.extend(_retag(rows, "project_docs"))

    for glob_pattern in args.boq_docs:
        files = sorted(glob.glob(glob_pattern))
        rows: List[Dict[str, str]] = []
        for f in files:
            rows.extend(_load_jsonl(f))
        by_source_before["boq_docs"] = by_source_before.get("boq_docs", 0) + len(rows)
        pool.extend(_retag(rows, "boq_docs"))

    if not pool:
        print("error: no input rows found", file=sys.stderr)
        return 1

    # Stage A: dedupe by normalized instruction text. Catches near-duplicate
    # questions across sources (same BOQ row paraphrased in two generator
    # runs, or whitespace/truncation drift). The existing _validate_scenarios
    # only dedupes by response cosine, so two rows asking the same question
    # with slightly different answers would BOTH survive without this pass.
    pool, dropped_instructions = _dedupe_by_instruction(pool)

    # Stage B: validate (empties / short responses / response-cosine dupes).
    if args.no_validate:
        kept_all = pool
        validation_report = {"validation": "skipped",
                             "instruction_dupes_dropped": dropped_instructions}
    else:
        kept_all, validation_report = _validate_scenarios(pool)
        validation_report["instruction_dupes_dropped"] = dropped_instructions

    # Group kept rows back by source label for the report.
    for label in by_source_before:
        by_source_kept[label] = [r for r in kept_all if r.get("source") == label]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summaries: List[Dict[str, object]] = []
    for label in by_source_before:
        summaries.append(_summarise(label, by_source_before[label], by_source_kept[label]))

    total_before = sum(by_source_before.values())
    total_after = len(kept_all)

    report = {
        "total_before": total_before,
        "total_after": total_after,
        "global_drop_pct": round((total_before - total_after) * 100 / total_before, 1)
                           if total_before else 0.0,
        "validation": validation_report,
        "output_path": os.path.abspath(args.out),
        "by_source": summaries,
    }

    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
