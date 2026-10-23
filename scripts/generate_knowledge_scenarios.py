#!/usr/bin/env python3
"""Generate deterministic Q&A training pairs from app/core/construction_knowledge.py.

These pairs come from running the actual code (or reading the actual rule
text), not from an LLM. Zero hallucination risk — the answers are exactly
what the production code returns. Use them as a "rules baked in"
supplement to the document-driven training_scenarios.jsonl.

Output schema matches scripts/generate_training_scenarios.py:

    {"instruction": "<question>", "response": "<answer>", "source": "..."}

Sources are tagged ``construction_knowledge.py:<section>`` so the operator
can trace any row back to the function or constant it was derived from.

CLI:

    python scripts/generate_knowledge_scenarios.py \\
        --out data/learning/knowledge_scenarios.jsonl

    # Or merge straight into an existing document-driven file:
    python scripts/generate_knowledge_scenarios.py \\
        --out data/learning/training_scenarios.jsonl --append
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, Iterator, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import construction_knowledge as ck


# ── helpers ────────────────────────────────────────────────────────────────

def _row(instruction: str, response: str, source: str) -> Dict[str, str]:
    return {
        "instruction": instruction.strip(),
        "response": response.strip(),
        "source": f"construction_knowledge.py:{source}",
    }


# ── CRITICAL_RULES (3 Q&A per entry) ──────────────────────────────────────

def gen_critical_rules() -> Iterator[Dict[str, str]]:
    for rule_id, entry in ck.CRITICAL_RULES.items():
        rule_text = entry["rule"]
        procedure = entry.get("procedure", "")
        violation = entry["violation_message"]
        section = f"CRITICAL_RULES.{rule_id}"

        yield _row(
            f"What is the contract rule about '{rule_id.replace('_', ' ')}' in our construction procedures?",
            f"{rule_text} See procedure {procedure}.",
            section,
        )
        yield _row(
            f"If someone violates the '{rule_id.replace('_', ' ')}' rule, what message should the system surface?",
            violation,
            section,
        )
        yield _row(
            f"Which controlled procedure governs the '{rule_id.replace('_', ' ')}' rule?",
            f"Procedure {procedure} (see CRITICAL_RULES in construction_knowledge.py).",
            section,
        )


# ── enforce_critical_rules — exercise the live detector ───────────────────

def gen_enforce_rules() -> Iterator[Dict[str, str]]:
    cases: List[Tuple[str, str]] = [
        ("The design package was approved by the consultant yesterday.",
         "live detector input — design context with 'approved'"),
        ("The architectural drawing has been approved for issue.",
         "live detector input — drawing context with 'approved'"),
        ("The construction schedule was approved by the project manager.",
         "live detector input — no design keyword, should NOT flag"),
        ("Materials approval is pending from the procurement team.",
         "live detector input — procurement context, no design keyword"),
    ]
    for text, label in cases:
        violations = ck.enforce_critical_rules(text)
        if violations:
            v = violations[0]
            response = (
                f"The text triggers rule '{v['rule_id']}' (procedure {v.get('procedure')}). "
                f"Violation message: {v['violation_message']}"
            )
        else:
            response = "No critical-rule violations detected in this text."
        yield _row(
            f"Does the following statement trigger a critical-rule violation? Statement: \"{text}\"",
            response,
            "enforce_critical_rules",
        )


# ── generate_doc_number — every documented doc type × representative seqs ─

_DOC_TYPES_NO_YEAR = ["RFI", "IR", "VO", "RFM", "JR", "DD", "PR", "WP"]
_DOC_TYPES_YEAR = ["NCR", "PDN"]
_SAMPLE_SEQS = [1, 5, 23, 42, 100, 999]
_SAMPLE_YEARS = [2024, 2025, 2026]


def gen_doc_numbers() -> Iterator[Dict[str, str]]:
    for dt in _DOC_TYPES_NO_YEAR:
        for seq in _SAMPLE_SEQS:
            number = ck.generate_doc_number(dt, seq)
            yield _row(
                f"What is the correctly-formatted document number for {dt} sequence {seq}?",
                f"The document number is '{number}'.",
                f"generate_doc_number({dt})",
            )
    for dt in _DOC_TYPES_YEAR:
        for year in _SAMPLE_YEARS:
            for seq in (1, 7, 42, 200):
                number = ck.generate_doc_number(dt, seq, year=year)
                yield _row(
                    f"Generate the document number for {dt} sequence {seq} in year {year}.",
                    f"The document number is '{number}'.",
                    f"generate_doc_number({dt}+year)",
                )
    # Unknown doc type → fallback formatting.
    for unknown in ("ABC", "ZZZ"):
        number = ck.generate_doc_number(unknown, 7)
        yield _row(
            f"If a project uses an unrecognised document type '{unknown}', "
            f"what number does generate_doc_number produce for sequence 7?",
            f"It produces '{number}' (the fallback 4-digit zero-padded format).",
            f"generate_doc_number(fallback)",
        )


# ── VALID_DESIGN_STATUSES + FORBIDDEN_DESIGN_STATUSES + validate_design_status ─

def gen_design_statuses() -> Iterator[Dict[str, str]]:
    for status in sorted(ck.VALID_DESIGN_STATUSES):
        ok, msg = ck.validate_design_status(status)
        yield _row(
            f"Is '{status}' a valid design review status under PRC-951?",
            f"Yes — '{status}' is a valid status. {msg}",
            "validate_design_status(valid)",
        )
    for status in sorted(ck.FORBIDDEN_DESIGN_STATUSES):
        ok, msg = ck.validate_design_status(status)
        yield _row(
            f"Is '{status}' acceptable on a design document under PRC-951?",
            f"No — '{status}' is forbidden on design documents per PRC-951. {msg}",
            "validate_design_status(forbidden)",
        )
    # Case-insensitivity and dash/underscore tolerance.
    for variant in ("approved", "for-comment", "buy off"):
        ok, msg = ck.validate_design_status(variant)
        verdict = "valid" if ok else "not valid"
        yield _row(
            f"If a user types '{variant}' as a design status, is it accepted?",
            f"It is normalised (case + space + dash) and the validator says it is {verdict}. {msg}",
            "validate_design_status(normalisation)",
        )
    # Unknown status.
    for unknown in ("WIP", "DRAFT_2", "REVIEWED"):
        ok, msg = ck.validate_design_status(unknown)
        yield _row(
            f"Does '{unknown}' qualify as a valid design status?",
            f"No — {msg}",
            "validate_design_status(unknown)",
        )


# ── check_review_timeline — boundary values ──────────────────────────────

def gen_review_timeline() -> Iterator[Dict[str, str]]:
    cases = [
        ("2026-01-01", "2026-01-02", "1-day window (well under minimum)"),
        ("2026-01-01", "2026-01-06", "5-day window (under PRC-951 minimum)"),
        ("2026-01-01", "2026-01-07", "exactly 6 days (still under, boundary)"),
        ("2026-01-01", "2026-01-08", "7-day window (PRC-951 minimum compliant)"),
        ("2026-01-01", "2026-01-09", "8-day window (just over the minimum)"),
        ("2026-01-01", "2026-01-15", "14-day window (PRC-951 maximum recommended)"),
        ("2026-01-01", "2026-01-16", "15-day window (one day over the maximum)"),
        ("2026-01-01", "2026-01-21", "20-day window (over PRC-951 maximum)"),
        ("2026-06-15", "2026-06-20", "5 days during a real project window"),
        ("2026-09-01", "2026-09-08", "7 days in a real project window"),
    ]
    for dist, ws, label in cases:
        ok, msg = ck.check_review_timeline(dist, ws)
        verdict = "compliant" if ok else "non-compliant"
        yield _row(
            f"A design package was distributed on {dist} and the review workshop is scheduled for {ws}. "
            f"Is the timeline compliant with PRC-951?",
            f"It is {verdict}. {msg}",
            "check_review_timeline",
        )


# ── NCR workflow + dispositions ──────────────────────────────────────────

def gen_ncr_workflow() -> Iterator[Dict[str, str]]:
    for i, status in enumerate(ck.NCR_WORKFLOW_SEQUENCE):
        nxt = ck.next_ncr_status(status)
        if nxt is None:
            response = (
                f"'{status}' is the terminal state in the NCR workflow — there is no next status "
                f"(NCR is closed)."
            )
        else:
            response = (
                f"The next status after '{status}' in the NCR workflow is '{nxt}' "
                f"(per PRC-942 NCR_WORKFLOW_SEQUENCE)."
            )
        yield _row(
            f"In the PRC-942 NCR workflow, what status follows '{status}'?",
            response,
            "NCR_WORKFLOW_SEQUENCE",
        )
    # Invalid status.
    yield _row(
        "If a user reports an NCR status that does not exist in PRC-942 (e.g. 'PENDING_REVIEW'), "
        "what does next_ncr_status return?",
        "It returns None — the input is not part of NCR_WORKFLOW_SEQUENCE and has no successor.",
        "next_ncr_status(unknown)",
    )
    # Dispositions.
    for disp in sorted(ck.VALID_NCR_DISPOSITIONS):
        ok, msg = ck.validate_ncr_disposition(disp)
        yield _row(
            f"Is '{disp}' a valid NCR disposition under PRC-942?",
            f"Yes. {msg}",
            "validate_ncr_disposition(valid)",
        )
    for disp in ("CONCEDE", "ACCEPT", "REWORK"):
        ok, msg = ck.validate_ncr_disposition(disp)
        yield _row(
            f"Is '{disp}' a valid NCR disposition under PRC-942?",
            f"No. {msg}",
            "validate_ncr_disposition(invalid)",
        )


# ── score_risk — boundary grid ────────────────────────────────────────────

def gen_score_risk() -> Iterator[Dict[str, str]]:
    # Cover the entire 5x5 grid — there are only 25 combinations and the
    # band boundaries (4 GREEN ceiling, 5 AMBER floor, 9 AMBER ceiling,
    # 10 RED floor) are the ones the model most needs anchored.
    grid_samples: List[Tuple[int, int]] = [
        (p, i) for p in range(1, 6) for i in range(1, 6)
    ]
    for p, i in grid_samples:
        result = ck.score_risk(p, i)
        action = "requires escalation per PRC-932" if result["requires_action"] else "no escalation required"
        yield _row(
            f"What is the PRC-932 risk score for probability={p} and impact={i}?",
            f"Score is {result['score']}/25 — band {result['band']}, {action}.",
            "score_risk",
        )
    # Out-of-range guard.
    bad = ck.score_risk(6, 3)
    yield _row(
        "What does score_risk return when probability is 6 (out of the 1-5 range)?",
        f"It returns an error dict: {bad.get('error')}.",
        "score_risk(invalid)",
    )


# ── calculate_payment — typical PRC-965 scenarios ────────────────────────

def gen_payment() -> Iterator[Dict[str, str]]:
    cases = [
        (1_000_000, 950_000, 0.05, 0.0, 10_000_000),
        (500_000, 500_000, 0.10, 1_500_000, 5_000_000),
        (250_000, 200_000, 0.05, 0.0, 2_500_000),
        (750_000, 720_000, 0.05, 3_000_000, 12_000_000),
        (2_000_000, 1_800_000, 0.05, 5_000_000, 20_000_000),
        (100_000, 100_000, 0.0, 0.0, 1_000_000),
        (400_000, 350_000, 0.075, 800_000, 4_000_000),
        (1_500_000, 1_200_000, 0.05, 8_000_000, 15_000_000),
    ]
    for claimed, certified, retention, prev_cum, contract in cases:
        r = ck.calculate_payment(claimed, certified, retention, prev_cum, contract)
        yield _row(
            f"A contractor claims {claimed:,.0f} on a payment application. The Engineer certifies "
            f"{certified:,.0f}. Retention is {retention * 100:.1f}%, cumulative previously "
            f"certified is {prev_cum:,.0f}, contract value is {contract:,.0f}. "
            f"What is the net payment due and the percent complete?",
            f"Net payment due is {r['net_payment_due']:,.2f} (retention held: "
            f"{r['retention_held']:,.2f}). Disputed amount is {r['disputed_amount']:,.2f}. "
            f"Project is {r['percent_complete']}% complete.",
            "calculate_payment",
        )


# ── calculate_evm — CPI / SPI scenarios ──────────────────────────────────

def gen_evm() -> Iterator[Dict[str, str]]:
    cases = [
        (1_000_000, 500_000, 600_000, 550_000, "behind schedule, over budget"),
        (1_000_000, 600_000, 500_000, 550_000, "ahead of schedule, under budget"),
        (2_000_000, 1_000_000, 1_000_000, 1_000_000, "on time and on budget"),
        (5_000_000, 2_400_000, 2_500_000, 2_700_000, "slightly behind, over budget"),
        (3_000_000, 1_800_000, 1_500_000, 1_900_000, "ahead but slightly over"),
        (10_000_000, 7_500_000, 8_000_000, 7_200_000, "near completion, behind"),
        (500_000, 200_000, 250_000, 220_000, "early-stage, marginal"),
        (4_000_000, 2_000_000, 2_000_000, 2_400_000, "on schedule, materially over"),
        (1_500_000, 900_000, 1_000_000, 850_000, "behind but under-running cost"),
    ]
    for bac, bcwp, bcws, acwp, label in cases:
        r = ck.calculate_evm(bac, bcwp, bcws, acwp)
        yield _row(
            f"Calculate the EVM metrics for a project with BAC={bac:,.0f}, BCWP={bcwp:,.0f}, "
            f"BCWS={bcws:,.0f}, ACWP={acwp:,.0f}. What are CPI, SPI, and the cost/schedule status?",
            f"CPI={r['CPI']}, SPI={r['SPI']}. Cost status: {r['status']['cost']}. "
            f"Schedule status: {r['status']['schedule']}. CPI health: {r['status']['cpi_health']}. "
            f"Estimate at Completion (EAC) is {r['EAC']:,.2f}, Variance at Completion (VAC) is {r['VAC']:,.2f}.",
            "calculate_evm",
        )


# ── evaluate_tender — small representative panels ──────────────────────────

def gen_tender() -> Iterator[Dict[str, str]]:
    panels: List[Tuple[str, List[Dict[str, Any]]]] = [
        (
            "three balanced bidders",
            [
                {"name": "Bidder A", "technical_score": 85, "commercial_score": 75, "hse_score": 90, "local_content_score": 60},
                {"name": "Bidder B", "technical_score": 78, "commercial_score": 82, "hse_score": 85, "local_content_score": 70},
                {"name": "Bidder C", "technical_score": 70, "commercial_score": 88, "hse_score": 80, "local_content_score": 75},
            ],
        ),
        (
            "two-bidder shortlist with HSE tiebreaker",
            [
                {"name": "Alpha Eng", "technical_score": 80, "commercial_score": 80, "hse_score": 95, "local_content_score": 50},
                {"name": "Beta Build", "technical_score": 80, "commercial_score": 80, "hse_score": 70, "local_content_score": 50},
            ],
        ),
        (
            "four-bidder panel where commercial dominates",
            [
                {"name": "ContraCo", "technical_score": 90, "commercial_score": 55, "hse_score": 85, "local_content_score": 80},
                {"name": "BuildLab", "technical_score": 75, "commercial_score": 95, "hse_score": 70, "local_content_score": 40},
                {"name": "SiteWorks", "technical_score": 80, "commercial_score": 90, "hse_score": 75, "local_content_score": 55},
                {"name": "MEP Group", "technical_score": 82, "commercial_score": 80, "hse_score": 80, "local_content_score": 65},
            ],
        ),
        (
            "two-bidder local-content tiebreaker",
            [
                {"name": "Bidder A", "technical_score": 78, "commercial_score": 82, "hse_score": 80, "local_content_score": 95},
                {"name": "Dubai Civils", "technical_score": 80, "commercial_score": 82, "hse_score": 80, "local_content_score": 40},
            ],
        ),
    ]
    for label, tenderers in panels:
        result = ck.evaluate_tender(tenderers)
        winner = result["recommended"]
        ranking = ", ".join(
            f"{t['rank']}. {t['name']} ({t['weighted_total']})"
            for t in result["ranked_tenderers"]
        )
        yield _row(
            f"Per PRC-963, evaluate the following {label}: "
            + "; ".join(
                f"{t['name']} — technical {t['technical_score']}, commercial {t['commercial_score']}, "
                f"HSE {t['hse_score']}, local content {t.get('local_content_score', 0)}"
                for t in tenderers
            )
            + ". Which bidder is recommended and what is the ranking?",
            f"Recommended bidder: {winner['name']} (weighted total {winner['weighted_total']}). "
            f"Full ranking: {ranking}. Weights applied per PRC-963 default: "
            f"technical 45%, commercial 45%, HSE 7%, local content 3%.",
            "evaluate_tender",
        )


# ── get_procedure / get_system_prompt — knowledge-base lookups ────────────

_KNOWN_PROCEDURES = [
    "PRC-921", "PRC-931", "PRC-932", "PRC-942", "PRC-945", "PRC-946",
    "PRC-951", "PRC-952", "PRC-961", "PRC-962", "PRC-963", "PRC-964",
    "PRC-965", "PRC-966",
]


def gen_procedure_lookup() -> Iterator[Dict[str, str]]:
    for proc in _KNOWN_PROCEDURES:
        info = ck.get_procedure(proc)
        if not info:
            yield _row(
                f"What does the knowledge base say about procedure {proc}?",
                f"The procedure {proc} is not present in the loaded knowledge base.",
                "get_procedure(missing)",
            )
            continue
        title = info.get("title") or info.get("name") or proc
        purpose = info.get("purpose") or info.get("description") or ""
        # Keep the response short and grounded.
        body = f"{proc}: {title}."
        if purpose:
            body += f" Purpose: {purpose}"
        yield _row(
            f"What is procedure {proc} in the construction knowledge base?",
            body,
            f"get_procedure({proc})",
        )
    # System prompt presence check.
    sys_prompt = ck.get_system_prompt() or ""
    yield _row(
        "Does construction_knowledge.py expose a system prompt for downstream agents, and how long is it?",
        (
            f"Yes — ck.get_system_prompt() returns a {len(sys_prompt)}-character prompt "
            f"that downstream construction agents prepend to their conversations."
            if sys_prompt
            else "ck.get_system_prompt() returns an empty string in this build."
        ),
        "get_system_prompt",
    )


# ── orchestrator ──────────────────────────────────────────────────────────

_GENERATORS: List[Tuple[str, Callable[[], Iterator[Dict[str, str]]]]] = [
    ("critical_rules", gen_critical_rules),
    ("enforce_critical_rules", gen_enforce_rules),
    ("doc_numbers", gen_doc_numbers),
    ("design_statuses", gen_design_statuses),
    ("review_timeline", gen_review_timeline),
    ("ncr_workflow", gen_ncr_workflow),
    ("score_risk", gen_score_risk),
    ("payment", gen_payment),
    ("evm", gen_evm),
    ("tender", gen_tender),
    ("procedure_lookup", gen_procedure_lookup),
]


def generate_all() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _, gen in _GENERATORS:
        rows.extend(gen())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/learning/knowledge_scenarios.jsonl")
    parser.add_argument("--append", action="store_true",
                        help="Append to the output file instead of overwriting.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts per generator without writing.")
    args = parser.parse_args()

    counts: Dict[str, int] = {}
    rows: List[Dict[str, str]] = []
    for name, gen in _GENERATORS:
        produced = list(gen())
        counts[name] = len(produced)
        rows.extend(produced)

    print("== generator counts ==", file=sys.stderr)
    for name, n in counts.items():
        print(f"  {name:<24} {n}", file=sys.stderr)
    print(f"  {'TOTAL':<24} {len(rows)}", file=sys.stderr)

    if args.dry_run:
        return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} (mode={mode})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
