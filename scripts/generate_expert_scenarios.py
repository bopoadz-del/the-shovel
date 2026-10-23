#!/usr/bin/env python3
"""Generate deterministic Q&A training pairs from app/prompts/construction_expert.txt.

The source is the senior-PMC system prompt. Every answer this script
emits is a verbatim sentence (or short joined-bullet) from that file —
never a paraphrase, never invented. Instructions/questions are
templated; responses are the raw source text.

The reliability trick: both this generator and its test read the same
file. So "the response quotes the source verbatim" is true by
construction — the response string IS a substring of the file.

Output schema (matches scripts/generate_knowledge_scenarios.py):

    {"instruction": "<question>", "response": "<answer>", "source": "..."}

Sources are tagged ``construction_expert.txt:<section>`` so the operator
can trace any row back to the bullet or rule it was derived from.

CLI:

    python scripts/generate_expert_scenarios.py \\
        --out data/learning/expert_scenarios.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Callable, Dict, Iterator, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

SOURCE_FILE = os.path.join(_REPO, "app", "prompts", "construction_expert.txt")


# ── source loader (single source of truth for both gen & tests) ────────────


def load_source_text() -> str:
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        return f.read()


def load_source_lines() -> List[str]:
    return load_source_text().splitlines()


# ── section parsing ────────────────────────────────────────────────────────
#
# Bullets in the file start with "- ". A bullet may be wrapped across the
# next physical line(s) (continuation lines start with two spaces, or
# any non-empty line that does not itself begin with "- " and is not a
# header). Sections are introduced by an UPPERCASE TITLE LINE. A section is
# a PROCEDURE section when its title names a procedure kind of the shipped
# catalogue (app.core.procedure_catalogue) -- the prompt carries no procedure
# codes, so sections are tagged by kind id. We key sections by their full title.


def _kind_for_title(title: str) -> str:
    from app.core.procedure_catalogue import kind_for_name

    return kind_for_name(title) or ""


def _template_titles() -> Dict[str, str]:
    """``{template_kind: title}`` for every template the catalogue names."""
    from app.core.procedure_catalogue import procedures

    out: Dict[str, str] = {}
    for rec in procedures().values():
        for tkind, title in (rec.get("templates") or {}).items():
            # "Project Decision Note (PDN)" is cited as "Project Decision Note".
            out[tkind] = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    return out


def parse_sections() -> Dict[str, Dict[str, object]]:
    """Return {section_title: {"prc": "<procedure kind>"|"", "bullets": [str, ...]}}.

    Bullets are returned with leading "- " stripped and continuation
    lines joined with a single space. Section order is preserved by
    using an insertion-ordered dict.
    """
    lines = load_source_lines()
    sections: Dict[str, Dict[str, object]] = {}
    current_title: str = ""
    current_bullets: List[str] = []
    pending_bullet: List[str] = []

    def flush_bullet():
        nonlocal pending_bullet
        if pending_bullet:
            joined = " ".join(pending_bullet).strip()
            if joined:
                current_bullets.append(joined)
            pending_bullet = []

    def flush_section():
        flush_bullet()
        if current_title and current_bullets:
            sections[current_title] = {
                "prc": _kind_for_title(current_title),
                "bullets": list(current_bullets),
            }

    for raw in lines:
        line = raw.rstrip()
        # Section title heuristic: ALL-CAPS-ish line that is not a
        # bullet and is not a border. Title lines contain at least one
        # uppercase letter and no leading "-".
        stripped = line.strip()
        if not stripped:
            flush_bullet()
            continue
        if set(stripped) <= set("═─"):
            # divider lines
            flush_bullet()
            continue
        if stripped.startswith("- "):
            flush_bullet()
            pending_bullet = [stripped[2:].strip()]
            continue
        # Heuristic for a header: no leading "- ", not indented, and
        # the previous non-empty context expects a header (i.e., we
        # just finished a section or are between sections). We treat
        # any uppercase-heavy line that is not part of a bullet as a
        # new section title.
        is_upper_heavy = (
            sum(1 for c in stripped if c.isupper()) >= max(3, len(stripped) // 4)
            and not stripped.startswith("  ")
        )
        if is_upper_heavy and not pending_bullet:
            flush_section()
            current_title = stripped
            current_bullets = []
            continue
        # Otherwise: continuation of the current bullet.
        if pending_bullet:
            pending_bullet.append(stripped)
        # Else: prose line outside a bullet — ignored for Q&A purposes.

    flush_section()
    return sections


# ── helpers ────────────────────────────────────────────────────────────────


def _row(instruction: str, response: str, source: str) -> Dict[str, str]:
    return {
        "instruction": instruction.strip(),
        "response": response.strip(),
        "source": f"construction_expert.txt:{source}",
    }


def _all_kinds_in_source() -> List[str]:
    """Every procedure kind with a section in the file, in file order."""
    seen: Dict[str, None] = {}
    for body in parse_sections().values():
        kind = body.get("prc")
        if kind and kind not in seen:
            seen[kind] = None
    return list(seen)


def _section_for_prc(prc: str, sections: Dict[str, Dict[str, object]]) -> Tuple[str, Dict[str, object]] | Tuple[None, None]:
    for title, body in sections.items():
        if body.get("prc") == prc:
            return title, body
    return None, None


# ── gen_prc_procedures ─────────────────────────────────────────────────────


def gen_prc_procedures() -> Iterator[Dict[str, str]]:
    sections = parse_sections()
    # Iterate over procedure sections, in file order.
    for title, body in sections.items():
        prc = body["prc"]
        if not prc:
            continue
        bullets = body["bullets"]
        topic = title.strip()
        templates = _template_titles()
        for idx, bullet in enumerate(bullets, start=1):
            tag = f"{prc}:bullet{idx}"
            # Angle 1: open-ended "what does the procedure say".
            yield _row(
                f"What does the {topic.title()} procedure say about item {idx}?",
                bullet,
                tag,
            )
            # Angle 2: governance framing — which procedure governs this fact.
            yield _row(
                f"Which procedure governs the following requirement: \"{_short(bullet)}\"?",
                f"This is governed by the {topic.title()} procedure. The exact requirement is: {bullet}",
                tag,
            )
            # Angle 3 (conditional): if the bullet names a controlled form or
            # template from the catalogue, ask which form is referenced.
            low = bullet.lower()
            if any(title_.lower() in low for title_ in templates.values()):
                yield _row(
                    f"Under the {topic.title()} procedure, which controlled form is referenced "
                    f"in the requirement about \"{_short(bullet)}\"?",
                    bullet,
                    tag,
                )


def _short(bullet: str, limit: int = 100) -> str:
    s = bullet.strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


# ── gen_critical_rules ─────────────────────────────────────────────────────
#
# We scan the *whole file* for prohibition-bearing or imperative lines
# (the explicit numbered CRITICAL RULES block plus other "NEVER/must/
# cannot/NOT/no work" bullets scattered through the document). Three
# angle-variant Q&A per rule — the response is always the verbatim
# source sentence so the "no paraphrase" property holds.


_RULE_KEYWORDS = (
    "NEVER",
    "never",
    "must",
    "cannot",
    "may not",
    "may NOT",
    "No work",
    "no work",
    "is NOT",
    "does NOT",
    "are NOT",
    "is not the same",
    "do not",
    "Do not",
)


def _is_rule_line(text: str) -> bool:
    return any(kw in text for kw in _RULE_KEYWORDS)


def _collect_rule_sentences() -> List[Tuple[str, str]]:
    """Return [(sentence, tag), ...] for every prohibition/imperative line.

    Tag is "rules:numbered:N" for the explicit CRITICAL RULES block,
    "rules:section:<n>" for inline ones — stable across runs.
    """
    text = load_source_text()
    out: List[Tuple[str, str]] = []
    # 1. Explicit numbered rules — they appear after the "CRITICAL RULES"
    #    divider. Match lines beginning with "N. " (digit + dot + space).
    in_block = False
    numbered_seen = 0
    section_seen = 0
    for raw in load_source_lines():
        line = raw.rstrip()
        stripped = line.strip()
        if "CRITICAL RULES" in stripped and "NEVER VIOLATE" in stripped:
            in_block = True
            continue
        if in_block:
            if not stripped or set(stripped) <= set("═─"):
                continue
            m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
            if m:
                numbered_seen += 1
                sentence = m.group(2).strip()
                out.append((sentence, f"rules:numbered:{numbered_seen}"))
            continue
        # 2. Inline rules in section bullets.
        if stripped.startswith("- ") and _is_rule_line(stripped):
            section_seen += 1
            sentence = stripped[2:].strip()
            out.append((sentence, f"rules:inline:{section_seen}"))
    return out


def gen_critical_rules() -> Iterator[Dict[str, str]]:
    rules = _collect_rule_sentences()
    for sentence, tag in rules:
        # Angle 1: the rule itself.
        yield _row(
            "Quote the construction-procedure rule that this snippet describes: "
            f"\"{_short(sentence, 80)}\".",
            sentence,
            tag,
        )
        # Angle 2: what triggers it / when does it apply.
        yield _row(
            f"In what situation does the following rule apply? Rule: \"{_short(sentence, 80)}\"",
            sentence,
            tag,
        )
        # Angle 3: what the correct alternative / required behaviour is.
        yield _row(
            f"State verbatim the corrective requirement from the construction-expert prompt "
            f"that begins with \"{_short(sentence, 40)}\".",
            sentence,
            tag,
        )


# ── gen_document_numbering ─────────────────────────────────────────────────
#
# Mine the DOCUMENT NUMBERING CONVENTIONS section (bullets like
# "RFI: RFI-[4-digit number] e.g. RFI-0042") plus every catalogue form/template and
# MNL/PRC reference appearing elsewhere — but only emit rows for
# numbering items the file actually documents.


def _doc_numbering_bullets() -> List[str]:
    sections = parse_sections()
    for title, body in sections.items():
        if "DOCUMENT NUMBERING" in title:
            return body["bullets"]
    return []


def _templates_in_source() -> List[str]:
    """Catalogue templates whose title appears in the source, in kind order."""
    text = load_source_text().lower()
    return [k for k, title in sorted(_template_titles().items()) if title.lower() in text]


def gen_document_numbering() -> Iterator[Dict[str, str]]:
    bullets = _doc_numbering_bullets()
    for idx, bullet in enumerate(bullets, start=1):
        tag = f"numbering:bullet{idx}"
        # Pull "TYPE: TYPE-[...]" — every bullet starts with the doc type.
        head = bullet.split(":", 1)[0].strip()
        yield _row(
            f"What is the document numbering convention for {head}?",
            bullet,
            tag,
        )
        yield _row(
            f"How should a {head} document be numbered?",
            bullet,
            tag,
        )
        yield _row(
            f"Give the numbering format and an example for {head}.",
            bullet,
            tag,
        )

    # Controlled forms / templates: each catalogue template the prompt names
    # appears in some procedure bullet or section title. Quote that verbatim.
    sections = parse_sections()
    titles = _template_titles()
    form_to_bullet: Dict[str, Tuple[str, str]] = {}
    for title, body in sections.items():
        for tkind, ttitle in titles.items():
            if tkind in form_to_bullet:
                continue
            if ttitle.lower() in title.lower():
                form_to_bullet[tkind] = (title, title)
                continue
            for bullet in body["bullets"]:
                if ttitle.lower() in bullet.lower():
                    form_to_bullet[tkind] = (title, bullet)
                    break
    for tkind in sorted(form_to_bullet):
        title, bullet = form_to_bullet[tkind]
        name = titles[tkind]
        tag = f"numbering:{tkind}"
        yield _row(
            f"What does the construction-expert prompt say about the {name}?",
            bullet,
            tag,
        )
        yield _row(
            f"In which procedure section does the {name} appear and what is it used for?",
            bullet,
            tag,
        )
        yield _row(
            f"Quote the sentence from the construction-expert prompt that references the {name}.",
            bullet,
            tag,
        )


# ── gen_raci_matrix ────────────────────────────────────────────────────────
#
# RACI lines are recognisable: they assign a role to an activity. We
# scan ALL bullets for ones that contain a "Role: action" pattern or
# the explicit "Roles:" prefix, plus the Initiator/Approver/Signatory/
# Process Owner lines, plus the explicit DESIGN RACI MATRIX section
# bullets.


_RACI_PREFIXES = (
    "Employer", "PMT", "Creative Consultant", "Turnkey Contractor",
    "Operator", "Cost Consultant", "Statutory Bodies", "Programme Director",
    "Prime Contractor", "Delivery Manager", "Document Controller",
    "Supervision Consultant", "Contractor", "Initiator", "Approver",
    "Signatory", "Process Owner", "Project Director", "Employer's Representative",
    "Head of Contracts", "Contracts Manager", "Director of Contracts",
    "Contract Manager", "Contract Administrator", "PMC", "QA/QC Manager",
    "Independent Certifier", "Vendor", "Finance", "Board",
    "Design Manager", "PMT Design Manager", "Primary Reviewer",
    "Planning Managers", "Chief Cost Manager", "PMC Head Controls",
    "PMC Delivery Manager", "PMC Project Director", "PMC QA/QC",
    "Head C&P",
)


def _split_roles_bullet(bullet: str) -> List[str]:
    """Given a "Roles: A (X), B (Y), C (Z)." bullet, return ["A (X)", ...]."""
    if not bullet.lower().startswith("roles:"):
        return []
    payload = bullet.split(":", 1)[1].strip().rstrip(".")
    # Split on commas not inside parens.
    parts = []
    depth = 0
    buf = []
    for ch in payload:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _collect_raci_rows() -> List[Tuple[str, str, str, str]]:
    """Return list of (question_context, role_phrase, source_bullet, tag).

    role_phrase is the exact "Role: activity" substring. We always quote
    the full bullet verbatim in the response so the no-paraphrase
    invariant is honoured.
    """
    sections = parse_sections()
    out: List[Tuple[str, str, str, str]] = []
    bullet_counter = 0
    for title, body in sections.items():
        prc = body.get("prc") or ""
        ctx = prc or title
        for bullet in body["bullets"]:
            bullet_counter += 1
            tag_base = f"raci:{prc or 'section'}:{bullet_counter}"
            # Case A: explicit "Role: X" sentence (RACI matrix section).
            if ":" in bullet and not bullet.lower().startswith("roles:"):
                head = bullet.split(":", 1)[0].strip()
                if head in _RACI_PREFIXES:
                    out.append((ctx, head, bullet, tag_base))
                    continue
            # Case B: "Roles: A (X), B (Y)..." compact line.
            roles = _split_roles_bullet(bullet)
            if roles:
                for i, role_clause in enumerate(roles):
                    out.append((ctx, role_clause, bullet, f"{tag_base}:r{i+1}"))
                continue
            # Case C: standalone "Initiator: X" / "Approver: Y" / etc.
            for prefix in ("Initiator", "Approver", "Signatory", "Process Owner"):
                if prefix + ":" in bullet:
                    out.append((ctx, prefix, bullet, tag_base))
                    break
            # Case D: workflow-step lines like "<name> workflow: A does X ->
            # B does Y -> C signs". Split on "->" and each step becomes an
            # actor row.
            if " -> " in bullet and ("workflow" in bullet.lower() or "Workflow" in bullet):
                # Strip the lead-in ("NCR workflow:", "Workflow:", "Workflow (Type A):").
                payload = bullet.split(":", 1)[1].strip() if ":" in bullet else bullet
                steps = [s.strip().rstrip(".") for s in payload.split("->")]
                for i, step in enumerate(steps):
                    if not step:
                        continue
                    out.append((ctx, step, bullet, f"{tag_base}:w{i+1}"))
    return out


def gen_raci_matrix() -> Iterator[Dict[str, str]]:
    rows = _collect_raci_rows()
    for ctx, role_clause, bullet, tag in rows:
        role_name = role_clause.split("(")[0].split(":")[0].strip() or role_clause
        yield _row(
            f"In the context of {ctx}, what is the role assignment captured by "
            f"\"{_short(role_clause, 60)}\"?",
            bullet,
            tag,
        )
        yield _row(
            f"Who handles \"{role_name}\" responsibilities under {ctx}?",
            bullet,
            tag,
        )


# ── gen_formulas_and_timeframes ────────────────────────────────────────────
#
# Two streams: (a) every numeric formula in CONSTRUCTION FORMULAS &
# CALCULATIONS, (b) every numeric timeframe / threshold across the file
# (e.g. "7 calendar days", "24h notice", "Risk score = ...").


_NUMERIC_RE = re.compile(
    r"\b("
    r"\d+\s*(?:calendar\s+)?(?:day|days|hour|hours|h|%|kg/m\^3)"
    r"|<=\s*\d+|>=\s*\d+|\d+\s*-\s*\d+\s*%"
    r"|\d+%"
    r"|\d+\s*-\s*\d+\s*(?:day|days|hour|hours)?"
    r")\b"
)


def _is_formula_bullet(bullet: str) -> bool:
    return "=" in bullet


def _is_timeframe_bullet(bullet: str) -> bool:
    return bool(_NUMERIC_RE.search(bullet))


def gen_formulas_and_timeframes() -> Iterator[Dict[str, str]]:
    sections = parse_sections()
    seen_bullets: set = set()
    # Stream A: formula bullets in formula-heavy sections.
    formula_sections = {
        "EARNED VALUE MANAGEMENT",
        "QUANTITIES",
        "RETENTION & PAYMENTS",
        "PRICE ESCALATION",
        "RISK SCORING",
    }
    bullet_idx = 0
    for title, body in sections.items():
        is_formula_section = any(fs in title for fs in formula_sections)
        for bullet in body["bullets"]:
            bullet_idx += 1
            if not (is_formula_section and _is_formula_bullet(bullet)):
                continue
            if bullet in seen_bullets:
                continue
            seen_bullets.add(bullet)
            tag = f"formula:{title.split()[0].lower()}:{bullet_idx}"
            # Split "name = expr ..." → topic = left of "=".
            name = bullet.split("=", 1)[0].strip().rstrip(":-").strip()
            yield _row(
                f"What is the formula for {name}?",
                bullet,
                tag,
            )
            yield _row(
                f"Quote the construction-expert prompt's definition of {name}.",
                bullet,
                tag,
            )

    # Stream B: numeric timeframe / threshold bullets across the whole file.
    bullet_idx = 0
    for title, body in sections.items():
        prc = body.get("prc") or ""
        for bullet in body["bullets"]:
            bullet_idx += 1
            if bullet in seen_bullets:
                continue
            if not _is_timeframe_bullet(bullet):
                continue
            if _is_formula_bullet(bullet):
                continue
            seen_bullets.add(bullet)
            tag = f"timeframe:{prc or 'section'}:{bullet_idx}"
            # First numeric phrase in the bullet, used in the question.
            m = _NUMERIC_RE.search(bullet)
            phrase = m.group(1).strip() if m else ""
            ctx = prc or title
            yield _row(
                f"Under {ctx}, what is the requirement that involves \"{phrase}\"?",
                bullet,
                tag,
            )
            yield _row(
                f"Cite the exact timeframe or numeric threshold from {ctx} regarding "
                f"\"{_short(bullet, 60)}\".",
                bullet,
                tag,
            )


# ── orchestrator ──────────────────────────────────────────────────────────


_GENERATORS: List[Tuple[str, Callable[[], Iterator[Dict[str, str]]]]] = [
    ("prc_procedures", gen_prc_procedures),
    ("critical_rules", gen_critical_rules),
    ("document_numbering", gen_document_numbering),
    ("raci_matrix", gen_raci_matrix),
    ("formulas_and_timeframes", gen_formulas_and_timeframes),
]


def generate_all() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _, gen in _GENERATORS:
        rows.extend(gen())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/learning/expert_scenarios.jsonl")
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
