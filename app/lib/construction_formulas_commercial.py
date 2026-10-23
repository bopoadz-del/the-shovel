"""Commercial / PM arithmetic calculators (additive library, gap-fill).

Deterministic arithmetic — currency/units are whatever the caller passes
(SAR, USD, m2, sf). Code-agnostic. Rates and quantities are parameters.
"""
from __future__ import annotations

from app.lib.formula_registry import formula

import logging

import os

import re

# Formulas (and the helpers only they use) live in their owner's package;
# imported back so existing imports of this module keep working.
from app.agents.base.formulas.construction_formulas_commercial import (  # noqa: F401 -- moved
    cost_per_area,
    productivity_rate,
    unit_cost_total,
)
from app.agents.hats.commercial.formulas.construction_formulas_commercial import (  # noqa: F401 -- moved
    roi_calculator,
)
from app.agents.hats.contracts.formulas.construction_formulas_commercial import (  # noqa: F401 -- moved
    delay_damages_daily,
)

logger = logging.getLogger(__name__)

# A delay-damages daily-amount ask (an amount in SAR per calendar day, not
# the rate itself). The delay-rate ask already
# surfaces the rate string (0.1% of Contract Price per calendar day) and the
# ACA ask surfaces the ACA.
# The model quoted sources and never multiplied. This path composes
# rate × ACA into a daily figure and must not invent either operand.
# Kill-switch: COMPOSE_DELAY_DAMAGES_DAILY=0 restores the FAIL (rate
# quoted, no SAR/day). Distinct from the delay-rate rescue (#503).
_DD_ASK_RE = re.compile(r"(?i)(?:delay|liquidated)\s+damages")

_DD_CAP_KEY_RE = re.compile(
    r"(?i)\b(?:maximum|max(?:imum)?\s+amount|capped?)\b",
)

_DD_RATE_PCT_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?"
    r"(?:contract\s+price|accepted\s+contract\s+amount)"
    r"[^.]{0,48}\bper\b",
)

_DD_RATE_NEAR_LABEL_RE = re.compile(
    r"(?i)(?:delay|liquidated)\s+damages.{0,240}?"
    r"(\d+(?:\.\d+)?)\s*%[^%]{0,80}\bper\b",
)

_ACA_LABEL_RE = re.compile(
    r"(?i)\b(?:accepted\s+contract\s+amount|contract\s+price)\b",
)

_EXCL_VAT_RE = re.compile(
    r"(?i)\bexclud(?:ing|es|ed)\b.{0,12}\bvat\b|"
    r"\bexcl\.?\s*vat\b|\bexclusive\s+of\s+vat\b",
)

_INCL_VAT_RE = re.compile(
    r"(?i)\binclud(?:ing|es|ed)\b.{0,12}\bvat\b|"
    r"\bincl\.?\s*vat\b|\binclusive\s+of\s+vat\b",
)

_MONEY_RE = re.compile(
    r"(?i)\b(SAR|AED|USD|EUR|GBP|QAR|BHD|KWD|OMR)\s*"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2})",
)

_POINTER_RE = re.compile(
    r"(?i)at\s+the\s+rate\s+stated\s+in\s+the\s+contract\s+data",
)

_CONTRACT_PRICE_RE = re.compile(r"(?i)\bcontract\s+price\b")

_CONTRACT_DATA_CTX_RE = re.compile(
    r"(?i)\b(?:contract\s+data|particulars|8\.8)\b",
)

_WHOLE_WORKS_RE = re.compile(r"(?i)\bwhole\s+of\s+the\s+works\b")

_SUBCLAUSE_87_RE = re.compile(r"(?i)\b(?:sub[- ]?clause\s+)?8\.7\b")

# Live daily-amount ask after #535: CoC 8.7/8.8 windows state 0.015% of the
# excl-VAT ACA (a much smaller SAR/day product). Contract Data 8.8 is 0.1%
# of the Contract Price (the figure the ask wants). First-match compose elected
# 0.015% whenever that window led the excerpts; the lookalike is now
# always rejected.
_LOOKALIKE_RATE_PERCENT = 0.015

_PREFERRED_WHOLE_WORKS_RATE = 0.1

# Live daily-amount ask on c5c6dfa: Contract Data 8.8 chunks 9–11 carried a
# FIDIC worked-example ACA of SAR 10,000,000. Compose elected it
# (0.1% → SAR 10,000/day) instead of the filled excl-VAT row
# (a billion-scale amount). The worked example is always rejected.
_TOY_ACA_AMOUNT = 10_000_000.0

_TOY_DAILY_AMOUNT = 10_000.0

_CLAUSE_111_RE = re.compile(r"(?i)\b1\.1\.1\b")

_TOY_EXAMPLE_CUE_RE = re.compile(
    r"(?i)\b(?:for\s+example|worked\s+example|daily\s+amount|"
    r"if\s+the\s+accepted\s+contract\s+amount)\b",
)

def compose_delay_damages_daily_enabled() -> bool:
    """ON by default. ``COMPOSE_DELAY_DAMAGES_DAILY=0`` is the kill-switch."""
    raw = (os.getenv("COMPOSE_DELAY_DAMAGES_DAILY", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")

def delay_damages_rate_is_coc_lookalike(rate: float, ctx: str = "") -> bool:
    """True for CoC 8.7/8.8 0.015%-of-ACA, not Contract Data 0.1% of Price.

    Live daily-amount ask: chunks 9–11 restated delay damages as 0.015% of
    the filled excl-VAT ACA, a different SAR/day product. The
    Contract Data particular is 0.1% of the Contract Price. A genuine
    Contract Data row that itself says 0.015% of the Contract Price is
    not a lookalike.
    """
    if abs(float(rate) - _LOOKALIKE_RATE_PERCENT) > 1e-9:
        return False
    ctx = ctx or ""
    if _CONTRACT_PRICE_RE.search(ctx) and _CONTRACT_DATA_CTX_RE.search(ctx):
        return False
    if _SUBCLAUSE_87_RE.search(ctx):
        return True
    if _ACA_LABEL_RE.search(ctx) and not _CONTRACT_PRICE_RE.search(ctx):
        return True
    return True

_MILESTONE_OR_SECTION_ASK_RE = re.compile(r"(?i)\b(?:milestone|section)s?\b")

def ask_is_about_a_milestone_or_section(query: str) -> bool:
    """True when the operator asked about one Milestone, or said Section.

    Contract Data 8.8.1 states the 0.015% rate per calendar day per
    Milestone. There is no separate per-Section rate. A question that
    says Section selects this same row, and the answer states it as per
    Milestone. The other daily rate is 0.1% of the Contract Price for
    the whole of the Works. The parser never saw the question, so a
    Milestone ask was scored by whole-of-Works preferences and the
    Milestone row lost (live milestone delay-damages ask).
    """
    return bool(_MILESTONE_OR_SECTION_ASK_RE.search(query or ""))

def delay_damages_rate_preference_score(
    rate: float, ctx: str = "", ask: str = "",
) -> int:
    """Higher wins. Whole-of-Works prefers Contract Data 0.1% over 0.015%.

    A question that says Milestone or Section demotes the whole-of-Works
    row. Both are answered with the 8.8.1 per-Milestone rate.
    """
    ctx = ctx or ""
    if _DD_CAP_KEY_RE.search(ctx):
        return -1
    if ask_is_about_a_milestone_or_section(ask):
        score = 1
        if _CONTRACT_DATA_CTX_RE.search(ctx):
            score += 2
        if abs(float(rate) - _PREFERRED_WHOLE_WORKS_RATE) <= 1e-9:
            score -= 3       # answers a different question
        return score
    if delay_damages_rate_is_coc_lookalike(rate, ctx):
        return 0
    score = 1
    if _CONTRACT_DATA_CTX_RE.search(ctx):
        score += 4
    if _CONTRACT_PRICE_RE.search(ctx):
        score += 3
    if _WHOLE_WORKS_RE.search(ctx):
        score += 2
    if abs(float(rate) - _PREFERRED_WHOLE_WORKS_RATE) <= 1e-9:
        score += 2
    return score

def aca_amount_is_toy_example(amount: float, ctx: str = "") -> bool:
    """True for a FIDIC worked-example / placeholder ACA, not a filled row.

    Live daily-amount ask: unlabeled ``SAR 10,000,000`` in an 8.8 window and its
    0.1% daily product ``SAR 10,000``. A filled 1.1.1 excluding-VAT
    particular of exactly 10M is not a toy. Non-10M/10k figures are
    never the worked example — Contract Data template cues
    (``insert``, ``for example``) must not stain the filled excl-VAT
    ACA (a billion-scale amount, e.g. SAR 1,234,567,890.00).
    """
    amt = float(amount)
    is_10m = abs(amt - _TOY_ACA_AMOUNT) <= 0.005
    is_10k = abs(amt - _TOY_DAILY_AMOUNT) <= 0.005
    if not (is_10m or is_10k):
        return False
    ctx = ctx or ""
    if is_10m and _CLAUSE_111_RE.search(ctx) and _EXCL_VAT_RE.search(ctx):
        # A neighboring filled 1.1.1 row must not un-toy the 8.8
        # worked example when excerpts are concatenated.
        if _TOY_EXAMPLE_CUE_RE.search(ctx):
            return True
        return False
    return True

def chunk_has_real_accepted_contract_amount(text: str) -> bool:
    """True when ``text`` states a non-toy ACA money figure."""
    return any(not toy for _amt, _cur, _kind, toy in _iter_aca_candidates(text))

def chunk_accepted_contract_amount_is_only_toy(text: str) -> bool:
    """True when every ACA money figure in ``text`` is a toy/example."""
    cands = _iter_aca_candidates(text)
    return bool(cands) and all(toy for _amt, _cur, _kind, toy in cands)

def query_asks_delay_damages_daily_amount(query: str) -> bool:
    """True for a daily-amount ask (calculate … delay damages … in SAR), not a rate lookup.

    Delegates to the retriever twin so retrieval rescue and compose
    cannot drift on the ask class.
    """
    try:
        from app.core.rag.retriever import (
            query_asks_delay_damages_daily_amount as _asks,
        )
        return _asks(query)
    except Exception:  # noqa: BLE001 — never break a turn over an import
        q = query or ""
        return bool(q and _DD_ASK_RE.search(q) and re.search(
            r"(?i)\b(?:calculate|compute|work\s+out|how\s+much)\b", q,
        ) and re.search(
            r"(?i)\b(?:sar|aed|usd|eur|gbp|qar|bhd|kwd|omr)\b", q,
        ))

def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

# A candidate's context must be its OWN row. The fixed 96/48-character window
# spans neighbours: in a Contract Data block the Milestone rate sits between
# the whole-of-Works rate and the "Maximum Amount of Delay Damages" cap, so
# the cap guard disqualified the very row a Milestone ask needs.
_ROW_SEP_RE = re.compile(r"[|;]|(?<=\.)\s")

def _row_around(blob: str, start: int, end: int) -> str:
    """The row containing ``blob[start:end]``, clipped at row separators."""
    left = max((m.end() for m in _ROW_SEP_RE.finditer(blob, 0, start)),
               default=max(0, start - 96))
    right_match = _ROW_SEP_RE.search(blob, end)
    right = right_match.start() if right_match else min(len(blob), end + 48)
    return blob[left:right]

def _iter_delay_rate_candidates(text: str, ask: str = "") -> list[tuple[float, int]]:
    """``(rate_percent, preference)`` whole-of-Works daily rates in ``text``."""
    t = text or ""
    if not t:
        return []
    out: list[tuple[float, int]] = []

    def _add(pct: float, ctx: str) -> None:
        score = delay_damages_rate_preference_score(pct, ctx, ask)
        if score < 0:
            return
        out.append((pct, score))

    try:
        from app.core.contract_data_chunks import filled_particulars_rows
        for key, val in filled_particulars_rows(t):
            if not _DD_ASK_RE.search(key):
                continue
            if _DD_CAP_KEY_RE.search(key):
                continue
            if not re.search(r"(?i)\bper\b", val):
                continue
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", val)
            if m:
                _add(float(m.group(1)), f"{key} {val}")
    except Exception:  # noqa: BLE001 — fall through to the scanned regex
        logger.debug("particulars rate parse failed; using scanned regex", exc_info=True)
    blob = _collapse_ws(t)
    if _POINTER_RE.search(blob) and not _DD_RATE_PCT_RE.search(blob):
        return out
    # A Milestone ask reads each candidate's OWN row: in a Contract Data block
    # the Milestone rate sits between the whole-of-Works rate and the cap row,
    # and the wide window let the cap guard disqualify it. The whole-of-Works
    # path keeps the window it was tuned with (daily-amount ask).
    row_scoped = ask_is_about_a_milestone_or_section(ask)
    for m in _DD_RATE_PCT_RE.finditer(blob):
        ctx = (_row_around(blob, m.start(), m.end()) if row_scoped
               else blob[max(0, m.start() - 96):m.end() + 48])
        _add(float(m.group(1)), ctx)
    for m in _DD_RATE_NEAR_LABEL_RE.finditer(blob):
        window = blob[max(0, m.start() - 48):m.end()]
        if _DD_CAP_KEY_RE.search(window):
            continue
        ctx = _row_around(blob, m.start(), m.end())
        _add(float(m.group(1)), ctx)
    return out

def parse_delay_damages_rate_percent(text: str, ask: str = "") -> float | None:
    """Daily Delay Damages *rate* as a percentage, or None.

    A cap row (``Maximum amount of delay damages: 10%…``) and a General
    Conditions pointer are not the rate. When a CoC 8.7/8.8 window
    restates 0.015% of the ACA next to the Contract Data 0.1% of
    Contract Price, the Contract Data particular wins — first-match
    used to emit the 0.015% product. Does not invent a percentage.
    """
    cands = _iter_delay_rate_candidates(text, ask)
    if not cands:
        return None
    preferred = [(pct, score) for pct, score in cands if score >= 2]
    # Score 0 is the CoC 0.015% lookalike. Do not compose a figure
    # from that alone — retrieval must still surface Contract Data 0.1%.
    pool = preferred or [(pct, score) for pct, score in cands if score >= 1]
    if not pool:
        return None
    pool.sort(key=lambda item: -item[1])
    return pool[0][0]

def _iter_aca_candidates(text: str) -> list[tuple[float, str, str, bool]]:
    """``(amount, currency, vat_kind, is_toy)`` ACA figures in ``text``.

    ``vat_kind`` is ``excl``, ``neutral``, or ``incl``. Filled particulars
    rows first, then a 160-char scanned window. Does not invent a figure.
    """
    t = text or ""
    if not t:
        return []
    out: list[tuple[float, str, str, bool]] = []

    def _bucket(amount: float, currency: str, ctx: str) -> None:
        if amount < 1000:
            return
        # A delay-damages *rate* sentence names Contract Price / ACA as
        # the percentage base and is not itself the money row. Do not
        # stain a real Accepted Contract Amount figure just because the
        # 0.1%-per-day row sits in the same 160-char window (live daily-amount
        # ask, scanned Contract Data: rate chunk then excl-VAT ACA).
        if _DD_ASK_RE.search(ctx) and "%" in ctx:
            if not re.search(r"(?i)accepted\s+contract\s+amount", ctx):
                return
        if not _ACA_LABEL_RE.search(ctx):
            return
        if _EXCL_VAT_RE.search(ctx):
            kind = "excl"
        elif _INCL_VAT_RE.search(ctx):
            kind = "incl"
        else:
            kind = "neutral"
        out.append(
            (amount, currency, kind, aca_amount_is_toy_example(amount, ctx)),
        )

    try:
        from app.core.contract_data_chunks import filled_particulars_rows
        for key, val in filled_particulars_rows(t):
            money = _MONEY_RE.search(val)
            if not money:
                continue
            amount = float(money.group(2).replace(",", ""))
            _bucket(amount, money.group(1).upper(), f"{key} {val}")
    except Exception:  # noqa: BLE001 — scanned fallback still runs
        logger.debug("particulars ACA parse failed; using scanned regex", exc_info=True)

    blob = _collapse_ws(t)
    for m in _MONEY_RE.finditer(blob):
        amount = float(m.group(2).replace(",", ""))
        start = max(0, m.start() - 160)
        ctx = blob[start:m.end() + 24]
        _bucket(amount, m.group(1).upper(), ctx)

    return out

def parse_accepted_contract_amount(text: str) -> tuple[float, str] | None:
    """ACA / Contract Price money amount from client text, or None.

    Prefers an excluding-VAT figure when both incl/excl are present
    (FIDIC Contract Price / the rate's base is the net amount). When
    a FIDIC worked-example ACA (live: SAR 10,000,000) sits next to a
    filled excl-VAT row, the toy is dropped — the product must not
    invent a figure and must not elect the example. A percentage-of-
    ACA cap row is skipped.
    """
    t = text or ""
    if not t:
        return None
    buckets: dict[str, list[tuple[float, str]]] = {
        "excl": [], "neutral": [], "incl": [],
    }
    for amount, currency, kind, toy in _iter_aca_candidates(t):
        # The worked example is never elected, even when it is the only
        # candidate: SAR 10,000/day from a toy is a fabricated answer.
        if not toy:
            buckets[kind].append((amount, currency))
    picked = buckets["excl"] or buckets["neutral"] or buckets["incl"]
    if not picked:
        return None
    seen: set[tuple[float, str]] = set()
    ordered: list[tuple[float, str]] = []
    for item in picked:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered[0]

def delay_damages_daily_basis(ask: str) -> str:
    """``milestone`` or ``whole``.

    Contract Data 8.8.1 is per calendar day per Milestone. A question
    that says Section selects that same rate, and the sentence states
    it as per Milestone. The whole of the Works stays its own row.
    """
    if ask_is_about_a_milestone_or_section(ask):
        return "milestone"
    return "whole"

# A Section sum, a change-control "contract value", or a kickoff note is
# not the Contract Price the 0.015% row is a percentage of.
_NOT_THE_CONTRACT_PRICE_RE = re.compile(
    r"(?i)\b(?:contract\s+value|change\s+control|kick-?\s*off|"
    r"section\s+(?:amount|price|value|sum)|package\s+value)\b",
)

def _contract_price_row_score(amount: float, row: str, wide: str) -> int:
    """Higher is the Contract Data Contract Price. 0 means do not use it.

    The rate sentence says "of the Contract Price", so a nearby SAR figure
    matches the ACA label without being clause 1.1.1. That is how another
    amount was multiplied by the per-Milestone rate.
    """
    row = row or ""
    wide = wide or ""
    label_blob = row if re.search(r"(?i)accepted\s+contract\s+amount", row) else wide
    if not re.search(r"(?i)accepted\s+contract\s+amount", label_blob):
        return 0
    if (
        _DD_ASK_RE.search(row)
        and "%" in row
        and not _CLAUSE_111_RE.search(row)
    ):
        return 0
    if _NOT_THE_CONTRACT_PRICE_RE.search(row) and not _CLAUSE_111_RE.search(row):
        return 0
    if aca_amount_is_toy_example(amount, wide):
        return 0
    # The clause token may sit in its own table cell ahead of the "|" that
    # clips ``row`` (live: "1.1.1: | | Accepted Contract Amount: SAR …").
    clause = bool(_CLAUSE_111_RE.search(label_blob) or _CLAUSE_111_RE.search(wide))
    excl = bool(_EXCL_VAT_RE.search(label_blob))
    if _INCL_VAT_RE.search(label_blob) and not excl:
        return 0
    # A VAT-neutral figure is the base only as clause 1.1.1 (the net amount
    # by definition). Any other unlabeled figure still needs "excluding VAT"
    # — that is what keeps a partial or a section sum from being the price.
    if not excl and not clause:
        return 0
    score = 3
    if clause:
        score += 4
    return score

def _best_contract_price_figure(text: str) -> tuple[int, float, str] | None:
    """``(score, amount, currency)`` for the Contract Price in ``text``."""
    blob = _collapse_ws(text)
    best: tuple[int, float, str] | None = None
    for m in _MONEY_RE.finditer(blob):
        amount = float(m.group(2).replace(",", ""))
        if amount < 1000:
            continue
        row = _row_around(blob, m.start(), m.end())
        wide = blob[max(0, m.start() - 96): m.end() + 48]
        score = _contract_price_row_score(amount, row, wide)
        if score <= 0:
            continue
        cand = (score, amount, m.group(1).upper())
        if best is None or cand[0] > best[0] or (
            cand[0] == best[0] and cand[1] > best[1]
        ):
            best = cand
    return best

def _figure_is_clause_111(text: str, amount: float) -> bool:
    """True when ``amount`` is this text's clause 1.1.1 net Accepted Contract Amount.

    Live Contract Data states the clause as a table row —
    ``1.1.1: | | Accepted Contract Amount: SAR 1,234,567,890.00(One Billion …) |``
    — with NO VAT qualifier; the including-VAT figure is a separate
    particular in the same chunk. Demanding the literal "excluding VAT"
    scored the real base 0 and left a partial ACA to compose (live milestone
    delay-damages ask 0/6, twice). Clause 1.1.1 IS the net figure the rate applies to, so a
    row is the base unless it says *including* VAT. The clause token sits
    in its own cell ahead of the ``|``, so it is read from the wider
    window, not the pipe-clipped row.
    """
    blob = _collapse_ws(text)
    for match in _MONEY_RE.finditer(blob):
        got = float(match.group(2).replace(",", ""))
        if abs(got - float(amount)) > 0.02:
            continue
        row = _row_around(blob, match.start(), match.end())
        wide = blob[max(0, match.start() - 96): match.end() + 48]
        label = row if re.search(
            r"(?i)accepted\s+contract\s+amount", row,
        ) else wide
        if (
            (_CLAUSE_111_RE.search(label) or _CLAUSE_111_RE.search(wide))
            and not _INCL_VAT_RE.search(label)
            and re.search(r"(?i)accepted\s+contract\s+amount", label)
        ):
            return True
    return False

def delay_damages_base_is_clause_111(excerpts: str, amount: float) -> bool:
    """True when ``amount`` is clause 1.1.1 in its own retrieved segment.

    Per document, so the next chunk's clause number cannot glue onto a
    partial Accepted Contract Amount and make that partial look filled.
    """
    for _doc, seg in _excerpt_segments(excerpts or ""):
        if _figure_is_clause_111(seg, amount):
            return True
    return False

def text_states_clause_111_aca(text: str) -> bool:
    """True when ``text`` carries a clause 1.1.1 excluding-VAT Accepted
    Contract Amount money row — the real delay-damages base.

    Amount-agnostic twin of ``_figure_is_clause_111``. The rescue that
    joins loaded Contract Data chunks (retriever.py) uses this to keep the
    1.1.1 base ahead of partial ACA rows before it caps the excerpt, so an
    unmarked bundle still composes off the filled clause and not a partial
    (live milestone delay-damages ask: 0.015% x SAR 39,098,392.98).
    """
    blob = _collapse_ws(text)
    for m in _MONEY_RE.finditer(blob):
        try:
            amt = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        if amt < 1000:
            continue
        if _figure_is_clause_111(text, amt):
            return True
    return False

def _contract_price_beside_the_rate(
    text: str, rate: float, ask: str,
) -> tuple[float, str] | None:
    """Contract Price for ``rate``.

    Clause 1.1.1 excluding VAT in the document that states the rate
    wins. The same clause retrieved under another doc id still beats a
    partial Accepted Contract Amount that only shares the rate's chunk
    (live milestone delay-damages ask: SAR 39,098,392.98 on the leading chunk, clause 1.1.1
    later). A non-1.1.1 amount from another document is not the price.
    The rate document's own clause 1.1.1 beats another document's
    clause 1.1.1 (a purchase order of SAR 55,000,000.00 ahead of the
    rate document).
    """
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for doc, seg in _excerpt_segments(text):
        if doc not in grouped:
            order.append(doc)
            grouped[doc] = []
        grouped[doc].append(seg)
    rate_docs: set[str] = set()
    for doc in order:
        blob = "\n".join(grouped[doc])
        got = parse_delay_damages_rate_percent(blob, ask)
        if got is None or abs(float(got) - float(rate)) > 1e-9:
            continue
        rate_docs.add(doc)
    # 100: clause 1.1.1 in the rate's document.
    # 80: clause 1.1.1 retrieved under another doc id.
    # 40: a non-1.1.1 excluding-VAT amount in the rate's document,
    #     used only when no clause 1.1.1 was retrieved.
    best: tuple[int, float, str] | None = None
    for doc in order:
        blob = "\n".join(grouped[doc])
        fig = _best_contract_price_figure(blob)
        if fig is None:
            continue
        row_score, amount, currency = fig
        is_111 = row_score >= 7 and _figure_is_clause_111(blob, amount)
        if is_111 and doc in rate_docs:
            score = 100
        elif is_111:
            score = 80
        elif doc in rate_docs:
            score = 40
        else:
            continue
        if best is None or score > best[0]:
            best = (score, amount, currency)
    if best is None:
        return None
    return best[1], best[2]

def elected_delay_damages_clause(text: str, ask: str) -> str:
    """The Contract Data row that supplied the elected rate, or ""."""
    rate = parse_delay_damages_rate_percent(text, ask)
    if rate is None:
        return ""
    blob = _collapse_ws(text)
    best_row = ""
    best_score = -1
    for m in _DD_RATE_PCT_RE.finditer(blob):
        try:
            pct = float(m.group(1))
        except ValueError:
            logger.debug("delay-damages clause percent was not a float", exc_info=True)
            continue
        if abs(pct - float(rate)) > 1e-9:
            continue
        row = _row_around(blob, m.start(), m.end()).strip()
        score = delay_damages_rate_preference_score(pct, row, ask)
        if score > best_score:
            best_score = score
            best_row = row
    return best_row

def drop_whole_of_works_delay_claims(text: str) -> str:
    """Drop sentences that call delay damages the whole of the Works.

    A decimal point inside ``SAR 1,200.00`` is not the end of the
    sentence. The lead is one claim, and the claim goes.
    """
    raw = text or ""
    if not raw or not _WHOLE_WORKS_RE.search(raw):
        return raw
    parts = re.split(r"(?<=[.!])\s+|\n+", raw)
    kept = [part.strip() for part in parts if part.strip() and not _WHOLE_WORKS_RE.search(part)]
    return " ".join(kept).strip()

def _strip_excerpt_chrome(text: str) -> str:
    """Drop ``[doc_id=…]`` markers and collapse the whitespace they leave."""
    cleaned = re.sub(r"\[[^\]]*\]", " ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip(" ,;.")

def _milestone_quote_without_whole_of_works(quote: str, percent: float) -> str:
    """The elected-rate fragment, without the neighbouring whole-of-Works label.

    A row window reaches into that label when the two particulars are
    not separated. Quoting it would call a Milestone answer the whole
    of the Works, and the quote was then dropped entirely — the live
    line stated the figure and credited no clause.
    """
    raw = _strip_excerpt_chrome(quote)
    if not raw or not _WHOLE_WORKS_RE.search(raw):
        return raw
    token = f"{float(percent):g}"
    pct_re = re.compile(rf"(?<![\d.]){re.escape(token)}\s*%")
    for match in pct_re.finditer(raw):
        cuts = [
            raw.rfind(sep, 0, match.start())
            for sep in (". ", "; ", " and ")
        ]
        hits = [idx for idx in cuts if idx >= 0]
        # " and " is 5 characters; ". " and "; " are 2.
        if hits:
            boundary = max(hits)
            start = boundary + (5 if raw[boundary:boundary + 5].lower() == " and " else 2)
        else:
            start = 0
        end = len(raw)
        for sep in (". ", "; "):
            idx = raw.find(sep, match.end())
            if idx >= 0:
                end = min(end, idx)
        frag = raw[start:end].strip(" ,;.")
        mile = re.search(r"(?i)per\s+milestone\b", frag)
        pct_here = pct_re.search(frag)
        if mile and pct_here and mile.end() > pct_here.start():
            frag = frag[:mile.end()]
        whole = _WHOLE_WORKS_RE.search(frag)
        pct_here = pct_re.search(frag)
        if whole and pct_here and whole.start() > pct_here.start():
            frag = frag[:whole.start()]
        frag = re.sub(r"\s+", " ", frag).strip(" ,;.")
        if frag and not _WHOLE_WORKS_RE.search(frag) and pct_re.search(frag):
            return frag
    return ""

def format_delay_damages_daily_line(composed: dict) -> str:
    """User-facing one-liner for the composed daily figure."""
    cur = composed.get("currency") or "SAR"
    daily = float(composed["daily_amount"])
    pct = float(composed["rate_percent"])
    base = float(composed["contract_amount"])
    basis = (composed.get("basis") or "whole").strip()
    quote = (composed.get("clause_quote") or "").strip()
    # A question that says Section is the 8.8.1 per-Milestone rate.
    # compose_delay_damages_daily_from_excerpts always sets basis from
    # the ask before this runs, so a section or Milestone ask does not
    # arrive here with basis missing. The missing-basis line below is
    # the whole-of-Works sentence.
    source = (composed.get("source_label") or "").strip()
    if basis in ("milestone", "section"):
        line = (
            f"Delay damages per Milestone are "
            f"{cur} {daily:,.2f} per calendar day "
            f"({pct:g}% of the Contract Price {cur} {base:,.2f})."
        )
        safe_quote = _milestone_quote_without_whole_of_works(quote, pct)
        if safe_quote:
            if source and not re.search(r"(?i)\bcontract\s+data\b", safe_quote):
                safe_quote = f"{source} {safe_quote}"
            line = f"{line} The contract states: {safe_quote}"
        elif source:
            line = f"{line} Source: {source}."
        return line
    line = (
        f"Whole-of-Works delay damages: "
        f"{cur} {daily:,.2f} per calendar day "
        f"({pct:g}% of Accepted Contract Amount {cur} {base:,.2f})."
    )
    if source:
        line = f"{line} Source: {source}."
    return line

# Excerpts arrive as "[doc_id=<id> chunk=<n> ...] <text>" blocks. A rate and a
# base figure that sit in DIFFERENT documents are two contracts' numbers, and
# multiplying across them is how #701 produced 0.015% x another project's ACA --
# an amount belonging to another project entirely.
# "AB-2002-202", "AB-CDE-001": a contract/document reference shape.
_CONTRACT_ID_RE = re.compile(r"\b([A-Z]{2,4}-\d{3,4}-\d{2,4})\b")

_DOC_MARKER_RE = re.compile(r"\[doc_id=([^\s\]]+)[^\]]*\]")

def _excerpt_segments(text: str) -> list[tuple[str, str]]:
    """``[(doc_id, segment_text), ...]``; one ("", text) when unmarked."""
    blob = text or ""
    marks = list(_DOC_MARKER_RE.finditer(blob))
    if not marks:
        return [("", blob)]
    out: list[tuple[str, str]] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(blob)
        out.append((m.group(1), blob[m.start():end]))
    return out

def rate_and_base_from_one_document(
    text: str, ask: str = "",
) -> tuple[float, tuple[float, str], str] | None:
    """``(rate_percent, (amount, currency), doc_id)`` from ONE document.

    Documents are tried in the order they were retrieved, so the
    best-ranked document that carries BOTH a rate and a base wins. Returns
    None when no single document carries both -- the honest outcome, because
    a rate from one contract and an amount from another compose a figure that
    belongs to neither.
    """
    segments = _excerpt_segments(text)
    rate = parse_delay_damages_rate_percent(text, ask)
    if rate is None:
        return None
    # A per-document pairing was tried first and removed: parsing one segment
    # at a time loses the preference ordering inside
    # parse_accepted_contract_amount (a filled particular beats a scanned
    # window), and it picked a partial figure over the real Contract Data row
    # on the live A2 fixture. The bundle is parsed as a whole; what makes that
    # safe is the contract check below.
    # Cross-document pairing is allowed only while ONE contract is in view.
    # Two contract identifiers in the bundle means the bases belong to
    # different contracts, and picking one is how #701 answered a Milestone
    # rate against another project's Accepted Contract Amount.
    contracts = set()
    for _doc, seg in segments:
        contracts.update(m.group(1).upper() for m in _CONTRACT_ID_RE.finditer(seg))
    if len(contracts) > 1:
        return None
    # Both asks take the rate document's clause 1.1.1 when the bundle
    # has it. Whole-of-Works used to keep the first excluding-VAT
    # figure, which elected a purchase order's clause 1.1.1 ahead of
    # the rate document. A Milestone ask does not fall back to that
    # first match: another document's non-1.1.1 amount is the #701 leak.
    # Whole-of-Works still falls back when no excluding-VAT price
    # scored, so an unlabeled Accepted Contract Amount keeps composing.
    base = _contract_price_beside_the_rate(text, rate, ask)
    if base is None and not ask_is_about_a_milestone_or_section(ask):
        base = parse_accepted_contract_amount(text)
    if base is None:
        return None
    return rate, base, segments[0][0]

def compose_delay_damages_daily_from_excerpts(
    query: str,
    excerpts: str,
) -> dict | None:
    """Compose rate × ACA from retrieved client text, or None.

    Returns None when the ask is not a daily-amount ask, the kill-switch is off,
    or either operand is missing — never invents a figure.
    """
    if not compose_delay_damages_daily_enabled():
        return None
    # The cap is a percentage of the Accepted Contract Amount, not SAR/day.
    if query_names_delay_damages_cap(query):
        return None
    if not query_asks_delay_damages_daily_amount(query):
        return None
    found = rate_and_base_from_one_document(excerpts, query)
    if found is None:
        return None
    rate, aca, _doc_id = found
    amount, currency = aca
    out = delay_damages_daily(
        rate_percent=rate,
        contract_amount=amount,
        currency=currency,
    )
    if out.get("error"):
        return None
    basis = delay_damages_daily_basis(query)
    out["basis"] = basis
    if re.search(r"(?i)\bcontract\s+data\b", excerpts or ""):
        out["source_label"] = "Contract Data"
    if basis != "whole":
        out["clause_quote"] = elected_delay_damages_clause(excerpts, query)
    return out

def answer_states_daily_amount(text: str, daily_amount: float) -> bool:
    """True when ``text`` already states the composed daily figure."""
    if not text:
        return False
    formatted = f"{daily_amount:,.2f}"
    compact = f"{daily_amount:.2f}"
    blob = text.replace(" ", "")
    return (
        formatted in text
        or compact in text
        or formatted.replace(",", "") in blob
    )

_PCT_PARTICULAR_SPECS = (
    (
        "advance payment",
        "Advance Payment",
        re.compile(r"(?i)advance\s+payment"),
    ),
    (
        "limitation of liability",
        "Limitation of Liability",
        re.compile(r"(?i)limitation\s+of\s+liability|limit(?:ation)?\s+of\s+liabilit"),
    ),
    (
        "retention money",
        "Retention Money",
        re.compile(r"(?i)retention\s+money|percentage\s+of\s+retention"),
    ),
    (
        "maximum amount of delay damages",
        "Maximum Amount of Delay Damages",
        re.compile(
            r"(?i)maximum\s+amount\s+of\s+(?:delay|liquidated)\s+damages"
        ),
    ),
)

_PCT_BOND_RE = re.compile(r"(?i)\bbond\b|\bguarantee\b|\bsecurity\b")

_PCT_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# The next particular ends this label's figure. A later copy of the same
# label ends it too, so an earlier mention cannot reach across Delay
# Damages and take that row's percent.
_COMPETING_PARTICULAR_RES = (
    re.compile(r"(?i)advance\s+payment"),
    re.compile(r"(?i)limitation\s+of\s+liability"),
    re.compile(r"(?i)(?:delay|liquidated)\s+damages"),
    re.compile(r"(?i)percentage\s+of\s+retention|retention\s+money"),
    re.compile(
        r"(?i)maximum\s+amount\s+of\s+(?:delay|liquidated)\s+damages"
    ),
)

_PER_DAY_AFTER_PCT_RE = re.compile(
    r"(?i)per\s+(?:calendar\s+|working\s+)?day\b",
)

_MONEY_ARITHMETIC_ASK_RE = re.compile(
    r"(?i)\b(?:calculate|compute|work\s+out|how\s+much)\b",
)

_MONEY_UNIT_ASK_RE = re.compile(
    r"(?i)\b(?:sar|aed|usd|eur|gbp|qar|bhd|kwd|omr)\b",
)

_EACH_DAYS_LATE_RE = re.compile(
    r"(?i)each\s+(\d+)\s*(?:calendar\s+|working\s+)?days?\s+"
    r"(?:late|of\s+delay|delay(?:ed)?|behind|overdue)",
)

_DAYS_LATE_RE = re.compile(
    r"(?i)(\d+)\s*(?:calendar\s+|working\s+)?days?\s+"
    r"(?:late|of\s+delay|delay(?:ed)?|behind|overdue)",
)

_MILESTONE_LIST_RE = re.compile(
    r"(?i)milestones?\s+(\d+(?:\s*(?:,|and|&)\s*\d+)+)",
)

_MILESTONE_ONE_RE = re.compile(r"(?i)milestone\s+(\d+)")

_MILESTONE_RATE_RE = re.compile(
    r"(?i)milestone\s+(\d+)\s*[|:]\s*"
    r"(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?"
    r"(?:contract\s+price|accepted\s+contract\s+amount)"
    r"[^.]{0,64}\bper\b",
)

_UNBOUND_DD_NOTE_RE = re.compile(
    r"(?i)0\s*%\s+of\s+(?:SAR|AED|USD|EUR|GBP|QAR|BHD|KWD|OMR)"
    r"\s+0(?:\.00)?",
)

def compose_percentage_of_aca_enabled() -> bool:
    raw = (os.getenv("COMPOSE_PERCENTAGE_OF_ACA", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")

def compose_delay_damages_period_enabled() -> bool:
    raw = (os.getenv("COMPOSE_DELAY_DAMAGES_PERIOD", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")

def _asked_percentage_spec(query: str):
    q = query or ""
    for key, display, rx in _PCT_PARTICULAR_SPECS:
        if rx.search(q):
            return key, display, rx
    return None

def query_names_delay_damages_cap(query: str) -> bool:
    """True for the Maximum Amount of Delay Damages cap, not SAR/day.

    The cap is a percentage of the Accepted Contract Amount. A per-day
    rate and a "N days late" sum stay on the delay-damages composers.
    """
    q = query or ""
    if not re.search(
        r"(?i)maximum\s+amount\s+of\s+(?:delay|liquidated)\s+damages", q,
    ):
        return False
    if query_applies_a_delay_duration(q):
        return False
    if re.search(r"(?i)\bper\s+(?:calendar\s+|working\s+)?day\b", q):
        return False
    return True

def query_asks_named_percentage_particular(query: str) -> bool:
    """True for a named percentage of the Accepted Contract Amount.

    Advance Payment, Limitation of Liability, Retention Money, and the
    Maximum Amount of Delay Damages cap. A daily delay-damages rate stays
    on its own composer. Definition questions ("what does Advance Payment
    mean") are not this class.
    """
    q = query or ""
    if not q:
        return False
    if _DD_ASK_RE.search(q) and not query_names_delay_damages_cap(q):
        return False
    if re.search(r"(?i)\bwhat\s+does\b.{0,40}\bmean\b", q):
        return False
    return _asked_percentage_spec(q) is not None

def query_asks_percentage_particular_in_money(query: str) -> bool:
    """True for a %-of-ACA ask like "Calculate the Advance Payment in SAR"."""
    if not query_asks_named_percentage_particular(query):
        return False
    q = query or ""
    return bool(_MONEY_ARITHMETIC_ASK_RE.search(q) and _MONEY_UNIT_ASK_RE.search(q))

def _percent_owned_by_label(
    text: str,
    label_rx: re.Pattern,
    *,
    ask_wants_bond: bool,
    horizon: int | None = None,
) -> tuple[float, str] | None:
    """Percentage that follows this label, not a neighbouring particular.

    A peeled Contract Data row can open on Advance Payment and then hold
    the delay-damages daily rate (0.1% per day) before the filled Advance
    Payment percentage. The first percent in that row is the daily rate.
    Take the percent that follows this label and stop at the next
    particular. A per-day rate is the delay-damages figure, not an
    advance-payment or liability percentage. An Advance Payment Bond row
    is not the Advance Payment itself.
    """
    blob = text or ""
    for match in label_rx.finditer(blob):
        rest = blob[match.end():]
        if horizon is not None:
            rest = rest[:horizon]
        cut = len(rest)
        for other in _COMPETING_PARTICULAR_RES:
            hit = other.search(rest)
            if hit is not None and hit.start() < cut:
                cut = hit.start()
        segment = rest[:cut]
        pct = _PCT_VALUE_RE.search(segment)
        if pct is None:
            continue
        tail = segment[pct.end(): pct.end() + 64]
        if _PER_DAY_AFTER_PCT_RE.search(tail):
            continue
        between = segment[: pct.start()]
        if not ask_wants_bond and _PCT_BOND_RE.search(match.group(0) + between):
            continue
        snippet = blob[match.start(): match.end() + pct.end()].strip()
        return float(pct.group(1)), snippet
    return None

def extract_named_percentage_particular(
    query: str,
    excerpts: str,
) -> dict | None:
    """Filled percentage for the named particular, or None.

    Does not invent a percentage. Skips an Advance Payment *Bond* row
    when the ask is the Advance Payment itself. Does not elect a
    neighbouring delay-damages daily rate that shares the same window.
    """
    spec = _asked_percentage_spec(query)
    if not spec:
        return None
    _key, display, label_rx = spec
    t = excerpts or ""
    if not t:
        return None
    ask_wants_bond = bool(_PCT_BOND_RE.search(query or ""))

    def _from_pair(key: str, val: str) -> dict | None:
        owned = _percent_owned_by_label(
            f"{key} {val}", label_rx, ask_wants_bond=ask_wants_bond,
        )
        if owned is None:
            return None
        percent, _snippet = owned
        return {
            "label": display,
            "percent": percent,
            "value": val.strip(),
            "key": key.strip(),
        }

    try:
        from app.core.contract_data_chunks import filled_particulars_rows
        for key, val in filled_particulars_rows(t):
            parsed = _from_pair(key, val)
            if parsed:
                return parsed
    except Exception:  # noqa: BLE001 — scanned fallback still runs
        logger.debug(
            "particulars percentage parse failed; using scanned regex",
            exc_info=True,
        )
    owned = _percent_owned_by_label(
        _collapse_ws(t), label_rx, ask_wants_bond=ask_wants_bond, horizon=200,
    )
    if owned is None:
        return None
    percent, snippet = owned
    return {
        "label": display,
        "percent": percent,
        "value": snippet,
        "key": display,
    }

def format_named_percentage_line(parsed: dict) -> str:
    pct = float(parsed["percent"])
    label = parsed.get("label") or "particular"
    return (
        f"The {label} is {pct:g}% of the Accepted Contract Amount."
    )

def text_states_percent(text: str, percent: float) -> bool:
    """True when ``text`` states ``percent`` as its own percentage.

    ``10`` must not match inside ``0.10%`` — that token is the 0.1%
    delay-damages daily rate, not the advance-payment percentage.
    """
    token = f"{float(percent):g}"
    return bool(re.search(
        rf"(?i)(?<![\d.]){re.escape(token)}\s*(?:%|percent\b)",
        text or "",
    ))

def strip_conflicting_named_percentage(
    text: str, label: str, percent: float,
) -> str:
    """Drop sentences that give ``label`` a percentage other than ``percent``.

    Leaves a neighbouring particular alone: "Delay Damages are 0.1% per
    day" does not name Advance Payment. A sentence that calls the Advance
    Payment 0.1% is the daily rate wearing the wrong label.
    """
    raw = text or ""
    if not raw or not label:
        return raw
    pat = re.compile(
        rf"(?i)[^.!\n]*\b{re.escape(label)}\b[^.!\n]{{0,160}}?"
        rf"(\d+(?:\.\d+)?)\s*%[^.!\n]*[.!]?"
    )

    def _keep(match: re.Match) -> str:
        got = float(match.group(1))
        if abs(got - float(percent)) <= 1e-9:
            return match.group(0)
        return ""

    cleaned = pat.sub(_keep, raw)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()

def compose_percentage_of_aca_from_excerpts(
    query: str,
    excerpts: str,
) -> dict | None:
    """Compose named-row % × ACA, or None. Never invents an operand."""
    if not compose_percentage_of_aca_enabled():
        return None
    if not query_asks_percentage_particular_in_money(query):
        return None
    parsed = extract_named_percentage_particular(query, excerpts)
    aca = parse_accepted_contract_amount(excerpts)
    if parsed is None or aca is None:
        return None
    amount, currency = aca
    product = round(amount * (float(parsed["percent"]) / 100.0), 2)
    return {
        "amount": product,
        "percent": float(parsed["percent"]),
        "contract_amount": round(amount, 2),
        "currency": (currency or "SAR").upper(),
        "label": parsed.get("label") or "particular",
    }

def format_percentage_of_aca_line(composed: dict) -> str:
    cur = composed.get("currency") or "SAR"
    amt = float(composed["amount"])
    pct = float(composed["percent"])
    base = float(composed["contract_amount"])
    label = composed.get("label") or "particular"
    return (
        f"The {label} is {cur} {amt:,.2f} "
        f"({pct:g}% of Accepted Contract Amount {cur} {base:,.2f})."
    )

def answer_states_money_amount(text: str, amount: float) -> bool:
    """True when ``text`` already states ``amount`` as written money."""
    if not text:
        return False
    formatted = f"{amount:,.2f}"
    compact = f"{amount:.2f}"
    blob = text.replace(" ", "")
    return (
        formatted in text
        or compact in text
        or formatted.replace(",", "") in blob
        or f"{amount:,.1f}" in text
    )

# A percentage of the Accepted Contract Amount is one figure, on the
# excluding-VAT base. "If calculated on the VAT-inclusive figure…" is a
# second amount the Contract Data does not authorise.
_VAT_ALT_RE = re.compile(
    r"(?i)vat[-\s]?inclusive|\bincluding\s+vat\b|\bincl\.?\s*vat\b|"
    r"which\s+base\s+applies|"
    r"letter\s+of\s+award|"
    r"(?:does|do)\s+not\s+state\s+which\s+base"
)

_VAT_ALT_SPLIT_RE = re.compile(r"(?<=[.!;])\s+|\n+|\s+[—–]\s+")

def vat_inclusive_alternative_amounts(
    excerpts: str, percent: float,
) -> list[float]:
    """Including-VAT ACA, and that base times ``percent``.

    The excluding-VAT row is not an alternative. Empty when the excerpts
    do not state an including-VAT Accepted Contract Amount.
    """
    out: list[float] = []
    seen: set[float] = set()
    for amt, _cur, kind, toy in _iter_aca_candidates(excerpts or ""):
        if toy or kind != "incl":
            continue
        base = round(float(amt), 2)
        product = round(base * (float(percent) / 100.0), 2)
        for value in (base, product):
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
    return out

def strip_vat_inclusive_percentage_alternative(
    text: str,
    *,
    keep_amount: float,
    forbidden_amounts: list[float],
) -> str:
    """Drop a VAT-inclusive alternative from a percentage-of-ACA answer.

    Keeps a clause that states ``keep_amount`` and does not also state a
    forbidden figure. A hedge that questions which base applies is not
    kept unless that same clause states the composed amount and no
    forbidden figure.
    """
    raw = text or ""
    if not raw:
        return raw
    parts = [
        p.strip() for p in _VAT_ALT_SPLIT_RE.split(raw) if p and p.strip()
    ]
    kept: list[str] = []
    for part in parts:
        if any(
            answer_states_money_amount(part, amt) for amt in forbidden_amounts
        ):
            continue
        if (
            _VAT_ALT_RE.search(part)
            and not answer_states_money_amount(part, keep_amount)
        ):
            continue
        kept.append(part)
    return " ".join(kept).strip()

def query_applies_a_delay_duration(query: str) -> bool:
    q = query or ""
    return bool(_EACH_DAYS_LATE_RE.search(q) or _DAYS_LATE_RE.search(q))

def query_asks_delay_damages_over_a_period(query: str) -> bool:
    """True for "Milestone 3 and 4 are each 20 days late … delay damages".

    A rate lookup (no duration) and a per-calendar-day amount with no
    delay period stay off this path.
    """
    q = query or ""
    if not q or not _DD_ASK_RE.search(q):
        return False
    return query_applies_a_delay_duration(q)

def parse_asked_milestones(query: str) -> list[int]:
    """Milestone numbers the question names, in order, de-duplicated."""
    q = query or ""
    nums: list[int] = []
    listed = _MILESTONE_LIST_RE.search(q)
    if listed:
        nums = [int(x) for x in re.findall(r"\d+", listed.group(1))]
    if not nums:
        nums = [int(x) for x in _MILESTONE_ONE_RE.findall(q)]
    out: list[int] = []
    seen: set[int] = set()
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

def parse_delay_period_days(query: str) -> int | None:
    m = _EACH_DAYS_LATE_RE.search(query or "") or _DAYS_LATE_RE.search(query or "")
    if not m:
        return None
    days = int(m.group(1))
    return days if days > 0 else None

def _window_is_whole_of_works_rate(blob: str) -> bool:
    """True when the window is the whole-of-Works daily rate, not a Milestone.

    A per-milestone delay-damages ask: a packed particulars row said "per Milestone"
    and then the whole-of-Works delay-damages rate (0.1%). First-
    percent parse elected 0.1% and composed the whole-of-Works product.
    The Contract Data milestone rate is a different, smaller percentage.
    """
    return bool(_WHOLE_WORKS_RE.search(blob or ""))

def parse_milestone_delay_rate_percent(
    excerpts: str,
    milestone: int,
) -> float | None:
    """Per-day % for one Milestone row. Does not invent; ignores the cap.

    Rejects a whole-of-Works 0.1% packed under a "per Milestone" label —
    that product is a different figure (live per-milestone ask vs 0.1% × 40).
    """
    t = excerpts or ""
    if not t or milestone <= 0:
        return None
    for m in _MILESTONE_RATE_RE.finditer(t):
        if int(m.group(1)) != milestone:
            continue
        window = t[max(0, m.start() - 96): m.end() + 48]
        if _window_is_whole_of_works_rate(window):
            continue
        return float(m.group(2))
    try:
        from app.core.contract_data_chunks import filled_particulars_rows
        for key, val in filled_particulars_rows(t):
            if _DD_CAP_KEY_RE.search(key):
                continue
            blob = f"{key} {val}"
            if not re.search(rf"(?i)milestone\s+{milestone}\b", blob):
                continue
            if _window_is_whole_of_works_rate(blob):
                continue
            if not re.search(r"(?i)\bper\b", val):
                continue
            pct = _PCT_VALUE_RE.search(val)
            if pct:
                return float(pct.group(1))
    except Exception:  # noqa: BLE001 — scanned regex already ran
        logger.debug("milestone rate particulars parse failed", exc_info=True)
    return None

def _shared_milestone_delay_rate(excerpts: str) -> float | None:
    """The per-Milestone rate when every listed row carries the same %."""
    rates: list[float] = []
    t = excerpts or ""
    for m in _MILESTONE_RATE_RE.finditer(t):
        window = t[max(0, m.start() - 96): m.end() + 48]
        if _window_is_whole_of_works_rate(window):
            continue
        rates.append(float(m.group(2)))
    if not rates:
        return None
    first = rates[0]
    if all(abs(r - first) < 1e-9 for r in rates):
        return first
    return None

def compose_delay_damages_over_period_from_excerpts(
    query: str,
    excerpts: str,
) -> dict | None:
    """Compose rate × ACA × days [× milestones], or None.

    Uses the Milestone row the question names. Does not fall through to
    the whole-of-Works 0.1% when the ask is a Milestone scenario — that
    product is a different figure.
    """
    if not compose_delay_damages_period_enabled():
        return None
    if not query_asks_delay_damages_over_a_period(query):
        return None
    days = parse_delay_period_days(query)
    if days is None:
        return None
    # A section or Milestone day-count uses the Contract Price that sits
    # with the elected rate. Another document's Accepted Contract Amount
    # is a different figure. When the rate's own document has no price,
    # keep the bundle parse so a rate chunk and a price chunk of one
    # contract still compose.
    if ask_is_about_a_milestone_or_section(query):
        rate_hint = parse_delay_damages_rate_percent(excerpts, query)
        owned = (
            _contract_price_beside_the_rate(excerpts, rate_hint, query)
            if rate_hint is not None else None
        )
        aca = owned if owned is not None else parse_accepted_contract_amount(excerpts)
    else:
        aca = parse_accepted_contract_amount(excerpts)
    if aca is None:
        return None
    amount, currency = aca
    milestones = parse_asked_milestones(query)
    rates: list[float] = []
    if milestones:
        shared = _shared_milestone_delay_rate(excerpts)
        for n in milestones:
            rate = parse_milestone_delay_rate_percent(excerpts, n)
            if rate is None:
                rate = shared
            if rate is None:
                return None
            rates.append(rate)
    else:
        rate = parse_delay_damages_rate_percent(excerpts, query)
        if rate is None:
            return None
        rates = [rate]
    total = round(sum(amount * (r / 100.0) * days for r in rates), 2)
    return {
        "amount": total,
        "days": days,
        "rates": rates,
        "milestones": milestones,
        "contract_amount": round(amount, 2),
        "currency": (currency or "SAR").upper(),
    }

def format_delay_damages_period_line(composed: dict) -> str:
    cur = composed.get("currency") or "SAR"
    amt = float(composed["amount"])
    days = int(composed["days"])
    base = float(composed["contract_amount"])
    rates = composed.get("rates") or []
    milestones = composed.get("milestones") or []
    rate_txt = ", ".join(f"{float(r):g}%" for r in rates) or "the stated rate"
    if milestones:
        ms = " and ".join(f"Milestone {n}" for n in milestones)
        return (
            f"Combined delay damages for {ms} "
            f"({days} days each) are {cur} {amt:,.2f} "
            f"({rate_txt} of Accepted Contract Amount {cur} {base:,.2f} "
            f"× {days} days × {len(milestones)})."
        )
    return (
        f"Delay damages for {days} days are {cur} {amt:,.2f} "
        f"({rate_txt} of Accepted Contract Amount {cur} {base:,.2f} "
        f"× {days} days)."
    )

def answer_is_unbound_delay_damages(text: str) -> bool:
    """True for the empty-args '0% of SAR 0.00' poison."""
    return bool(_UNBOUND_DD_NOTE_RE.search(text or ""))

_PRICED_TAKEOFF_MATERIAL_RE = re.compile(
    r"(?i)\b(concrete|footing|pad\s+footing|raft|slab)\b",
)

_PRICED_TAKEOFF_VERB_RE = re.compile(
    r"(?i)\b(take\s*off|volume|price|priced|cost)\b",
)

_USER_UNIT_RATE_RE = re.compile(
    r"(?i)\b(SAR|USD|AED|QAR|EUR|GBP)\s*([\d,]+(?:\.\d+)?)\s*"
    r"(?:per|/)\s*(?:(?:cubic\s+)?m(?:³|3)|cubic\s+m(?:etre|eter)s?)\b",
)

_FOOTING_COUNT_RE = re.compile(
    r"(?i)(\d+)\s+(?:pad\s+)?footings?\b",
)

_WASTE_PCT_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*%\s*waste")

_CONTINGENCY_PCT_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*%\s*contingenc")

def compose_user_priced_takeoff_enabled() -> bool:
    raw = (os.getenv("COMPOSE_USER_PRICED_TAKEOFF", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")

def query_asks_user_priced_takeoff(query: str) -> bool:
    """True for concrete take-off + an operator-instructed unit rate."""
    q = query or ""
    if not q:
        return False
    if not _PRICED_TAKEOFF_MATERIAL_RE.search(q):
        return False
    if not _PRICED_TAKEOFF_VERB_RE.search(q):
        return False
    return bool(_USER_UNIT_RATE_RE.search(q))

def _parse_lwt_metres(text: str) -> tuple[float, float, float] | None:
    from app.lib.construction_formulas_quantities import parse_lwt_metres

    return parse_lwt_metres(text)

def _parse_footing_count(text: str) -> int:
    match = _FOOTING_COUNT_RE.search(text or "")
    if not match:
        return 1
    n = int(match.group(1))
    return n if n > 0 else 1

def _parse_user_unit_rate(text: str) -> tuple[float, str] | None:
    match = _USER_UNIT_RATE_RE.search(text or "")
    if not match:
        return None
    return float(match.group(2).replace(",", "")), match.group(1).upper()

def _format_qty(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.4f}".rstrip("0").rstrip(".")

def compose_user_priced_takeoff_from_ask(text: str) -> dict | None:
    """Volume × user rate × stated waste/contingency, or None.

    Operands come from the question only. A missing unit rate is not
    filled in. Kill-switch ``COMPOSE_USER_PRICED_TAKEOFF=0`` returns None.
    """
    if not compose_user_priced_takeoff_enabled():
        return None
    if not query_asks_user_priced_takeoff(text):
        return None
    dims = _parse_lwt_metres(text)
    rate = _parse_user_unit_rate(text)
    if dims is None or rate is None:
        return None
    length, width, depth = dims
    unit_rate, currency = rate
    if unit_rate <= 0 or min(length, width, depth) <= 0:
        return None
    count = _parse_footing_count(text)
    net = count * length * width * depth
    waste_m = _WASTE_PCT_RE.search(text or "")
    contingency_m = _CONTINGENCY_PCT_RE.search(text or "")
    waste_pct = float(waste_m.group(1)) if waste_m else 0.0
    contingency_pct = float(contingency_m.group(1)) if contingency_m else 0.0
    if waste_pct < 0 or waste_pct > 100 or contingency_pct < 0 or contingency_pct > 100:
        return None
    with_waste = net * (1.0 + waste_pct / 100.0)
    base_cost = round(with_waste * unit_rate, 2)
    total_cost = round(base_cost * (1.0 + contingency_pct / 100.0), 2)
    return {
        "count": count,
        "length": length,
        "width": width,
        "depth": depth,
        "net_volume_m3": net,
        "volume_with_waste_m3": with_waste,
        "waste_percent": waste_pct,
        "contingency_percent": contingency_pct,
        "unit_rate": unit_rate,
        "currency": currency,
        "base_cost": base_cost,
        "total_cost": total_cost,
    }

def _stated_total_figure_present(text: str, value: float) -> bool:
    compact = (text or "").replace(",", "").replace(" ", "")
    shown = {f"{value:.2f}", f"{value:.3f}".rstrip("0").rstrip(".")}
    trimmed = f"{value:.2f}".rstrip("0").rstrip(".")
    if trimmed:
        shown.add(trimmed)
    if abs(value - round(value)) < 1e-6:
        shown.add(str(int(round(value))))
    return any(token and token in compact for token in shown)

def compose_stated_total_follow_up(current: str, prior: str) -> dict | None:
    """Price a "that total" follow-up from the earlier element's count.

    The current ask has the waste and the rate. The count and the
    per-element size are in ``prior`` ("N elements, each L m by …").
    One cap (7.5 m³) is the miss. No rate in the follow-up → None.
    Kill-switch ``COMPOSE_USER_PRICED_TAKEOFF=0`` returns None.
    The waste is the percentage this ask states. Silence is 0 here —
    the documented 5% default is not applied on top of a stated total.
    """
    if not compose_user_priced_takeoff_enabled():
        return None
    from app.lib.construction_formulas_quantities import (
        element_count_from_text,
        follow_up_refers_to_stated_total,
        unit_dims_metres,
        waste_factor_from_text,
    )
    if not follow_up_refers_to_stated_total(current):
        return None
    rate = _parse_user_unit_rate(current)
    dims = unit_dims_metres(prior)
    count = element_count_from_text(prior)
    if rate is None or dims is None or not count or count <= 1:
        return None
    length, width, depth = dims
    unit_rate, currency = rate
    if unit_rate <= 0 or min(length, width, depth) <= 0:
        return None
    stated = waste_factor_from_text(current)
    waste_pct = 0.0 if stated is None else stated * 100.0
    net = count * length * width * depth
    with_waste = round(net * (1.0 + waste_pct / 100.0), 2)
    total_cost = round(with_waste * unit_rate, 2)
    return {
        "count": count,
        "length": length,
        "width": width,
        "depth": depth,
        "net_volume_m3": net,
        "volume_with_waste_m3": with_waste,
        "waste_percent": waste_pct,
        "unit_rate": unit_rate,
        "currency": currency,
        "total_cost": total_cost,
    }

def answer_states_stated_total(text: str, composed: dict) -> bool:
    """True when ``text`` already states the with-waste volume and the price."""
    return (
        _stated_total_figure_present(text, float(composed["volume_with_waste_m3"]))
        and _stated_total_figure_present(text, float(composed["total_cost"]))
    )

def format_stated_total_follow_up_line(composed: dict) -> str:
    """User-facing line for a follow-up that continues from the stated total."""
    cur = composed.get("currency") or "SAR"
    count = int(composed["count"])
    length = float(composed["length"])
    width = float(composed["width"])
    depth = float(composed["depth"])
    net = float(composed["net_volume_m3"])
    with_waste = float(composed["volume_with_waste_m3"])
    waste_pct = float(composed.get("waste_percent") or 0.0)
    rate = float(composed["unit_rate"])
    total = float(composed["total_cost"])
    net_s = str(int(round(net))) if abs(net - round(net)) < 1e-6 else f"{net:.2f}"
    waste_s = f"{with_waste:.2f}"
    factor = 1.0 + waste_pct / 100.0
    parts = [
        (
            f"Continuing from the stated total "
            f"({count} × {length:g} × {width:g} × {depth:g} = {net_s} m³ net)."
        ),
    ]
    if waste_pct:
        parts.append(
            f"With {waste_pct:g}% waste: {net_s} × {factor:g} = {waste_s} m³."
        )
    else:
        parts.append(f"Volume: {waste_s} m³.")
    parts.append(
        f"At {cur} {rate:g}/m³: {waste_s} × {rate:g} = {cur} {total:,.2f}."
    )
    return " ".join(parts)

def format_user_priced_takeoff_line(composed: dict) -> str:
    """User-facing take-off line from ``compose_user_priced_takeoff_from_ask``."""
    cur = composed.get("currency") or "SAR"
    count = int(composed["count"])
    length = float(composed["length"])
    width = float(composed["width"])
    depth = float(composed["depth"])
    net = float(composed["net_volume_m3"])
    with_waste = float(composed["volume_with_waste_m3"])
    waste_pct = float(composed.get("waste_percent") or 0.0)
    contingency_pct = float(composed.get("contingency_percent") or 0.0)
    rate = float(composed["unit_rate"])
    base = float(composed["base_cost"])
    total = float(composed["total_cost"])
    parts = [
        f"Net volume {count} × {_format_qty(length)} × {_format_qty(width)} "
        f"× {_format_qty(depth)} = {_format_qty(net)} m3.",
    ]
    if waste_pct:
        parts.append(
            f"With {waste_pct:g}% waste: {_format_qty(with_waste)} m3."
        )
    parts.append(
        f"At the instructed {cur} {_format_qty(rate)}/m3 = {cur} {base:,.2f}."
    )
    if contingency_pct:
        parts.append(
            f"Plus {contingency_pct:g}% contingency: {cur} {total:,.2f}."
        )
    return " ".join(parts)

ADDITIONAL_CALCULATORS = {
    "roi_calculator": roi_calculator,
    "unit_cost_total": unit_cost_total,
    "cost_per_area": cost_per_area,
    "productivity_rate": productivity_rate,
    "delay_damages_daily": delay_damages_daily,
}
