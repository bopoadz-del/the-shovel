"""Rows from a model's read of bill pages: the LLM says which figure is which, code checks.

The derived bill sheet (``bill_derived``) is written by
``bill_table_geometry.write_bill_workbook`` from ``BillRow`` objects. Where
the page's word geometry cannot place a figure, a model reads the page and
returns, per page, its lines as JSON::

    {"type": "item", "ref": "A", "description": "...", "unit": "m3",
     "quantity": 1277, "rate": 148, "amount": 188996,
     "figures": {"<a money column the page prints>": "<its value>"}}

This module turns those lines into ``BillRow`` objects for the same writer.
It decides nothing about the figures; it only states facts about them:

* a ``heading`` line is not a row; a run of consecutive heading lines is the
  heading path of the rows that follow it, under the label of the bill the
  read was for (the caller passes it, taken from the document's own summary);
* every other line is a row and keeps the model's line type in the
  ``Line Type`` column, so a collection or a carried line stays in the sheet
  and the sheet's own arithmetic (``bill_aggregate``) can confirm it;
* any further money column the page prints (``figures``) is kept as its own
  column under the label the page gives it; nothing is forced into rate or
  amount;
* each row carries its check: ``""`` when accepted, otherwise what failed
  (a figure that is not a number, or quantity x rate != amount). A figure
  that fails is kept as read and flagged; it is never repaired;
* a page whose read is missing or is not a list of lines is NOT READ, never
  "a page with no rows";
* a key of the read that is not a page number (decimal digits only) is not dropped
  in silence: it is listed in ``keys_not_pages`` so the caller can say so.

No word of any line is read: types come from the model, checks from cells.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core import bill_table_geometry as g

HEADING = "heading"
LINE_TYPE_LABEL = "Line Type"
REF_LABEL = "Item"
CHECK_NOT_A_NUMBER = "figure not a number"

#: The model's figure keys, mapped to the sheet's canonical fields.
_FIGURES = (("quantity", "quantity"), ("rate", "rate"), ("amount", "total"))


@dataclass
class PageReadRows:
    """Rows of a model read, the pages that gave none, keys that are not pages, flagged rows."""

    rows: list[g.BillRow] = field(default_factory=list)
    pages_not_read: list[int] = field(default_factory=list)
    keys_not_pages: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[g.BillRow]:
        return [r for r in self.rows if r.check]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _figure(value: Any) -> tuple[str, bool]:
    """The cell text for one figure and whether it reads as a number.

    A string is kept exactly as the model gave it (its printed places carry the
    bill's own rounding). A number is written without a spurious ``.0``.
    """
    if value is None or value == "":
        return "", True
    if isinstance(value, bool):
        return str(value), False
    if isinstance(value, int):
        return str(value), True
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value), False
        return (str(int(value)) if value.is_integer() else repr(value)), True
    text = _text(value)
    return text, g.parse_number(text) is not None


def rows_from_page_reads(
    pages: Mapping[Any, Any], bill_label: str = "",
) -> PageReadRows:
    """``BillRow`` objects from a model's per-page line reads, pages in order."""
    out = PageReadRows()
    keyed: list[tuple[int, Any]] = []
    for key, lines in pages.items():
        name = _text(key)
        if name.isdecimal():
            keyed.append((int(name), lines))
        else:
            out.keys_not_pages.append(name)
    keyed.sort(key=lambda kv: kv[0])
    top = (bill_label.strip(),) if bill_label.strip() else ()
    headings: list[str] = []
    after_row = True
    for page, lines in keyed:
        if (not isinstance(lines, list) or not lines
                or not all(isinstance(line, Mapping) for line in lines)):
            out.pages_not_read.append(page)
            continue
        for line in lines:
            kind = _text(line.get("type")).lower()
            if kind == HEADING:
                if after_row:
                    headings = []
                    after_row = False
                label = _text(line.get("description"))
                if label:
                    headings.append(label)
                continue
            after_row = True
            out.rows.append(_row(line, page, top + tuple(headings), kind))
    return out


def _row(line: Mapping[str, Any], page: int, path: tuple[str, ...], kind: str) -> g.BillRow:
    fields = {"description": _text(line.get("description")),
              "unit": _text(line.get("unit"))}
    bad = False
    for key, name in _FIGURES:
        text, ok = _figure(line.get(key))
        fields[name] = text
        bad = bad or not ok
    extra: dict[str, str] = {}
    ref = _text(line.get("ref"))
    if ref:
        extra[REF_LABEL] = ref
    extra[LINE_TYPE_LABEL] = kind
    printed = line.get("figures")
    if isinstance(printed, Mapping):
        for label, value in printed.items():
            name = _text(label)
            if not name:
                continue
            text, ok = _figure(value)
            extra[name] = text
            bad = bad or not ok
    check = CHECK_NOT_A_NUMBER if bad else g.check_row(fields)
    return g.BillRow(
        section=path[-1] if path else "",
        source_page=page,
        fields=fields,
        extra=extra,
        check=check,
        path=path,
    )


def rows_of_model_reads(
    read_page: Callable[[Any], Any], bill_label: str = "",
) -> Callable[[Sequence[Any]], tuple[list[g.BillRow], list[int]]]:
    """A ``bill_derived.RowsOfReads`` from a per-page model reader.

    ``read_page`` gets one ``bill_classify.PageRead`` (0-based ``index``) and
    returns the model's lines for that page (or anything else, which is NOT
    READ). The rows and the 1-based pages not read are what
    :func:`rows_from_page_reads` gives for those pages; the third value is
    every reply exactly as the model gave it, by 1-based page, so the copy
    keeps the raw output beside the rows it accepted."""

    def rows_of(
        page_reads: Sequence[Any],
    ) -> tuple[list[g.BillRow], list[int], dict[str, Any]]:
        pages = {str(r.index + 1): read_page(r) for r in page_reads}
        got = rows_from_page_reads(pages, bill_label)
        return got.rows, got.pages_not_read, pages

    return rows_of


#: The reply allowance for one page. The model spends hidden reasoning tokens
#: out of the same budget: a short allowance comes back empty, not short.
PAGE_READ_MAX_TOKENS = 8000
#: Calls per page: the first, and one retry when the reply is empty or not JSON.
PAGE_READ_ATTEMPTS = 2
#: The resolution a page is rendered at for the model (figures legible).
PAGE_READ_DPI = 200

#: The default instruction sent with each page image. It names the JSON shape
#: only (line types, figures as printed, null for what is not printed); it says
#: nothing about any one document's headings, columns or wording.
PAGE_READ_PROMPT = (
    "This image is one page of a priced table of quantities. Read every line of "
    "the table and return ONLY a JSON object: {\"lines\": [{\"type\": one of "
    "\"item\", \"carried\", \"brought_forward\", \"subtotal\", \"total\", "
    "\"heading\", \"summary\", \"bill\": string or null, \"ref\": string or "
    "null, \"description\": string, \"unit\": "
    "string or null, \"quantity\": number or null, \"rate\": number or null, "
    "\"amount\": number or null, \"figures\": object or null}]}. An item is a "
    "work line with its own quantity. A carried, brought_forward, subtotal or "
    "total line is a sum of other lines. bill is the number or name of the bill "
    "the line belongs to, as the page prints it, or null when the page does not "
    "say. A summary line is a line of a summary page that lists the total of a "
    "whole bill: give that bill in bill and its total in amount. When the page prints money columns "
    "other than rate and amount, give each one in figures under the label the "
    "page prints. Copy figures exactly as printed, without thousands "
    "separators. Use null for anything not printed; never calculate or guess "
    "a figure."
)


def page_read_messages(prompt: str, png: bytes) -> list[dict[str, Any]]:
    """One user message carrying the instruction and the page image as content blocks."""
    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]


def lines_of_reply(text: Any) -> list[Any] | None:
    """The page's lines from one reply, or ``None`` when the reply gives none.

    The reply is a JSON object with a ``lines`` list, optionally inside a code
    fence. An empty reply, a reply that is not JSON, a JSON value without a
    ``lines`` list, or an empty list all give ``None``: the page was not read,
    which is never the same as "a page with no rows"."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t[:4].lower() == "json":
            t = t[4:]
    try:
        data: Any = json.loads(t)
    except ValueError as exc:
        # not JSON: no lines; the caller keeps the reply text as why
        data = {"not_json": type(exc).__name__}
    lines = data.get("lines") if isinstance(data, dict) else None
    if not isinstance(lines, list) or not lines:
        return None
    return lines


#: The raw-output entry of a page the model gave no lines for, after every
#: attempt: ``{NOT_READ_KEY: True, "attempts": n, "replies": [...]}``, each
#: reply cut short or the class of the error the call raised. Not a list, so
#: :func:`rows_from_page_reads` counts the page as NOT READ.
NOT_READ_KEY = "not_read"
_REPLY_KEPT = 200
#: What a provider call raises on a failed attempt: an HTTP or network error,
#: an unset key or a non-2xx (``RuntimeError``), a timeout (``OSError``), or a
#: reply without the expected shape. Anything else is a bug and propagates.
_CALL_ERRORS = (httpx.HTTPError, RuntimeError, OSError, LookupError, TypeError, ValueError)


def model_page_reader(
    ask: Callable[[list[dict[str, Any]]], Any],
    render_png: Callable[[int], bytes],
    *,
    prompt: str = PAGE_READ_PROMPT,
    attempts: int = PAGE_READ_ATTEMPTS,
) -> Callable[[Any], Any]:
    """A ``read_page`` for :func:`rows_of_model_reads`.

    For one ``bill_classify.PageRead`` it renders the page (``render_png`` gets
    the 0-based index), sends it with ``prompt`` through ``ask`` (messages ->
    reply text) and returns the reply's lines. An empty or non-JSON reply, or a
    call that raises, is tried again up to ``attempts`` calls in all; then the
    page is a not-read entry (:data:`NOT_READ_KEY`), which :func:`rows_from_page_reads` counts as
    NOT READ. A figure is never filled in here."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    def read_page(page_read: Any) -> Any:
        messages = page_read_messages(prompt, render_png(page_read.index))
        replies: list[str] = []
        for _ in range(attempts):
            try:
                text = ask(messages)
            except _CALL_ERRORS as exc:  # an attempt that failed, never a page of no rows
                replies.append(type(exc).__name__)
                continue
            lines = lines_of_reply(text)
            if lines is not None:
                return lines
            replies.append(str(text)[:_REPLY_KEPT])
        return {NOT_READ_KEY: True, "attempts": attempts, "replies": replies}

    return read_page


def render_page_png(file_path: str, index: int, dpi: int = PAGE_READ_DPI) -> bytes:
    """One page of a stored PDF (0-based ``index``) as PNG bytes at ``dpi``."""
    import fitz  # PyMuPDF

    from app.core import file_crypto

    with file_crypto.open_plaintext(file_path) as readable:
        doc = fitz.open(readable)
        try:
            pix = doc[index].get_pixmap(dpi=dpi)
            try:
                return pix.tobytes("png")
            finally:
                del pix
        finally:
            doc.close()


def provider_ask(max_tokens: int = PAGE_READ_MAX_TOKENS) -> Callable[[list[dict[str, Any]]], str]:
    """``ask`` through the product's provider client (``llm_client.complete``):
    the configured provider and model, its usage recording and fallback."""
    import asyncio

    from app.core import llm_client

    def ask(messages: list[dict[str, Any]]) -> str:
        return asyncio.run(llm_client.complete(
            messages, temperature=0.0, max_tokens=max_tokens,
            agent_name="bill_page_read"))

    return ask


ITEM = "item"
STATE_RECONCILED = "RECONCILED"
STATE_FAILED = "FAILED"
STATE_UNCHECKED = "UNCHECKED"
STATE_NOT_READ = "NOT READ"
STATE_TOTAL_ONLY = "TOTAL ONLY"


@dataclass(frozen=True)
class BillSpan:
    """One bill of the document as its own summary states it: label, 1-based
    inclusive page range, and the printed bill total (``None`` = not printed)."""

    label: str
    first_page: int
    last_page: int
    printed_total: float | None = None


_NOT_MONEY = frozenset({REF_LABEL, LINE_TYPE_LABEL})


def _carries_money(row: g.BillRow) -> bool:
    """True when the row prints a figure in a money column: rate, amount, or a
    money column the page labels itself. Quantity is not money."""
    if _text(row.fields.get("rate")) or _text(row.fields.get("total")):
        return True
    return any(_text(v) for k, v in row.extra.items() if k not in _NOT_MONEY)


def bill_states(
    rows: Sequence[g.BillRow],
    pages_not_read: Sequence[int],
    *,
    spans: Sequence[BillSpan],
    tolerance: float,
) -> dict[str, Any]:
    """Per bill, the check of its item rows against its printed total.

    A row belongs to the bill whose page range holds its source page. Only
    rows the model typed ``item`` are summed. The state is ``NOT READ`` when a
    page of the bill was not read, ``UNCHECKED`` when the bill prints no total,
    ``TOTAL ONLY`` when the bill prints a total and has item rows but not one
    of them carries a money figure in any column (rate, amount, or a money
    column of its own): the document prints no item prices for that bill.
    ``FAILED`` when the item sum differs from the printed total by more than
    ``tolerance`` (the caller states it from the bill's own rounding) or any
    row of the bill is flagged, else ``RECONCILED``. Rows under no bill are
    counted, never moved into one. Figures are read, never repaired."""
    out: list[dict[str, Any]] = []
    placed: set[int] = set()
    missing = set(pages_not_read)
    for span in spans:
        members = [i for i, r in enumerate(rows)
                   if span.first_page <= r.source_page <= span.last_page]
        placed.update(members)
        items_sum = 0.0
        summed = without = flagged = items = priced = 0
        for i in members:
            r = rows[i]
            flagged += 1 if r.check else 0
            if r.extra.get(LINE_TYPE_LABEL) != ITEM:
                continue
            items += 1
            priced += 1 if _carries_money(r) else 0
            amount = g.parse_number(r.fields.get("total", ""))
            if amount is None:
                without += 1
                continue
            summed += 1
            items_sum += amount
        not_read = sorted(p for p in missing if span.first_page <= p <= span.last_page)
        printed = span.printed_total
        diff = None if printed is None else items_sum - printed
        if not_read:
            state = STATE_NOT_READ
        elif diff is None:
            state = STATE_UNCHECKED
        elif items and not priced and not flagged:
            state = STATE_TOTAL_ONLY
        elif abs(diff) > tolerance or flagged:
            state = STATE_FAILED
        else:
            state = STATE_RECONCILED
        out.append({"label": span.label, "pages": [span.first_page, span.last_page],
                    "rows": len(members), "rows_summed": summed,
                    "rows_without_amount": without, "rows_flagged": flagged,
                    "item_rows_priced": priced,
                    "items_sum": items_sum, "printed_total": printed, "diff": diff,
                    "pages_not_read": not_read, "state": state})
    return {"bills": out, "rows_unassigned": len(rows) - len(placed)}


SUMMARY = "summary"
BILL_KEY = "bill"


def _bill_key(value: Any) -> str:
    """A bill's identifier compared without case, spacing or punctuation."""
    return " ".join(re.findall(r"[^\W_]+", _text(value).casefold()))


@dataclass
class SpanRead:
    """Bill spans from the page reads, and every page or bill they could not place."""

    spans: list[BillSpan] = field(default_factory=list)
    #: Pages whose lines name more than one bill (not a member of any span).
    pages_split: list[int] = field(default_factory=list)
    #: Bills whose page range holds a page of another bill (no span is given).
    bills_interleaved: list[str] = field(default_factory=list)
    #: Bills the summary lists with two different totals (span total ``None``).
    totals_conflicting: list[str] = field(default_factory=list)
    #: Bills the summary lists that no read page belongs to.
    summary_without_pages: list[str] = field(default_factory=list)


def spans_from_page_reads(pages: Mapping[Any, Any]) -> SpanRead:
    """Each bill's page range and printed total, from the model's own page reads.

    The model names, per line, the bill it belongs to (``bill``) and types the
    lines of a summary page that list a whole bill's total ``summary``. Code
    only groups: a page belongs to the one bill its non-summary lines name; a
    bill's span runs from its first to its last page, so a page not read or
    naming no bill inside that range is inside the bill (and ``bill_states``
    says NOT READ for it); the printed total is the amount the summary lists
    for the bill, ``None`` when it lists none. Nothing is guessed: a page
    naming two bills, a bill whose range holds another bill's page, and a bill
    the summary lists twice with different totals are reported, not resolved.
    Bills are compared without case, spacing or punctuation; the label kept is
    the first one the reads print."""
    out = SpanRead()
    label: dict[str, str] = {}
    members: dict[str, list[int]] = {}
    totals: dict[str, set[float]] = {}
    for key, lines in pages.items():
        name = _text(key)
        if not name.isdecimal() or not isinstance(lines, list):
            continue
        page = int(name)
        named: set[str] = set()
        for line in lines:
            if not isinstance(line, Mapping):
                continue
            bill = _bill_key(line.get(BILL_KEY))
            if not bill:
                continue
            label.setdefault(bill, _text(line.get(BILL_KEY)))
            if _text(line.get("type")).lower() == SUMMARY:
                text, ok = _figure(line.get("amount"))
                amount = g.parse_number(text) if ok and text else None
                if amount is not None:
                    totals.setdefault(bill, set()).add(amount)
                continue
            named.add(bill)
        if len(named) > 1:
            out.pages_split.append(page)
        elif named:
            members.setdefault(next(iter(named)), []).append(page)
    out.pages_split.sort()
    ranges = {b: (min(ps), max(ps)) for b, ps in members.items()}
    for bill, (first, last) in sorted(ranges.items(), key=lambda kv: kv[1]):
        inside = any(first <= p <= last for other, ps in members.items()
                     if other != bill for p in ps)
        if inside or any(first <= p <= last for p in out.pages_split):
            out.bills_interleaved.append(label[bill])
            continue
        found = totals.get(bill, set())
        if len(found) > 1:
            out.totals_conflicting.append(label[bill])
        printed = next(iter(found)) if len(found) == 1 else None
        out.spans.append(BillSpan(label[bill], first, last, printed))
    out.summary_without_pages = sorted(label[b] for b in totals if b not in members)
    return out
