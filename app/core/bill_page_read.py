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
  (a figure that is not a number, quantity x rate != amount, or a sum line
  that is not the sum of the lines it closes - its section, page or bill,
  :func:`check_sum_lines`). A figure that fails is kept as read and flagged;
  it is never repaired;
* a page whose read is missing or is not a list of lines is NOT READ, never
  "a page with no rows";
* a key of the read that is not a page number (decimal digits only) is not dropped
  in silence: it is listed in ``keys_not_pages`` so the caller can say so.

No word of any line is read: types come from the model, checks from cells.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g

HEADING = "heading"
LINE_TYPE_LABEL = "Line Type"
REF_LABEL = "Item"
#: The bill the model says a line belongs to, kept on the row so a row is
#: placed by its own line's bill, not only by the page it sits on.
BILL_LABEL = "Bill"
CHECK_NOT_A_NUMBER = "figure not a number"
CHECK_SUM_LINE = "sum line is not the sum of the lines it closes"
#: Line types the read prompt defines as a sum of other lines of the page.
#: ``brought_forward`` restates an earlier page: it is added, never checked.
SUM_LINE_TYPES = frozenset({"carried", "subtotal", "total"})
BROUGHT_FORWARD = "brought_forward"

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


def read_lines(entry: Any) -> list[Mapping[str, Any]] | None:
    """A page's lines when ``entry`` is a read of the page, else ``None``.

    A read is a non-empty list of line objects. Anything else (a not-read
    entry, an empty list, a list holding text, ``null`` or a nested list) is
    a page NOT READ. Every reader of the page reads (rows, bill spans, the
    reply parser and the saved reads) asks this one question, so a page is
    never read for one of them and not read for another."""
    if not isinstance(entry, list) or not entry:
        return None
    if not all(isinstance(line, Mapping) for line in entry):
        return None
    return entry


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
    for page, entry in keyed:
        lines = read_lines(entry)
        if lines is None:
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
    check_sum_lines(out.rows)
    return out


def check_sum_lines(rows: Sequence[g.BillRow]) -> list[g.BillRow]:
    """Flag every sum line that adds up to none of the runs it can close.

    Rows are walked page by page in page order, in their order on the page.
    A line the model typed as a sum of other lines (:data:`SUM_LINE_TYPES`)
    can close one of three nested runs, and it agrees when its amount equals
    the sum of any one of them:

    ====================  ==================================================
    run                   lines summed
    ====================  ==================================================
    section               every non-sum line with an amount since the last
                          sum line on its page (or the page top)
    page                  every non-sum line with an amount above it on its
                          page, whatever sum lines stand between
    bill                  every ``item`` amount of its bill so far, across
                          pages; its bill is the one the line names, else the
                          one bill the items of its section name (none when
                          they name no bill or more than one); when the
                          bill the line names has no run, the one bill of its
                          section
    stretch               every ``item`` amount across pages since the last
                          item naming another bill, items naming no bill
                          included; offered to a line whose bill is the
                          stretch's or unknown on either side, and closed by a
                          line that names a bill and agrees with it
    ====================  ==================================================

    The bill and stretch runs are counted from the pages read, so they are
    not offered on a page that prints a brought-forward amount above the
    line: there the page run, which adds that amount, is the document's own
    run across pages, and an earlier page may not have been read.

    A sum line with no amount closes no run: the section, the stretch and the
    bills its section names carry on past it (``_section_bills`` reads the
    same boundary).

    "Equals" is within one unit of the amount's own last printed place (the
    same arithmetic ``bill_aggregate.row_kinds`` decides a collection by).
    When the section holds at least one ``item`` line and no run agrees, the
    line's check is :data:`CHECK_SUM_LINE`. A section with no item line (a
    collection page listing other pages' totals) is closed without a verdict:
    the page itself holds nothing to add up. A row that already carries a
    check keeps it. Returns the rows flagged here."""
    flagged: list[g.BillRow] = []
    by_page: dict[int, list[g.BillRow]] = {}
    for row in rows:
        by_page.setdefault(row.source_page, []).append(row)
    bill_items: dict[str, float] = {}
    stretch, stretch_bill = 0.0, ""
    for page in sorted(by_page):
        section = on_page = 0.0
        items = 0
        named: set[str] = set()
        carried_in = False
        for row in by_page[page]:
            kind = _text(row.extra.get(LINE_TYPE_LABEL)).lower()
            text = _text(row.fields.get("total"))
            amount = g.parse_number(text)
            if kind not in SUM_LINE_TYPES:
                if amount is None:
                    continue
                carried_in = carried_in or kind == BROUGHT_FORWARD
                section += amount
                on_page += amount
                if kind == ITEM:
                    items += 1
                    bill = _row_bill(row)
                    if bill:
                        named.add(bill)
                        bill_items[bill] = bill_items.get(bill, 0.0) + amount
                        if stretch_bill and bill != stretch_bill:
                            stretch = 0.0
                        stretch_bill = bill
                    stretch += amount
                continue
            if amount is None:
                continue
            own = _row_bill(row)
            single = next(iter(named)) if len(named) == 1 else ""
            bill = own if own in bill_items else (single or own)
            runs = [section, on_page]
            if bill in bill_items and not carried_in:
                runs.append(bill_items[bill])
            if not carried_in and (not bill or not stretch_bill or bill == stretch_bill):
                runs.append(stretch)
            place = agg._last_place(text)
            if items and not row.check and not any(
                    agg._same_figure(amount, run, place) for run in runs):
                row.check = CHECK_SUM_LINE
                flagged.append(row)
            if own and agg._same_figure(amount, stretch, place):
                stretch, stretch_bill = 0.0, ""
            section, items, named = 0.0, 0, set()
    return flagged


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
    bill = _text(line.get("bill"))
    if bill:
        extra[BILL_LABEL] = bill
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


@dataclass(frozen=True)
class PageInContext:
    """A page to read with what the page before it ended on: the bill the
    model's read of that page last named (``bill_before``), or ``None`` when
    there is no page before, it was not read, or its read named no bill."""

    index: int
    source: str = ""
    words: tuple = ()
    bill_before: str | None = None


def bill_ended(entry: Any) -> str | None:
    """The bill the last line of a page read names, as printed; ``None`` when
    the page was not read or no line of it names a bill (summary lines list a
    whole bill's total and are not where the page's text runs to)."""
    lines = read_lines(entry)
    if lines is None:
        return None
    for line in reversed(lines):
        if _text(line.get("type")).lower() == SUMMARY:
            continue
        bill = _text(line.get(BILL_KEY))
        if bill:
            return bill
    return None


def rows_of_model_reads(
    read_page: Callable[[Any], Any], bill_label: str = "",
) -> Callable[[Sequence[Any]], tuple[list[g.BillRow], list[int], dict[str, Any]]]:
    """A ``bill_derived.RowsOfReads`` from a per-page model reader.

    ``read_page`` gets one ``bill_classify.PageRead`` (0-based ``index``) and
    returns the model's lines for that page (or anything else, which is NOT
    READ). The rows and the 1-based pages not read are what
    :func:`rows_from_page_reads` gives for those pages; the third value is
    every reply exactly as the model gave it, by 1-based page, so the copy
    keeps the raw output beside the rows it accepted.

    Pages are read in order, and each is handed over as a
    :class:`PageInContext` carrying the bill the read of the page before it
    ended on (:func:`bill_ended`), so the model, not code, says which bill
    the lines of a page that does not print its bill continue."""

    def rows_of(
        page_reads: Sequence[Any],
    ) -> tuple[list[g.BillRow], list[int], dict[str, Any]]:
        pages: dict[str, Any] = {}
        before: str | None = None
        for r in sorted(page_reads, key=lambda pr: pr.index):
            got = read_page(PageInContext(r.index, getattr(r, "source", ""),
                                          tuple(getattr(r, "words", ()) or ()), before))
            pages[str(r.index + 1)] = got
            before = bill_ended(got)
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


#: Added to the instruction when the page before was read and its last line
#: named a bill: the model says which lines continue it; code never does.
PAGE_READ_BEFORE = (
    " The page before this one ended in bill {bill}. A line of this page that "
    "continues that bill gives it in bill; a line under a new bill gives the "
    "new one."
)


def prompt_for(prompt: str, bill_before: Any) -> str:
    """``prompt`` with the page before's last bill added when there is one."""
    bill = _text(bill_before)
    return prompt + PAGE_READ_BEFORE.format(bill=json.dumps(bill)) if bill else prompt


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
    ``lines`` list, an empty list, or a list that holds anything but line
    objects (:func:`read_lines`) all give ``None``: the page was not read,
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
    return read_lines(data.get("lines") if isinstance(data, dict) else None)


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


#: The not-read entry's list of failed calls, one code per call that raised:
#: the error class, and the HTTP status when the error carries a response
#: (``HTTPStatusError:401``), so a key or provider fault is never read as a
#: page the model could not read.
ERRORS_KEY = "errors"


def attempt_error(exc: BaseException) -> str:
    """The code of one failed call: class name, plus ``:<status>`` when the
    error carries an HTTP response. Never the message (it may quote a key)."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    name = type(exc).__name__
    return f"{name}:{status}" if isinstance(status, int) else name


def model_page_reader(
    ask: Callable[[list[dict[str, Any]]], Any],
    render_png: Callable[[int], bytes],
    *,
    prompt: str = PAGE_READ_PROMPT,
    attempts: int = PAGE_READ_ATTEMPTS,
    ask_again: Callable[[list[dict[str, Any]]], Any] | None = None,
) -> Callable[[Any], Any]:
    """A ``read_page`` for :func:`rows_of_model_reads`.

    ``ask_again``, when given, makes every call after the first: the same
    messages through another ask (in production, the provider asked for a
    plain reply -- see :func:`provider_ask`). Without it every call goes
    through ``ask``.

    For one ``bill_classify.PageRead`` it renders the page (``render_png`` gets
    the 0-based index), sends it with ``prompt`` through ``ask`` (messages ->
    reply text) and returns the reply's lines. An empty or non-JSON reply, or a
    call that raises, is tried again up to ``attempts`` calls in all; then the
    page is a not-read entry (:data:`NOT_READ_KEY`), which :func:`rows_from_page_reads` counts as
    NOT READ. A figure is never filled in here."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    def read_page(page_read: Any) -> Any:
        messages = page_read_messages(
            prompt_for(prompt, getattr(page_read, "bill_before", None)),
            render_png(page_read.index))
        replies: list[str] = []
        errors: list[str] = []
        for attempt in range(attempts):
            try:
                text = (ask_again if ask_again is not None and attempt else ask)(messages)
            except _CALL_ERRORS as exc:  # an attempt that failed, never a page of no rows
                code = attempt_error(exc)
                replies.append(code)
                errors.append(code)
                continue
            lines = lines_of_reply(text)
            if lines is not None:
                return lines
            replies.append(str(text)[:_REPLY_KEPT])
        return {NOT_READ_KEY: True, "attempts": attempts, "replies": replies,
                ERRORS_KEY: errors}

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


def provider_ask(
    max_tokens: int = PAGE_READ_MAX_TOKENS, *, plain_reply: bool = False,
) -> Callable[[list[dict[str, Any]]], str]:
    """``ask`` through the product's provider client (``llm_client.complete``):
    the configured provider and model, its usage recording and fallback.

    ``plain_reply`` asks the provider to reply without its hidden reasoning.
    Measured on a dense page of a priced bill: with the reasoning on, the
    page's sum line was read right, but the reasoning took 5,400 to 16,000
    tokens of the page's allowance and about half the replies came back
    empty; with it off the page was read every time, in a fifth of the
    tokens, with every item right and one digit of that sum line wrong. So
    the first call keeps the reasoning and the plain reply is for the call
    after an empty one (``model_page_reader(ask_again=...)``)."""
    import asyncio

    from app.core import llm_client

    def ask(messages: list[dict[str, Any]]) -> str:
        return asyncio.run(llm_client.complete(
            messages, temperature=0.0, max_tokens=max_tokens,
            agent_name="bill_page_read", plain_reply=plain_reply))

    return ask


ITEM = "item"
STATE_RECONCILED = "RECONCILED"
STATE_FAILED = "FAILED"
STATE_UNCHECKED = "UNCHECKED"
STATE_NOT_READ = "NOT READ"
STATE_TOTAL_ONLY = "TOTAL ONLY"
#: A bill some item row could belong to that the model named no bill for:
#: code cannot check its sum, so it is never FAILED for those rows.
STATE_NOT_CHECKED = "NOT CHECKED"


@dataclass(frozen=True)
class BillSpan:
    """One bill of the document as its own summary states it: label, 1-based
    inclusive page range, and the printed bill total (``None`` = not printed)."""

    label: str
    first_page: int | None
    last_page: int | None
    printed_total: float | None = None
    #: Pages not read outside the range that the bill could still reach (up to
    #: the next page another bill names, or the end of the document). The bill
    #: cannot be checked while one of them is unread, so it is NOT READ.
    pages_unread_reach: tuple[int, ...] = ()
    #: Read pages holding item lines that name no bill and that no bill's range
    #: holds (or a page naming two bills), which this bill could own. Code
    #: cannot place those rows, so the bill is NOT CHECKED, never FAILED.
    pages_unplaced_reach: tuple[int, ...] = ()


_NOT_MONEY = frozenset({REF_LABEL, LINE_TYPE_LABEL, BILL_LABEL})


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

    A row belongs to the bill its own line names (the model's read, line by
    line); code never moves a row into a bill by its page. Only
    rows the model typed ``item`` are summed. The state is ``NOT READ`` when a
    page of the bill was not read, ``UNCHECKED`` when the bill prints no total,
    ``TOTAL ONLY`` when the bill prints a total and has item rows but not one
    of them carries a money figure in any column (rate, amount, or a money
    column of its own): the document prints no item prices for that bill.
    ``FAILED`` when the item sum differs from the printed total by more than
    ``tolerance`` (the caller states it from the bill's own rounding) or any
    row of the bill is flagged (a sum line that does not add up and names no
    bill counts against the bills the items it closes name, or, when they name
    none, every bill whose range holds or reaches its page), else
    ``RECONCILED``. An item row whose line
    names no bill could belong to every bill whose range holds its page or
    that reaches it (``pages_unplaced_reach``): each such bill is ``NOT
    CHECKED`` with the count (``rows_unplaced``), never FAILED for them. Rows
    under no bill are counted, never moved into one. Figures are read, never
    repaired."""
    out: list[dict[str, Any]] = []
    placed: set[int] = set()
    missing = set(pages_not_read)
    own = [_row_bill(r) for r in rows]
    closes = {i: _section_bills(rows, own, i) for i, r in enumerate(rows)
              if not own[i] and r.check == CHECK_SUM_LINE}
    for span in spans:
        ranged = span.first_page is not None and span.last_page is not None
        key = _bill_key(span.label)
        members = [i for i in range(len(rows)) if own[i] and own[i] == key]
        reach = set(span.pages_unplaced_reach)
        rows_unplaced = sum(
            1 for i, r in enumerate(rows)
            if not own[i] and _text(r.extra.get(LINE_TYPE_LABEL)).lower() == ITEM
            and ((ranged and span.first_page <= r.source_page <= span.last_page)
                 or r.source_page in reach))
        placed.update(members)
        items_sum = 0.0
        summed = without = items = priced = 0
        # A sum line that names no bill and does not add up fails the bills
        # the items it closes name; when those items name none, every bill
        # whose range holds or reaches its page.
        flagged = sum(
            1 for i, bills in closes.items()
            if (key in bills if bills else
                ((ranged and span.first_page <= rows[i].source_page <= span.last_page)
                 or rows[i].source_page in reach)))
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
        not_read = sorted(p for p in missing
                          if (ranged and span.first_page <= p <= span.last_page)
                          or p in span.pages_unread_reach)
        printed = span.printed_total
        diff = None if printed is None else items_sum - printed
        unplaced = sorted(span.pages_unplaced_reach)
        if not_read:
            state = STATE_NOT_READ
        elif rows_unplaced:
            state = STATE_NOT_CHECKED
        elif diff is None:
            state = STATE_UNCHECKED
        elif items and not priced and not flagged:
            state = STATE_TOTAL_ONLY
        elif abs(diff) > tolerance or flagged:
            state = STATE_FAILED
        else:
            state = STATE_RECONCILED
        out.append({"label": span.label,
                    "pages": [span.first_page, span.last_page] if ranged else None,
                    "rows": len(members), "rows_summed": summed,
                    "rows_without_amount": without, "rows_flagged": flagged,
                    "item_rows_priced": priced,
                    "items_sum": items_sum, "printed_total": printed, "diff": diff,
                    "pages_not_read": not_read, "pages_unplaced": unplaced,
                    "rows_unplaced": rows_unplaced,
                    "state": state})
    return {"bills": out, "rows_unassigned": len(rows) - len(placed)}


SUMMARY = "summary"
BILL_KEY = "bill"


def _bill_key(value: Any) -> str:
    """A bill's identifier compared without case, spacing or punctuation."""
    return " ".join(re.findall(r"[^\W_]+", _text(value).casefold()))


def _section_bills(rows: Sequence[g.BillRow], own: Sequence[str], at: int) -> frozenset[str]:
    """The bills the ``item`` lines of the section a sum line closes name:
    the rows above ``rows[at]`` on its page, back to the sum line before it
    that has an amount (a sum line with none closes no run, as in
    :func:`check_sum_lines`)."""
    bills: set[str] = set()
    for j in range(at - 1, -1, -1):
        row = rows[j]
        if row.source_page != rows[at].source_page:
            break
        kind = _text(row.extra.get(LINE_TYPE_LABEL)).lower()
        if kind in SUM_LINE_TYPES and g.parse_number(row.fields.get("total", "")) is not None:
            break
        if kind == ITEM and own[j]:
            bills.add(own[j])
    return frozenset(bills)


def _row_bill(row: g.BillRow) -> str:
    """The bill a row's own line names (compared form); empty for a summary row,
    which lists a bill's total and is no row of that bill."""
    if _text(row.extra.get(LINE_TYPE_LABEL)).lower() == SUMMARY:
        return ""
    return _bill_key(row.extra.get(BILL_LABEL))


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
    the first one the reads print.

    A page not read is never charged to the document: the summary prints each
    bill's total, not its pages, so a bill's last read page is not proof of its
    edge. Every unread page a bill could reach (outside its range, up to the
    nearest page another bill or a split page names, or the document's end) is
    kept on its span (``pages_unread_reach``), so the bill is NOT READ, not
    FAILED. A bill the summary lists that no read page names gets a span with
    no range when some page was not read, reaching every unread page outside
    the other bills' ranges; with every page read it is only listed.

    A read page is never charged to the document either. Item lines that name
    no bill on a page no bill's range holds, or on a page naming two bills,
    are rows code cannot place: every bill that could own them keeps the page
    (``pages_unplaced_reach``, the same reach as an unread page; a page naming
    two bills is reached by those two), so the bill is NOT CHECKED, not FAILED.
    Item lines that do name their bill are placed by that name
    (``bill_states``), wherever the page sits."""
    out = SpanRead()
    label: dict[str, str] = {}
    members: dict[str, list[int]] = {}
    totals: dict[str, set[float]] = {}
    unread: list[int] = []
    #: Read pages holding an item line that names no bill, with the bills
    #: their other lines name.
    unnamed_items: dict[int, set[str]] = {}
    for key, lines in pages.items():
        name = _text(key)
        if not name.isdecimal():
            continue
        read = read_lines(lines)
        if read is None:
            unread.append(int(name))
            continue
        page = int(name)
        named: set[str] = set()
        loose = False
        for line in read:
            bill = _bill_key(line.get(BILL_KEY))
            if not bill:
                loose = loose or _text(line.get("type")).lower() == ITEM
                continue
            label.setdefault(bill, _text(line.get(BILL_KEY)))
            if _text(line.get("type")).lower() == SUMMARY:
                text, ok = _figure(line.get("amount"))
                amount = g.parse_number(text) if ok and text else None
                if amount is not None:
                    totals.setdefault(bill, set()).add(amount)
                continue
            named.add(bill)
        if loose:
            unnamed_items[page] = named
        if len(named) > 1:
            out.pages_split.append(page)
        elif named:
            members.setdefault(next(iter(named)), []).append(page)
    out.pages_split.sort()
    ranges = {b: (min(ps), max(ps)) for b, ps in members.items()}

    def printed_of(bill: str) -> float | None:
        found = totals.get(bill, set())
        if len(found) > 1:
            out.totals_conflicting.append(label[bill])
        return next(iter(found)) if len(found) == 1 else None

    def in_range(p: int) -> bool:
        return any(a <= p <= z for a, z in ranges.values())

    #: Unplaced pages a fence-bounded reach can give: no bill named, no range.
    unplaced = [p for p, names in unnamed_items.items() if not names and not in_range(p)]

    def reach_of(pages_: Sequence[int], first: int, last: int,
                 low: int | None, high: int | None) -> list[int]:
        return [p for p in pages_
                if (p < first and (low is None or p > low))
                or (p > last and (high is None or p < high))]

    for bill, (first, last) in sorted(ranges.items(), key=lambda kv: kv[1]):
        others = [p for other, ps in members.items() if other != bill for p in ps]
        inside = any(first <= p <= last for p in others)
        if inside or any(first <= p <= last for p in out.pages_split):
            out.bills_interleaved.append(label[bill])
            continue
        fences = others + out.pages_split
        low = max((p for p in fences if p < first), default=None)
        high = min((p for p in fences if p > last), default=None)
        reach = tuple(sorted(reach_of(unread, first, last, low, high)))
        split_here = [p for p in out.pages_split if bill in unnamed_items.get(p, ())]
        unplaced_reach = tuple(sorted(
            reach_of(unplaced, first, last, low, high) + split_here))
        out.spans.append(BillSpan(label[bill], first, last, printed_of(bill), reach,
                                  unplaced_reach))
    out.summary_without_pages = sorted(label[b] for b in totals if b not in members)
    loose = tuple(sorted(p for p in unread if not in_range(p)))
    loose_rows = tuple(sorted(unplaced))
    if loose or loose_rows:
        for bill in sorted((b for b in totals if b not in members), key=lambda b: label[b]):
            out.spans.append(BillSpan(label[bill], None, None, printed_of(bill), loose,
                                      loose_rows))
    return out


#: One page's saved reply under the run's work directory (0-based index).
SAVED_READ_NAME = "model_page_{index:05d}.json"


#: A saved read's file: what the page was asked (a digest of the instruction
#: sent with it) and the lines the model gave.
SAVED_ASKED = "asked"
SAVED_READ = "read"


def page_question(prompt: str = PAGE_READ_PROMPT) -> Callable[[Any], str]:
    """What a page is asked, as :func:`model_page_reader` sends it with
    ``prompt``: the instruction, with the bill the page before ended on when
    there is one (:func:`prompt_for`)."""

    def question(page_read: Any) -> str:
        return prompt_for(prompt, getattr(page_read, "bill_before", None))

    return question


def asked_digest(question: str) -> str:
    """The digest a saved read carries for the question it answers."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def saved_read_for(saved: Any, asked: str) -> list[Mapping[str, Any]] | None:
    """The lines of a saved file when it is a read of the page made with the
    question whose digest is ``asked``; ``None`` for anything else -- a read
    made with another question (no bill before, another instruction), a file
    saved before the question was recorded, or a file that is not a read."""
    if not isinstance(saved, Mapping) or saved.get(SAVED_ASKED) != asked:
        return None
    return read_lines(saved.get(SAVED_READ))


def saved_page_reader(
    read_page: Callable[[Any], Any], work_dir: str,
    question: Callable[[Any], str] | None = None,
) -> Callable[[Any], Any]:
    """``read_page`` that keeps each page's lines in ``work_dir`` and gives the
    saved lines back on a later run, so the plan and its execute see the same
    reads and the same plan hash.

    A saved read answers one question: the instruction the page was sent with,
    which carries the bill the page before ended on (``question``; the default
    is what :func:`model_page_reader` sends with the default instruction, and
    a reader built with another instruction passes
    ``page_question(that_instruction)``). A read is given back only for that
    same question. When the page before is read differently this time -- it
    failed last run and names its bill now -- the page is asked again, with
    the context it now has; so is a page whose file was saved under another
    instruction. Only a read (:func:`read_lines`) is saved: a not-read entry
    is returned as it is and the page is asked again on the next run, and a
    saved file that is not a read is asked again too."""
    ask_of = question or page_question()

    def read(page_read: Any) -> Any:
        asked = asked_digest(ask_of(page_read))
        path = os.path.join(work_dir, SAVED_READ_NAME.format(index=page_read.index))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                kept = saved_read_for(json.load(fh), asked)
            if kept is not None:
                return kept
        got = read_page(page_read)
        if read_lines(got) is not None:
            os.makedirs(work_dir, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({SAVED_ASKED: asked, SAVED_READ: got}, fh, ensure_ascii=False)
            os.replace(tmp, path)
        return got

    return read


def model_bill_read(
    read_page: Callable[[Any], Any], *, tolerance: float,
) -> tuple[Callable[[Sequence[Any]], Any], Callable[[Sequence[g.BillRow], Sequence[int]], dict[str, Any]]]:
    """The ``rows_of_reads`` and ``states_of_rows`` pair for ``bill_derived``.

    The rows come from the model's page reads; the states check them against
    the bill spans :func:`spans_from_page_reads` takes from the same reads, so
    the page ranges and printed totals are the document's own. What the spans
    could not place (pages naming two bills, interleaved bills, conflicting
    totals, summary bills with no page) is stored with the states."""
    seen: dict[str, Any] = {}
    rows_of = rows_of_model_reads(read_page)

    def rows(page_reads: Sequence[Any]) -> tuple[list[g.BillRow], list[int], dict[str, Any]]:
        got_rows, not_read, pages = rows_of(page_reads)
        seen.clear()
        seen.update(pages)
        return got_rows, not_read, pages

    def states(got_rows: Sequence[g.BillRow], not_read: Sequence[int]) -> dict[str, Any]:
        found = spans_from_page_reads(seen)
        out = bill_states(got_rows, not_read, spans=found.spans, tolerance=tolerance)
        out["tolerance"] = tolerance
        out["pages_split"] = list(found.pages_split)
        out["bills_interleaved"] = list(found.bills_interleaved)
        out["totals_conflicting"] = list(found.totals_conflicting)
        out["summary_without_pages"] = list(found.summary_without_pages)
        return out

    return rows, states
