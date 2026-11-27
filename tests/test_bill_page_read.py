"""Rows from a model's read of bill pages (``app.core.bill_page_read``)."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from app.core import bill_aggregate as agg
from app.core import bill_classify as bc
from app.core import bill_page_read as bpr
from app.core import bill_table_geometry as g


def _item(ref, q, r, a, desc="work", unit="m2", **kw):
    return {"type": "item", "ref": ref, "description": desc, "unit": unit,
            "quantity": q, "rate": r, "amount": a, **kw}


def _head(text):
    return {"type": "heading", "description": text}


def _carried(a):
    return {"type": "carried", "description": "to next", "amount": a}


@pytest.mark.parametrize("q,r,a", [(10, 5, 50), (3, 2.5, 7.5)])
def test_item_that_multiplies_is_accepted(q, r, a):
    out = bpr.rows_from_page_reads({"7": [_item("A", q, r, a)]})
    (row,) = out.rows
    assert row.check == ""
    assert row.source_page == 7
    assert out.flagged == []


@pytest.mark.parametrize("q,r,a", [(10, 5, 51), (3, 2.5, 9)])
def test_item_that_does_not_multiply_is_flagged_and_kept_as_read(q, r, a):
    out = bpr.rows_from_page_reads({"7": [_item("A", q, r, a)]})
    (row,) = out.rows
    assert row.check == "qty x rate != amount"
    assert row.fields["total"] == str(a)  # never repaired
    assert out.flagged == [row]


@pytest.mark.parametrize("bad", ["12a", True])
def test_figure_that_is_not_a_number_is_flagged(bad):
    out = bpr.rows_from_page_reads({"1": [_item("A", 2, bad, 10)]})
    assert out.rows[0].check == bpr.CHECK_NOT_A_NUMBER


@pytest.mark.parametrize("value,cell", [(188996.0, "188996"), (1234.5, "1234.5"),
                                        ("1,234.50", "1,234.50"), (7, "7")])
def test_figure_cell_text(value, cell):
    out = bpr.rows_from_page_reads({"1": [{"type": "carried", "amount": value}]})
    assert out.rows[0].fields["total"] == cell


@pytest.mark.parametrize("pages", [{"3": []}, {"3": None}, {"3": "no json"},
                                   {"3": [_item("A", 1, 1, 1), "junk"]}])
def test_page_without_a_list_of_lines_is_not_read(pages):
    out = bpr.rows_from_page_reads({**pages, "4": [_item("B", 2, 2, 4)]})
    assert out.pages_not_read == [3]
    assert [r.source_page for r in out.rows] == [4]


@pytest.mark.parametrize("key", ["p3", "", "-2", "3.5", "³"])
def test_key_that_is_not_a_page_number_is_listed_not_dropped(key):
    out = bpr.rows_from_page_reads({key: [_item("A", 1, 1, 1)],
                                    "4": [_item("B", 2, 2, 4)]})
    assert out.keys_not_pages == [key]
    assert [r.source_page for r in out.rows] == [4]
    assert out.pages_not_read == []


@pytest.mark.parametrize("key, page", [(" 12 ", 12), (7, 7), ("007", 7), ("٣", 3)])
def test_page_number_key_is_a_page(key, page):
    out = bpr.rows_from_page_reads({key: [_item("A", 1, 1, 1)]})
    assert out.keys_not_pages == []
    assert [r.source_page for r in out.rows] == [page]


@pytest.mark.parametrize("label", ["Bill 7", "Part B"])
def test_heading_run_is_the_path_under_the_bill_label(label):
    pages = {"2": [_head("Earth"), _head("Digging"), _item("A", 1, 2, 2),
                   _item("B", 1, 3, 3), _head("Fill"), _item("C", 1, 4, 4)]}
    rows = bpr.rows_from_page_reads(pages, bill_label=label).rows
    assert [r.path for r in rows] == [(label, "Earth", "Digging")] * 2 + [(label, "Fill")]
    assert rows[2].section == "Fill"


@pytest.mark.parametrize("blank", ["", None])
def test_heading_without_words_adds_nothing_to_the_path(blank):
    pages = {"1": [_head("Earth"), {"type": "heading", "description": blank},
                   _item("A", 1, 2, 2)]}
    assert bpr.rows_from_page_reads(pages, bill_label="Bill 3").rows[0].path == (
        "Bill 3", "Earth")
    assert bpr.rows_from_page_reads(pages).rows[0].path == ("Earth",)


@pytest.mark.parametrize("label", ["", "   "])
def test_no_bill_label_gives_heading_path_only(label):
    rows = bpr.rows_from_page_reads({"1": [_head("H"), _item("A", 1, 1, 1)]},
                                    bill_label=label).rows
    assert rows[0].path == ("H",)


@pytest.mark.parametrize("first,second", [(9, 10), (99, 100)])
def test_pages_are_taken_in_page_order_and_the_path_carries_across(first, second):
    pages = {str(second): [_item("B", 1, 2, 2)],
             str(first): [_head("Roads"), _item("A", 1, 1, 1)],
             "x": [_item("Z", 1, 1, 1)]}
    rows = bpr.rows_from_page_reads(pages).rows
    assert [(r.source_page, r.path) for r in rows] == [
        (first, ("Roads",)), (second, ("Roads",))]


@pytest.mark.parametrize("cols", [{"Fixed Charge": "1,000", "Time Charge": 250},
                                  {"Labour": 40}])
def test_printed_money_columns_are_kept_as_their_own_columns(cols):
    line = {"type": "item", "ref": "A", "description": "x", "unit": "sum",
            "quantity": 1, "figures": cols}
    row = bpr.rows_from_page_reads({"1": [line]}).rows[0]
    for k, v in cols.items():
        assert row.extra[k] == str(v)
    assert row.fields["rate"] == "" and row.fields["total"] == ""
    assert row.check == ""


@pytest.mark.parametrize("cols", [{"Fixed Charge": "n/a"}, {"Time": False}])
def test_printed_money_column_not_a_number_is_flagged(cols):
    line = {"type": "item", "description": "x", "figures": cols}
    assert bpr.rows_from_page_reads({"1": [line]}).rows[0].check == bpr.CHECK_NOT_A_NUMBER


@pytest.mark.parametrize("kind", ["carried", "total"])
def test_non_heading_line_is_a_row_with_its_type(kind):
    line = {"type": kind, "description": "c", "amount": 9}
    row = bpr.rows_from_page_reads({"1": [line]}).rows[0]
    assert row.extra[bpr.LINE_TYPE_LABEL] == kind
    assert bpr.REF_LABEL not in row.extra


@pytest.mark.parametrize("a,b", [(100, 250), (1200, 37.5)])
def test_written_sheet_reads_back_through_the_aggregate_reader(tmp_path, a, b):
    """Public path: model lines -> rows -> the derived-sheet writer -> the sheet reader."""
    pages = {"5": [_head("Roads"), _item("A", 1, a, a), _item("B", 2, b / 2, b),
                   _carried(a + b)]}
    out = bpr.rows_from_page_reads(pages, bill_label="Bill 9")
    path = str(tmp_path / "d.xlsx")
    g.write_bill_workbook(out.rows, path)
    rows = agg.read_sheet_rows(path)
    assert [(r.ref, r.amount, r.pages) for r in rows] == [
        ("A", str(a), [5]), ("B", str(b), [5]), ("", str(a + b), [5])]
    assert all(r.path == ("Bill 9", "Roads") for r in rows)


@pytest.mark.parametrize("reads, want_pages, want_not_read", [
    ({0: [_item("A", 2, 3, 6)], 1: [_item("B", 4, 5, 20), _item("C", 1, 9, 9)]},
     [1, 2, 2], []),
    ({0: None, 1: "not lines", 2: [_head("Part"), _item("D", 7, 7, 49)]},
     [3], [1, 2]),
])
def test_rows_of_reads_each_page_and_numbers_it_from_one(reads, want_pages, want_not_read):
    from types import SimpleNamespace

    asked: list[int] = []

    def read_page(page_read):
        asked.append(page_read.index)
        return reads[page_read.index]

    rows_of = bpr.rows_of_model_reads(read_page, "Bill 9")
    rows, not_read, raw = rows_of([SimpleNamespace(index=i) for i in sorted(reads)])
    assert asked == sorted(reads)
    assert raw == {str(i + 1): reads[i] for i in reads}  # every reply, read or not
    assert [r.source_page for r in rows] == want_pages
    assert not_read == want_not_read
    assert all(tuple(r.path[:1]) == ("Bill 9",) for r in rows)


# --- per-bill state against the printed total -------------------------------

def _rows(pages):
    return bpr.rows_from_page_reads(pages).rows


def _states(pages, missing, *, spans, tolerance):
    """``bill_states`` of reads whose every line names the bill whose span
    holds its page, as the model names it (a page no span holds names none)."""
    named = {}
    for key, lines in pages.items():
        owner = next((sp.label for sp in spans if sp.first_page is not None
                      and sp.first_page <= int(key) <= sp.last_page), None)
        named[key] = [dict(ln, bill=owner) if owner and ln.get("type") != "summary"
                      else ln for ln in lines]
    return bpr.bill_states(_rows(named), missing, spans=spans, tolerance=tolerance)


_TWO_BILLS = {"1": [_item("A", 10, 5, 50), _item("B", 2, 3, 6)],
              "2": [_carried(56)],
              "3": [_item("C", 4, 25, 100), {"type": "item", "description": "lump"}],
              "5": [_item("D", 1, 7, 7)]}


@pytest.mark.parametrize("printed_one, printed_two, want_one, want_two", [
    (56, 100, bpr.STATE_RECONCILED, bpr.STATE_RECONCILED),
    (57, 99, bpr.STATE_FAILED, bpr.STATE_FAILED),
    (56.5, 100.5, bpr.STATE_RECONCILED, bpr.STATE_RECONCILED),
    (None, 100, bpr.STATE_UNCHECKED, bpr.STATE_RECONCILED),
])
def test_each_bill_is_checked_against_its_printed_total(printed_one, printed_two,
                                                        want_one, want_two):
    spans = [bpr.BillSpan("one", 1, 2, printed_one), bpr.BillSpan("two", 3, 4, printed_two)]
    got = _states(_TWO_BILLS, [], spans=spans, tolerance=0.5)
    one, two = got["bills"]
    assert (one["state"], two["state"]) == (want_one, want_two)
    # only item rows are summed: the carried line on page 2 is a row, not an item
    assert (one["rows"], one["rows_summed"], one["items_sum"]) == (3, 2, 56)
    assert (two["rows"], two["rows_summed"], two["rows_without_amount"]) == (2, 1, 1)
    assert two["items_sum"] == 100
    assert got["rows_unassigned"] == 1  # page 5 lies under no bill


@pytest.mark.parametrize("missing, want", [
    ([2], [bpr.STATE_NOT_READ, bpr.STATE_RECONCILED]),
    ([4], [bpr.STATE_RECONCILED, bpr.STATE_NOT_READ]),
    ([5], [bpr.STATE_RECONCILED, bpr.STATE_RECONCILED]),
])
def test_bill_with_a_page_not_read_is_not_read(missing, want):
    spans = [bpr.BillSpan("one", 1, 2, 56), bpr.BillSpan("two", 3, 4, 100)]
    got = _states(_TWO_BILLS, missing, spans=spans, tolerance=0)
    assert [b["state"] for b in got["bills"]] == want
    assert [b["pages_not_read"] for b in got["bills"]] == [
        [p for p in missing if 1 <= p <= 2], [p for p in missing if 3 <= p <= 4]]


@pytest.mark.parametrize("bad_amount", [57, "x"])
def test_bill_with_a_flagged_row_fails_even_when_the_sum_matches(bad_amount):
    pages = {"1": [_item("A", 10, 5, bad_amount)]}
    printed = 57 if bad_amount == 57 else 0
    got = _states(pages, [], spans=[bpr.BillSpan("b", 1, 1, printed)],
                          tolerance=0)
    (b,) = got["bills"]
    assert b["rows_flagged"] == 1
    assert b["state"] == bpr.STATE_FAILED


@pytest.mark.parametrize("tolerance, want", [(0.99, bpr.STATE_FAILED),
                                             (1, bpr.STATE_RECONCILED)])
def test_tolerance_boundary_is_inclusive(tolerance, want):
    got = _states({"1": [_item("A", 10, 5, 50)]}, [],
                          spans=[bpr.BillSpan("b", 1, 1, 51)], tolerance=tolerance)
    assert got["bills"][0]["state"] == want
    assert got["bills"][0]["diff"] == -1


@pytest.mark.parametrize("spans", [[], [bpr.BillSpan("far", 9, 12, 0)]])
def test_no_bill_holds_the_rows_they_are_unassigned(spans):
    rows = _rows(_TWO_BILLS)
    got = bpr.bill_states(rows, [], spans=spans, tolerance=0)
    assert got["rows_unassigned"] == len(rows)
    assert [b["rows"] for b in got["bills"]] == [0] * len(spans)


# --- a bill whose document prints no item prices ----------------------------

def _unpriced(ref, q, **kw):
    return {"type": "item", "ref": ref, "description": "work", "unit": "nr",
            "quantity": q, **kw}


@pytest.mark.parametrize("pages, printed", [
    ({"1": [_unpriced("A", 3), _unpriced("B", None)]}, 1000),
    ({"1": [_unpriced("A", None, rate=None, amount="")], "2": [_unpriced("C", 7)]}, 25),
])
def test_bill_with_blank_money_columns_is_total_only(pages, printed):
    got = _states(pages, [], spans=[bpr.BillSpan("b", 1, 2, printed)],
                          tolerance=0)
    (b,) = got["bills"]
    assert b["state"] == bpr.STATE_TOTAL_ONLY
    assert (b["item_rows_priced"], b["items_sum"], b["printed_total"]) == (0, 0, printed)


@pytest.mark.parametrize("pages, printed, want", [
    # one item prints a money column of its own: priced, so the sum is checked
    ({"1": [_unpriced("A", 3), _unpriced("B", 1, figures={"Fixed Charge": 500})]},
     1000, bpr.STATE_FAILED),
    # a rate is printed but no amount: priced, the bill does not add up
    ({"1": [_unpriced("A", 3, rate=10)]}, 30, bpr.STATE_FAILED),
    # no total printed: nothing to check against
    ({"1": [_unpriced("A", 3)]}, None, bpr.STATE_UNCHECKED),
    # no item rows at all: not "total only"
    ({"1": [_carried(None)]}, 1000, bpr.STATE_FAILED),
    # a flagged row is never hidden behind "total only"
    ({"1": [_unpriced("A", "three")]}, 1000, bpr.STATE_FAILED),
    # every item priced and summing: reconciled, not total only
    ({"1": [_item("A", 2, 5, 10)]}, 10, bpr.STATE_RECONCILED),
])
def test_total_only_does_not_fire_when_a_price_or_total_says_otherwise(pages, printed,
                                                                       want):
    got = _states(pages, [], spans=[bpr.BillSpan("b", 1, 1, printed)],
                          tolerance=0)
    assert got["bills"][0]["state"] == want


@pytest.mark.parametrize("missing", [[1], [2]])
def test_total_only_bill_with_a_page_not_read_is_not_read(missing):
    pages = {"1": [_unpriced("A", 3)], "2": [_unpriced("B", 4)]}
    got = _states(pages, missing, spans=[bpr.BillSpan("b", 1, 2, 9)],
                          tolerance=0)
    assert got["bills"][0]["state"] == bpr.STATE_NOT_READ


# --- read_page: the model reads a page image; an empty or non-JSON reply is NOT READ ---


def _pr(index):
    return bc.PageRead(index=index, source=bc.SOURCE_OCR, words=())


@pytest.mark.parametrize("text, want", [
    (json.dumps({"lines": [_item("A", 1, 2, 2)]}), [_item("A", 1, 2, 2)]),
    ("```json\n" + json.dumps({"lines": [_carried(9), _head("X")]}) + "\n```",
     [_carried(9), _head("X")]),
])
def test_reply_with_lines_gives_them(text, want):
    assert bpr.lines_of_reply(text) == want


@pytest.mark.parametrize("text", [
    "", "   ", "not json", '{"lines": []}', '{"rows": [{"type": "item"}]}',
    '[{"type": "item"}]', '{"lines": "none"}', None, 42,
])
def test_reply_without_lines_is_none(text):
    assert bpr.lines_of_reply(text) is None


class _Ask:
    """Replies in turn; an Exception instance is raised instead of returned."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _good(*lines):
    return json.dumps({"lines": list(lines)})


@pytest.mark.parametrize("index, png", [(0, b"\x89PNG-a"), (276, b"\x89PNG-bb")])
def test_page_image_and_prompt_go_in_one_message(index, png):
    seen = []
    ask = _Ask(_good(_item("A", 1, 1, 1)))
    read = bpr.model_page_reader(ask, lambda i: seen.append(i) or png, prompt="P" + str(index))
    read(_pr(index))
    assert seen == [index]
    ((msg,),) = ask.calls
    assert msg["role"] == "user"
    text, image = msg["content"]
    assert text == {"type": "text", "text": "P" + str(index)}
    url = image["image_url"]["url"]
    assert image["type"] == "image_url" and url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == png


@pytest.mark.parametrize("replies, calls", [
    ((_good(_item("A", 2, 3, 6)),), 1),
    (("", _good(_item("A", 2, 3, 6))), 2),
    (("not json", _good(_item("A", 2, 3, 6))), 2),
    ((TimeoutError("t"), _good(_item("A", 2, 3, 6))), 2),
])
def test_one_retry_on_empty_bad_or_failed_reply(replies, calls):
    ask = _Ask(*replies)
    got = bpr.model_page_reader(ask, lambda i: b"x")(_pr(0))
    assert got == [_item("A", 2, 3, 6)]
    assert len(ask.calls) == calls


@pytest.mark.parametrize("replies, kept, errors", [
    (("", "{}"), ["", "{}"], []),
    ((ValueError("v"), '{"lines": []}'), ["ValueError", '{"lines": []}'], ["ValueError"]),
])
def test_page_still_without_lines_after_retry_is_not_read(replies, kept, errors):
    ask = _Ask(*replies, _good(_item("never", 1, 1, 1)))
    got = bpr.model_page_reader(ask, lambda i: b"x")(_pr(3))
    assert got == {bpr.NOT_READ_KEY: True, "attempts": 2, "replies": kept,
                   bpr.ERRORS_KEY: errors}
    assert len(ask.calls) == 2


@pytest.mark.parametrize("first, second, want", [
    ("x" * 1000, "y" * 201, [200, 200]),
    ("x" * 200, "short", [200, 5]),
])
def test_long_reply_is_cut_in_the_not_read_entry(first, second, want):
    got = bpr.model_page_reader(_Ask(first, second), lambda i: b"x")(_pr(0))
    assert [len(r) for r in got["replies"]] == want


@pytest.mark.parametrize("error", [RuntimeError("key unset"), RuntimeError("502")])
def test_provider_error_on_every_attempt_is_not_read(error):
    ask = _Ask(error, error)
    got = bpr.model_page_reader(ask, lambda i: b"x")(_pr(1))
    assert got == {bpr.NOT_READ_KEY: True, "attempts": 2,
                   "replies": ["RuntimeError", "RuntimeError"],
                   bpr.ERRORS_KEY: ["RuntimeError", "RuntimeError"]}
    assert len(ask.calls) == 2


@pytest.mark.parametrize("attempts", [1, 3])
def test_attempts_is_the_number_of_calls(attempts):
    ask = _Ask(*([""] * 5))
    got = bpr.model_page_reader(ask, lambda i: b"x", attempts=attempts)(_pr(0))
    assert len(ask.calls) == attempts == got["attempts"]


@pytest.mark.parametrize("attempts", [0, -1])
def test_attempts_below_one_is_refused(attempts):
    with pytest.raises(ValueError):
        bpr.model_page_reader(_Ask(), lambda i: b"x", attempts=attempts)


@pytest.mark.parametrize("bad_index", [0, 1])
def test_not_read_page_reaches_the_copy_as_not_read_and_raw(bad_index):
    good = _good(_item("A", 2, 5, 10))
    replies = {bad_index: ["", ""], 1 - bad_index: [good]}
    calls = {0: 0, 1: 0}

    def ask_for(index):
        def ask(messages):
            r = replies[index][calls[index]]
            calls[index] += 1
            return r
        return ask

    def read_page(page_read):
        return bpr.model_page_reader(ask_for(page_read.index), lambda i: b"x")(page_read)

    rows, not_read, raw = bpr.rows_of_model_reads(read_page, "Bill")([_pr(0), _pr(1)])
    assert not_read == [bad_index + 1]
    assert [r.source_page for r in rows] == [2 - bad_index]
    assert raw[str(bad_index + 1)][bpr.NOT_READ_KEY] is True
    json.dumps(raw)  # the raw output is stored as JSON beside the copy


@pytest.mark.parametrize("max_tokens", [bpr.PAGE_READ_MAX_TOKENS, 12000])
def test_provider_ask_goes_through_the_product_client(monkeypatch, max_tokens):
    from app.core import llm_client

    seen = {}

    async def fake_complete(messages, **kw):
        seen.update(kw, messages=messages)
        return "reply-" + str(kw["max_tokens"])

    monkeypatch.setattr(llm_client, "complete", fake_complete)
    msgs = bpr.page_read_messages("p", b"img")
    assert bpr.provider_ask(max_tokens)(msgs) == "reply-" + str(max_tokens)
    assert seen["messages"] is msgs and seen["max_tokens"] == max_tokens
    assert seen["temperature"] == 0.0


def test_default_allowance_is_large():
    # a short allowance comes back empty: the model spends reasoning tokens first
    assert bpr.PAGE_READ_MAX_TOKENS >= 8000 and bpr.PAGE_READ_ATTEMPTS == 2


@pytest.mark.parametrize("index, dpi", [(0, 72), (1, 144)])
def test_render_page_png_renders_the_asked_page(tmp_path, index, dpi):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "two.pdf"
    doc = fitz.open()
    for size in ((200, 100), (100, 300)):
        doc.new_page(width=size[0], height=size[1])
    doc.save(str(path))
    doc.close()
    png = bpr.render_page_png(str(path), index, dpi=dpi)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pix = fitz.Pixmap(png)
    want = ((200, 100), (100, 300))[index]
    assert (pix.width, pix.height) == (round(want[0] * dpi / 72), round(want[1] * dpi / 72))


# --- bill spans from the page reads ------------------------------------------

def _line(bill, kind="item", amount=None, **kw):
    return {"type": kind, "bill": bill, "description": kw.pop("description", "x"),
            "amount": amount, **kw}


@pytest.mark.parametrize("one, two, pages_one, pages_two", [
    ("A", "B", (3, 4), (5, 7)),
    ("Section 9", "Section 10", (10, 12), (13, 13)),
])
def test_each_bill_spans_its_first_to_last_page(one, two, pages_one, pages_two):
    pages = {}
    for p in range(pages_one[0], pages_one[1] + 1):
        pages[str(p)] = [_line(one, amount=1)]
    for p in range(pages_two[0], pages_two[1] + 1):
        pages[str(p)] = [_line(two, amount=1)]
    got = bpr.spans_from_page_reads(pages)
    assert got.spans == [bpr.BillSpan(one, *pages_one, None),
                         bpr.BillSpan(two, *pages_two, None)]
    assert not (got.pages_split or got.bills_interleaved or got.summary_without_pages)


@pytest.mark.parametrize("on_page, on_summary, total", [
    ("Bill No. 2", "BILL NO 2", 32882840),
    ("bill 7 - roads", "Bill 7 Roads", "1500.5"),
])
def test_summary_line_gives_the_printed_total_whatever_the_case_or_punctuation(
        on_page, on_summary, total):
    pages = {"5": [_line(on_page, amount=10)], "6": [_line(on_page, amount=20)],
             "1": [_line(on_summary, kind="summary", amount=total)]}
    got = bpr.spans_from_page_reads(pages)
    assert got.spans == [bpr.BillSpan(on_page, 5, 6, float(total))]
    assert got.summary_without_pages == []


@pytest.mark.parametrize("amount", [None, "n/a"])
def test_summary_line_without_a_number_gives_no_total(amount):
    pages = {"2": [_line("A", amount=3)], "1": [_line("A", kind="summary", amount=amount)]}
    assert bpr.spans_from_page_reads(pages).spans == [bpr.BillSpan("A", 2, 2, None)]


@pytest.mark.parametrize("bill", ["A", "Bill 4"])
def test_summary_page_is_not_a_page_of_the_bill_it_lists(bill):
    pages = {"1": [_line(bill, kind="summary", amount=9)],
             "8": [_line(bill, amount=9)]}
    assert bpr.spans_from_page_reads(pages).spans == [bpr.BillSpan(bill, 8, 8, 9.0)]


@pytest.mark.parametrize("gap", [{"4": None}, {"4": [{"type": "heading", "description": "h"}]}])
def test_page_not_read_or_naming_no_bill_inside_a_range_is_inside_the_bill(gap):
    pages = {"3": [_line("A", amount=1)], **gap, "5": [_line("A", amount=1)]}
    assert bpr.spans_from_page_reads(pages).spans == [bpr.BillSpan("A", 3, 5, None)]


@pytest.mark.parametrize("before, after", [(2, 4), (10, 30)])
def test_page_naming_two_bills_is_reported_and_in_no_span(before, after):
    mid = before + 1
    pages = {str(before): [_line("A")], str(mid): [_line("A"), _line("B")],
             str(after): [_line("B")]}
    got = bpr.spans_from_page_reads(pages)
    assert got.pages_split == [mid]
    assert got.spans == [bpr.BillSpan("A", before, before), bpr.BillSpan("B", after, after)]


@pytest.mark.parametrize("split_page", [5, 6])
def test_bill_whose_range_holds_a_split_page_gets_no_span(split_page):
    pages = {"4": [_line("A")], str(split_page): [_line("A"), _line("B")], "7": [_line("A")]}
    got = bpr.spans_from_page_reads(pages)
    assert got.bills_interleaved == ["A"]
    assert got.spans == []


@pytest.mark.parametrize("order", [("A", "B", "A"), ("X", "Y", "X")])
def test_bill_whose_range_holds_another_bills_page_gets_no_span(order):
    pages = {str(i + 1): [_line(b)] for i, b in enumerate(order)}
    got = bpr.spans_from_page_reads(pages)
    assert got.bills_interleaved == [order[0]]
    assert got.spans == [bpr.BillSpan(order[1], 2, 2)]


@pytest.mark.parametrize("first, second, want_total, want_conflict", [
    (100, 200, None, ["A"]),
    (100, "100", 100.0, []),
])
def test_bill_listed_twice_with_different_totals_has_no_total(first, second,
                                                               want_total, want_conflict):
    pages = {"1": [_line("A", kind="summary", amount=first),
                   _line("A", kind="summary", amount=second)],
             "3": [_line("A", amount=1)]}
    got = bpr.spans_from_page_reads(pages)
    assert got.spans == [bpr.BillSpan("A", 3, 3, want_total)]
    assert got.totals_conflicting == want_conflict


@pytest.mark.parametrize("listed", [["Q"], ["Q", "R"]])
def test_summary_bill_with_no_pages_is_listed(listed):
    pages = {"1": [_line(b, kind="summary", amount=5) for b in listed],
             "2": [_line("A", amount=5)]}
    got = bpr.spans_from_page_reads(pages)
    assert got.summary_without_pages == sorted(listed)
    assert got.spans == [bpr.BillSpan("A", 2, 2)]


@pytest.mark.parametrize("pages, reach", [
    ({"cover": [_line("A")], "2": [_line("B")]}, ()),
    # the not-read page 1 places nothing, but bill B could reach it
    # a list holding no line object is a page NOT READ too: B could reach it
    ({"1": {"not_read": True}, "2": [_line("B")], "3": ["not a line"]}, (1, 3)),
])
def test_keys_that_are_not_pages_and_reads_that_are_not_lists_place_nothing(pages, reach):
    assert bpr.spans_from_page_reads(pages).spans == [bpr.BillSpan("B", 2, 2, None, reach)]


@pytest.mark.parametrize("bill", [None, "", " - "])
def test_line_naming_no_bill_places_no_page(bill):
    assert bpr.spans_from_page_reads({"1": [_line(bill)]}).spans == []


def test_page_read_prompt_asks_for_the_bill_and_summary_lines():
    assert '\\"bill\\"' in json.dumps(bpr.PAGE_READ_PROMPT)
    assert '\\"summary\\"' in json.dumps(bpr.PAGE_READ_PROMPT)


@pytest.mark.parametrize("printed, want", [(30, "RECONCILED"), (31, "FAILED")])
def test_spans_from_reads_feed_the_bill_check(printed, want):
    pages = {"1": [_line("A", kind="summary", amount=printed)],
             "2": [_line("A", quantity=1, rate=10, amount=10)],
             "3": [_line("A", quantity=2, rate=10, amount=20)]}
    got = bpr.rows_from_page_reads(pages)
    span = bpr.spans_from_page_reads(pages)
    states = bpr.bill_states(got.rows, got.pages_not_read, spans=span.spans, tolerance=0)
    assert [b["state"] for b in states["bills"]] == [want]
    assert states["bills"][0]["pages"] == [2, 3]


@pytest.mark.parametrize("pages, tolerance, want_states, want_split, want_conflict, want_listed", [
    # the printed total is 0.5 off the items: the stated tolerance decides
    ({0: [_line("A", kind="summary", amount=30.5)],
      1: [_line("A", quantity=1, rate=10, amount=10)],
      2: [_line("A", quantity=2, rate=10, amount=20)]},
     0.5, ["RECONCILED"], [], [], []),
    ({0: [_line("A", kind="summary", amount=30.5)],
      1: [_line("A", quantity=1, rate=10, amount=10)],
      2: [_line("A", quantity=2, rate=10, amount=20)]},
     0.0, ["FAILED"], [], [], []),
    ({0: [_line("A", kind="summary", amount=30), _line("A", kind="summary", amount=31),
          _line("C", kind="summary", amount=5)],
      1: [_line("A", quantity=1, rate=10, amount=10)],
      2: [_line("A", quantity=2, rate=10, amount=20), _line("B", quantity=1, rate=1, amount=1)]},
     0.5, ["UNCHECKED"], [3], ["A"], ["C"]),
])
def test_model_bill_read_states_check_the_spans_of_the_same_reads(
    pages, tolerance, want_states, want_split, want_conflict, want_listed,
):
    from types import SimpleNamespace

    rows, states = bpr.model_bill_read(lambda p: pages[p.index], tolerance=tolerance)
    got_rows, not_read, raw = rows([SimpleNamespace(index=i) for i in sorted(pages)])
    assert raw == {str(i + 1): pages[i] for i in pages}
    out = states(got_rows, not_read)
    assert [b["state"] for b in out["bills"]] == want_states
    assert out["tolerance"] == tolerance
    assert out["pages_split"] == want_split
    assert out["totals_conflicting"] == want_conflict
    assert sorted(out["summary_without_pages"]) == want_listed


@pytest.mark.parametrize("first, second", [(30, 31), (31, 30)])
def test_model_bill_read_states_use_the_latest_reads_only(first, second):
    from types import SimpleNamespace

    # the first run read one page more: its bill must not reach the second states
    runs = [{0: [_line("A", kind="summary", amount=first)],
             1: [_line("A", quantity=3, rate=10, amount=30)],
             2: [_line("B", quantity=1, rate=1, amount=1)]},
            {0: [_line("A", kind="summary", amount=second)],
             1: [_line("A", quantity=3, rate=10, amount=30)]}]
    current: list[dict] = []
    rows, states = bpr.model_bill_read(lambda p: current[0][p.index], tolerance=0)
    for run in runs:
        current[:] = [run]
        got_rows, not_read, _ = rows([SimpleNamespace(index=i) for i in sorted(run)])
    out = states(got_rows, not_read)
    assert [b["state"] for b in out["bills"]] == ["RECONCILED" if second == 30 else "FAILED"]
    assert out["bills"][0]["printed_total"] == second


@pytest.mark.parametrize("unread, want_reach", [
    ({"3": None}, (3,)),            # last page of the bill
    ({"1": None}, (1,)),            # first page of the bill
    ({"1": None, "3": None}, (1, 3)),
])
def test_unread_page_at_a_bill_edge_is_in_its_reach(unread, want_reach):
    pages = {"2": [_line("A", amount=1)], **unread,
             "4": [_line("A", kind="summary", amount=1)]}
    assert bpr.spans_from_page_reads(pages).spans == [
        bpr.BillSpan("A", 2, 2, 1.0, want_reach)]


@pytest.mark.parametrize("unread", ["3", "4"])
def test_unread_page_between_two_bills_is_in_both_reaches_and_no_further(unread):
    pages = {"1": [_line("A", amount=1)], "2": [_line("A", amount=1)],
             unread: None, "5": [_line("B", amount=1)], "7": None}
    got = bpr.spans_from_page_reads(pages).spans
    assert got == [bpr.BillSpan("A", 1, 2, None, (int(unread),)),
                   bpr.BillSpan("B", 5, 5, None, (int(unread), 7))]


@pytest.mark.parametrize("fence", [
    {"3": [_line("B", amount=1)]},
    {"3": [_line("B", amount=1), _line("C", amount=1)]},   # a split page
])
def test_unread_page_beyond_another_bills_page_is_not_in_the_reach(fence):
    pages = {"1": [_line("A", amount=1)], **fence, "4": None}
    a = bpr.spans_from_page_reads(pages).spans[0]
    assert (a.label, a.pages_unread_reach) == ("A", ())


@pytest.mark.parametrize("pages", [
    {"1": [_line("A", amount=1)], "2": [_line("A", amount=1)]},
    {"1": [_line("A", amount=1)], "2": [{"type": "heading", "description": "h"}]},
])
def test_every_page_read_gives_no_reach(pages):
    assert bpr.spans_from_page_reads(pages).spans[0].pages_unread_reach == ()


@pytest.mark.parametrize("unread", [["2"], ["2", "3"]])
def test_summary_bill_with_no_read_page_reaches_the_unread_pages(unread):
    pages = {"1": [_line("A", amount=4)], **{p: None for p in unread},
             "9": [_line("A", kind="summary", amount=4), _line("B", kind="summary", amount=7)]}
    got = bpr.spans_from_page_reads(pages)
    loose = tuple(int(p) for p in unread)
    assert got.spans == [bpr.BillSpan("A", 1, 1, 4.0, loose),
                         bpr.BillSpan("B", None, None, 7.0, loose)]
    assert got.summary_without_pages == ["B"]
    out = bpr.bill_states([], [int(p) for p in unread], spans=got.spans, tolerance=0.0)
    b = out["bills"][1]
    assert (b["state"], b["pages"], b["pages_not_read"]) == (
        bpr.STATE_NOT_READ, None, list(loose))


@pytest.mark.parametrize("inside", ["2", "3"])
def test_summary_bill_gets_no_span_when_the_only_unread_page_is_inside_another_bill(inside):
    pages = {"1": [_line("A", amount=1)], inside: None, "4": [_line("A", amount=1)],
             "9": [_line("B", kind="summary", amount=7)]}
    got = bpr.spans_from_page_reads(pages)
    assert [s.label for s in got.spans] == ["A"]
    assert got.summary_without_pages == ["B"]



@pytest.mark.parametrize("entry", [
    {"not_read": True}, None, [], ["page too faint to read"], [None], [[]],
    [{"type": "item"}, "and some text"],
])
def test_read_lines_says_not_read_for_anything_but_line_objects(entry):
    assert bpr.read_lines(entry) is None


@pytest.mark.parametrize("entry", [
    [{"type": "item", "amount": "5"}],
    [{"type": "heading", "description": "x"}, {"type": "item", "amount": "7"}],
])
def test_read_lines_gives_a_read_back_as_it_is(entry):
    assert bpr.read_lines(entry) is entry


@pytest.mark.parametrize("lines", ['["page too faint to read"]', "[null]", "[[]]"])
def test_reply_whose_lines_are_not_line_objects_gives_no_lines(lines):
    assert bpr.lines_of_reply('{"lines": ' + lines + '}') is None


@pytest.mark.parametrize("junk", [["page too faint"], [None], [[]]])
@pytest.mark.parametrize("edge", ["1", "3"])
def test_junk_read_on_a_bills_edge_page_is_in_its_reach(junk, edge):
    pages = {"1": [_line("A", amount=1)], "2": [_line("A", amount=1)],
             "3": [_line("A", amount=1)], "4": [_line("A", kind="summary", amount=3)]}
    pages[edge] = junk
    (span,) = bpr.spans_from_page_reads(pages).spans
    assert span.pages_unread_reach == (int(edge),)
    rows = bpr.rows_from_page_reads(pages)
    assert rows.pages_not_read == [int(edge)]


@pytest.mark.parametrize("offset", [0, 10])
def test_each_bill_reaches_only_the_unread_pages_next_to_it(offset):
    # A, unread, B, unread, C, unread, D, unread: a bill reaches the unread page
    # on each side up to the next bill, never every unread page on that side
    pages = {}
    for i, bill in enumerate("ABCD"):
        pages[str(offset + 2 * i + 1)] = [_line(bill, amount=1)]
        pages[str(offset + 2 * i + 2)] = {"not_read": True}
    got = {s.label: s.pages_unread_reach for s in bpr.spans_from_page_reads(pages).spans}
    o = offset
    assert got == {"A": (o + 2,), "B": (o + 2, o + 4), "C": (o + 4, o + 6), "D": (o + 6, o + 8)}


@pytest.mark.parametrize("figure", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("key", ["amount", "rate"])
def test_reply_with_a_non_finite_figure_is_flagged(figure, key):
    reply = '{"lines": [{"type": "item", "quantity": 1, "' + key + '": ' + figure + '}]}'
    lines = bpr.model_page_reader(_Ask(reply), lambda i: b"x")(_pr(0))
    (row,) = bpr.rows_from_page_reads({"1": lines}).rows
    assert row.check == bpr.CHECK_NOT_A_NUMBER


@pytest.mark.parametrize("bug", [AttributeError("a"), ZeroDivisionError("z")])
def test_reader_bug_propagates_and_is_never_a_page_not_read(bug):
    with pytest.raises(type(bug)):
        bpr.model_page_reader(_Ask(bug, _good(_item("A", 1, 1, 1))), lambda i: b"x")(_pr(0))


def _http_error(code):
    req = httpx.Request("POST", "http://p.invalid")
    return httpx.HTTPStatusError("no", request=req, response=httpx.Response(code, request=req))


@pytest.mark.parametrize("exc, want", [
    (RuntimeError("key unset"), "RuntimeError"),
    (_http_error(401), "HTTPStatusError:401"),
    (_http_error(503), "HTTPStatusError:503"),
    (httpx.ConnectError("refused"), "ConnectError"),
])
def test_attempt_error_is_the_class_and_status_never_the_message(exc, want):
    assert bpr.attempt_error(exc) == want


@pytest.mark.parametrize("bill, own", [("Bill 2", "Bill 2"), (None, None), ("  ", None)])
def test_row_keeps_the_bill_its_line_names_and_it_is_not_money(bill, own):
    line = {"type": "item", "bill": bill, "amount": None}
    (row,) = bpr.rows_from_page_reads({"1": [line]}).rows
    assert row.extra.get(bpr.BILL_LABEL) == own
    # a bill name is not a printed price: an item with no figures is unpriced
    assert not bpr._carries_money(row)
    priced = bpr.rows_from_page_reads({"1": [{**line, "amount": 5}]}).rows[0]
    assert bpr._carries_money(priced)


# --- the page before's ending, given to the model; code never places a row --


@pytest.mark.parametrize("entry, want", [
    ([_item("A", 1, 1, 1, bill="Bill 3"), _item("B", 1, 1, 1, bill="Bill 4")], "Bill 4"),
    ([_item("A", 1, 1, 1, bill="Bill 3"), _item("B", 1, 1, 1)], "Bill 3"),
    # a summary line lists a whole bill's total: not where the page's text ends
    ([_item("A", 1, 1, 1, bill="Bill 3"), {"type": "summary", "bill": "Bill 9"}], "Bill 3"),
    # must not fire: no line names a bill, the page was not read, an empty read
    ([_item("A", 1, 1, 1)], None),
    (None, None),
    ([], None),
    ({bpr.NOT_READ_KEY: True}, None),
])
def test_bill_ended_is_the_last_bill_a_line_of_the_read_names(entry, want):
    assert bpr.bill_ended(entry) == want


@pytest.mark.parametrize("before, added", [
    ("Bill 2", True), ("B-7 Drainage", True), (None, False), ("", False), ("  ", False),
])
def test_prompt_carries_the_page_before_only_when_it_named_a_bill(before, added):
    got = bpr.prompt_for("PROMPT.", before)
    want = "PROMPT." + (bpr.PAGE_READ_BEFORE.format(bill=json.dumps(before)) if added else "")
    assert got == want


@pytest.mark.parametrize("pages, want_before", [
    ({0: [_item("A", 1, 1, 1, bill="X")], 1: [_item("B", 1, 1, 1)],
      2: [_item("C", 1, 1, 1, bill="Y")]}, [None, "X", None]),  # page 2 named none: nothing carried
    ({0: None, 1: [_item("B", 1, 1, 1, bill="Y")], 2: [_item("C", 1, 1, 1)]},
     [None, None, "Y"]),
])
def test_pages_are_read_in_order_each_with_the_bill_before_it(pages, want_before):
    seen = []

    def read_page(page):
        seen.append((page.index, page.bill_before))
        return pages[page.index]

    reads = [SimpleNamespace(index=i) for i in reversed(range(len(pages)))]
    bpr.rows_of_model_reads(read_page)(reads)
    assert seen == list(enumerate(want_before))


@pytest.mark.parametrize("label, unnamed", [("!!", 2), ("  ", 1)])
def test_a_label_with_no_bill_in_it_collects_no_unnamed_row(label, unnamed):
    rows = _rows({"1": [_item(str(i), 1, 1, 1) for i in range(unnamed)]})
    (bill,) = bpr.bill_states(rows, [], spans=[bpr.BillSpan(label, 1, 1, 1)],
                              tolerance=0)["bills"]
    assert (bill["rows"], bill["rows_unplaced"], bill["state"]) == (
        0, unnamed, bpr.STATE_NOT_CHECKED)


@pytest.mark.parametrize("sum_line, printed, want", [
    (_carried(50), 51, bpr.STATE_RECONCILED),
    ({"type": "total", "description": "page total", "amount": 51}, 52, bpr.STATE_FAILED),
])
def test_a_sum_line_naming_no_bill_inside_the_range_unplaces_nothing(sum_line, printed, want):
    """Only an item line can be a bill's unplaced row: a carried or total line
    on the bill's own page that names no bill leaves the bill checked."""
    rows = _rows({"1": [_item("A", 10, 5, 50, bill="b"), sum_line],
                  "2": [_item("B", 1, 1, 1, bill="b")]})
    (bill,) = bpr.bill_states(rows, [], spans=[bpr.BillSpan("b", 1, 2, printed)],
                              tolerance=0)["bills"]
    assert (bill["rows_unplaced"], bill["state"]) == (0, want)


# --- rows no bill is named for, in the reach of a bill ---

@pytest.mark.parametrize("label, page, amount", [("Bill A", 1, 10), ("Section 2", 3, 40)])
def test_a_bill_with_no_printed_total_and_an_unnamed_row_in_reach_is_not_checked(label, page, amount):
    """The bill prints no total, and the page after it holds an item that
    names no bill: the bill is NOT CHECKED with that row counted, not
    UNCHECKED as a bill that merely prints no total is."""
    pages = {str(page): [_line(label, amount=amount)], str(page + 1): [_line(None, amount=5)]}
    found = bpr.spans_from_page_reads(pages)
    assert found.spans == [bpr.BillSpan(label, page, page, None, (), (page + 1,))]
    states = bpr.bill_states(bpr.rows_from_page_reads(pages).rows, [], spans=found.spans, tolerance=0.5)
    (bill,) = states["bills"]
    assert bill["state"] == bpr.STATE_NOT_CHECKED and bill["state"] != bpr.STATE_UNCHECKED
    assert bill["rows_unplaced"] == 1 and bill["pages_unplaced"] == [page + 1]
    assert bill["printed_total"] is None and bill["diff"] is None and bill["items_sum"] == float(amount)
    # the control: the same bill with no unnamed row after it is UNCHECKED
    alone = {str(page): [_line(label, amount=amount)]}
    control = bpr.bill_states(bpr.rows_from_page_reads(alone).rows, [],
                              spans=bpr.spans_from_page_reads(alone).spans, tolerance=0.5)
    assert control["bills"][0]["state"] == bpr.STATE_UNCHECKED


@pytest.mark.parametrize("one, two, only_summary, first", [("A", "B", "C", 1), ("X", "Y", "Z", 5)])
def test_a_page_naming_two_bills_with_an_unnamed_row_is_unplaced_for_both(one, two, only_summary, first):
    """Page ``first + 1`` names both bills and also holds a row naming none:
    it is in neither span, both bills are NOT CHECKED for it, and a bill the
    summary lists but no page names gets no span."""
    mid, last, summary = first + 1, first + 2, first + 3
    pages = {
        str(first): [_line(one, amount=1)],
        str(mid): [_line(one, amount=1), _line(two, amount=1), _line(None, amount=1)],
        str(last): [_line(two, amount=1)],
        str(summary): [_line(one, kind="summary", amount=2), _line(two, kind="summary", amount=2),
                       _line(only_summary, kind="summary", amount=9)],
    }
    found = bpr.spans_from_page_reads(pages)
    assert found.spans == [bpr.BillSpan(one, first, first, 2.0, (), (mid,)),
                           bpr.BillSpan(two, last, last, 2.0, (), (mid,))]
    assert found.pages_split == [mid] and found.summary_without_pages == [only_summary]
    assert only_summary not in [span.label for span in found.spans]
    states = bpr.bill_states(bpr.rows_from_page_reads(pages).rows, [], spans=found.spans, tolerance=0.5)
    assert [(b["label"], b["state"], b["pages_unplaced"], b["rows_unplaced"]) for b in states["bills"]] == [
        (one, bpr.STATE_NOT_CHECKED, [mid], 1), (two, bpr.STATE_NOT_CHECKED, [mid], 1)]


# --- a page's sum line is checked against the lines above it ----------------

def _sum_line(kind, a, bill=None):
    return {"type": kind, "description": "sum", "amount": a, "bill": bill}


def _bill_pages(bill, page_items, page_totals, collection_total, listed=None):
    """A bill as a summary page, one page per item list closed by its page
    total, and a collection page listing the page totals (``listed``, by
    default the page totals as read) and the bill total."""
    pages = {"1": [{"type": "summary", "bill": bill, "description": "bill",
                    "amount": collection_total}]}
    for n, (items, total) in enumerate(zip(page_items, page_totals), start=2):
        pages[str(n)] = [dict(_item(f"R{n}{k}", q, r, q * r), bill=bill)
                         for k, (q, r) in enumerate(items)]
        pages[str(n)].append(_sum_line("total", total, bill))
    last = str(len(page_items) + 2)
    pages[last] = [_sum_line("total", t, bill) for t in (listed or page_totals)]
    pages[last].append(_sum_line("total", collection_total, bill))
    return pages


#: Two bills' figures: (bill, items per page, the page totals as printed,
#: the bill total, the first page total as a read slipped one digit).
_BILLS = [
    ("7", [[(10, 5), (2, 3)], [(4, 25)]], [56, 100], 156, 65),
    ("18", [[(3, 4788883), (1, 1)], [(2, 39660000)]],
     [14366650, 79320000], 93686650, 14386650),
]


def _stored_and_aggregate(tmp_path, pages):
    """The public path: model reads -> stored states; the written sheet ->
    the aggregate's own check of the same bill."""
    reads = {int(k) - 1: v for k, v in pages.items()}
    rows_of, states_of = bpr.model_bill_read(lambda pr: reads[pr.index], tolerance=0)
    rows, not_read, _raw = rows_of([SimpleNamespace(index=i) for i in sorted(reads)])
    stored = states_of(rows, not_read)
    path = str(tmp_path / "bill.xlsx")
    g.write_bill_workbook(rows, path)
    items = agg.join_page_continuations(agg.read_sheet_rows(path))
    ranges = agg.bill_page_ranges({"bill_states": stored})
    checks = agg.bill_checks(items, [b["label"] for b in stored["bills"]], ranges)
    return rows, stored["bills"][0], checks[0]


@pytest.mark.parametrize("bill, items, totals, grand, _slip", _BILLS)
def test_page_totals_read_right_store_and_aggregate_both_reconcile(
        tmp_path, bill, items, totals, grand, _slip):
    rows, stored, aggregate = _stored_and_aggregate(
        tmp_path, _bill_pages(bill, items, totals, grand))
    assert not [r for r in rows if r.check]
    assert stored["state"] == bpr.STATE_RECONCILED
    assert aggregate["state"] == agg.STATE_RECONCILED
    assert aggregate["items_sum"] == stored["items_sum"] == grand


@pytest.mark.parametrize("bill, items, totals, grand, slip", _BILLS)
def test_page_total_read_wrong_is_flagged_and_no_copy_reconciles(
        tmp_path, bill, items, totals, grand, slip):
    rows, stored, aggregate = _stored_and_aggregate(
        tmp_path, _bill_pages(bill, items, [slip] + totals[1:], grand, listed=totals))
    flagged = [r for r in rows if r.check]
    assert [(r.source_page, r.fields["total"], r.check) for r in flagged] == [
        (2, str(slip), bpr.CHECK_SUM_LINE)]
    assert stored["state"] == bpr.STATE_FAILED
    assert stored["rows_flagged"] == 1
    assert aggregate["state"] == agg.STATE_FAILED


@pytest.mark.parametrize("kind", sorted(bpr.SUM_LINE_TYPES))
@pytest.mark.parametrize("amounts, total, want", [
    ([50, 6], 56, ""), ([50, 6], 58, bpr.CHECK_SUM_LINE),
    ([1200, 37.5], 1237.5, ""), ([1200, 37.5], 1236.4, bpr.CHECK_SUM_LINE),
])
def test_each_sum_type_is_checked_against_the_items_above_it(kind, amounts, total, want):
    lines = [_item(f"A{k}", 1, a, a) for k, a in enumerate(amounts)]
    rows = _rows({"4": lines + [_sum_line(kind, total)]})
    assert rows[-1].check == want
    assert not any(r.check for r in rows[:-1])


@pytest.mark.parametrize("total, printed, want", [
    ("56.4", "56.4", ""), ("56.4", "56", ""), ("56.6", "56", ""),
    ("57.0", "56", ""), ("57.01", "56", bpr.CHECK_SUM_LINE),
    ("56.4", "56.0", bpr.CHECK_SUM_LINE), ("57", "57.00", ""),
    ("57.02", "57.00", bpr.CHECK_SUM_LINE),
])
def test_sum_line_tolerance_is_its_own_last_printed_place(total, printed, want):
    """The run adds to ``total``; the line prints ``printed``: it agrees
    within one unit of the last place it prints, the boundary inclusive."""
    rows = _rows({"4": [_item("A", 1, total, total), _sum_line("carried", printed)]})
    assert rows[-1].check == want


@pytest.mark.parametrize("listed", [[56, 100], [14366650, 79320000, 1]])
def test_collection_page_with_no_item_line_gives_no_verdict(listed):
    """Page totals listed on a collection page, then a line that is not their
    sum: the page holds no item, so nothing here can be checked."""
    lines = [_sum_line("total", t) for t in listed] + [_sum_line("total", sum(listed) + 9)]
    assert not any(r.check for r in _rows({"9": lines}))


@pytest.mark.parametrize("bf, items, carried, want", [
    (100, [20, 5], 125, ""), (100, [20, 5], 25, bpr.CHECK_SUM_LINE),
    (7, [3], 10, ""), (7, [3], 3, bpr.CHECK_SUM_LINE),
])
def test_brought_forward_is_added_to_the_run_not_checked(bf, items, carried, want):
    lines = [_sum_line("brought_forward", bf)]
    lines += [_item(f"A{k}", 1, a, a) for k, a in enumerate(items)]
    rows = _rows({"3": lines + [_sum_line("carried", carried)]})
    assert rows[0].check == ""
    assert rows[-1].check == want


@pytest.mark.parametrize("first, second", [((50, 6), (100,)), ((3, 4), (9, 1))])
def test_each_sum_line_closes_its_run_and_each_page_is_its_own(first, second):
    """Two runs on one page, then a page whose total restates only its own items."""
    page = [_item("A", 1, a, a) for a in first] + [_sum_line("subtotal", sum(first))]
    page += [_item("B", 1, a, a) for a in second] + [_sum_line("subtotal", sum(second))]
    nxt = [_item("C", 1, a, a) for a in second] + [_sum_line("total", sum(second))]
    rows = _rows({"2": page, "3": nxt})
    assert not any(r.check for r in rows)
    wrong = _rows({"2": page[:-1] + [_sum_line("subtotal", sum(first) + sum(second))]})
    assert wrong[-1].check == bpr.CHECK_SUM_LINE


@pytest.mark.parametrize("amount", [None, ""])
def test_sum_line_without_an_amount_is_not_checked_and_keeps_the_run(amount):
    rows = _rows({"2": [_item("A", 1, 5, 5), _sum_line("total", amount),
                        _item("B", 1, 6, 6), _sum_line("total", 11)]})
    assert [r.check for r in rows] == ["", "", "", ""]
    rows = _rows({"2": [_item("A", 1, 5, 5), _sum_line("total", amount),
                        _item("B", 1, 6, 6), _sum_line("total", 6)]})
    assert rows[-1].check == bpr.CHECK_SUM_LINE


@pytest.mark.parametrize("q, r, a", [(2, 3, 7), (4, 5, 21)])
def test_sum_line_keeps_a_check_it_already_has_and_item_checks_stand(q, r, a):
    rows = _rows({"2": [_item("A", q, r, a), _sum_line("total", "x")]})
    assert rows[0].check and rows[0].check != bpr.CHECK_SUM_LINE
    assert rows[-1].check == bpr.CHECK_NOT_A_NUMBER
    assert bpr.check_sum_lines(rows) == []


@pytest.mark.parametrize("named", [None, ""])
def test_unnamed_sum_line_that_does_not_add_up_fails_the_bill_on_its_page(named):
    spans = [bpr.BillSpan("7", 2, 3, 56), bpr.BillSpan("8", 4, 4, 9)]
    pages = {"2": [dict(_item("A", 10, 5, 50), bill="7"), dict(_item("B", 2, 3, 6), bill="7"),
                   _sum_line("total", 65, named)],
             "4": [dict(_item("C", 1, 9, 9), bill="8"), _sum_line("total", 9, named)]}
    got = {b["label"]: b for b in bpr.bill_states(_rows(pages), [], spans=spans,
                                                  tolerance=0)["bills"]}
    assert got["7"]["state"] == bpr.STATE_FAILED and got["7"]["rows_flagged"] == 1
    assert got["8"]["state"] == bpr.STATE_RECONCILED and got["8"]["rows_flagged"] == 0
    reach = [bpr.BillSpan("7", 3, 3, 56, pages_unplaced_reach=(2,))]
    got = bpr.bill_states(_rows(pages), [], spans=reach, tolerance=0)["bills"][0]
    assert got["rows_flagged"] == 1


@pytest.mark.parametrize("flagged_page, want", [(2, 1), (5, 0)])
def test_check_sum_lines_returns_what_it_flagged(flagged_page, want):
    rows = _rows({"2": [_item("A", 1, 5, 5), _item("B", 1, 6, 6)],
                  str(flagged_page): [_item("C", 1, 3, 3)]})
    rows.append(g.BillRow(section="", source_page=2, fields={"total": "12"},
                          extra={bpr.LINE_TYPE_LABEL: "total"}))
    got = bpr.check_sum_lines(rows)
    assert len(got) == want and all(r.check == bpr.CHECK_SUM_LINE for r in got)
