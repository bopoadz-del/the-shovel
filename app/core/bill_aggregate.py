"""Count and sum bill rows deterministically, over the derived bill sheet (P-B3).

A question that asks how many bill rows match a description, or for the total
quantity or amount of those rows, is answered here and not by the model adding
whichever row chunks retrieval happened to return. Retrieval gives the top K
chunks, so a sum built from them is short by every matching row that was not
retrieved.

The source is the derived spreadsheet ``bill_derived.derive_bill_sheet``
writes: one sheet per heading path, a banner row holding that path (outer to
inner, joined by ``bill_table_geometry._PATH_SEP``), the canonical columns, any
columns the page printed that map to no canonical field, then ``Check`` and
``Source Page``. Every row of that sheet is read; none is sampled.

An item the page break splits is one item. The extractor starts a new item at
a line that prints a reference, but each page starts fresh, so an item whose
description begins at the foot of one page and whose figures (or the rest of
its description) are printed at the head of the next page comes out as two
rows. :func:`join_page_continuations` puts them back together: a row on the
next page with no reference, directly following the previous page's last row
of the same heading path, where one of the two rows carries no figure, is the
rest of that item.

Matching is by words, not by a phrase: every term must appear as a word in
the row's description (and every section term in its heading path). Nothing
here names a bill, a section, a row or a question.

Which rows a call selects is derived from the user's ask, not from the words
the model chose to pass (:func:`selection_from_ask`): the same ask over the
same bill selects the same rows every time, however the model words its call.
Each result carries a ``statement`` naming the rows it used and the figure, so
the answer can say which rows were counted and quote the tool's own total.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core import bill_table_geometry as g

_LOG = logging.getLogger(__name__)

MEASURE_QUANTITY = "quantity"
MEASURE_AMOUNT = "amount"
MEASURES = (MEASURE_QUANTITY, MEASURE_AMOUNT)

_CANONICAL = {label: name for name, label in g._FIELD_LABELS}
_CHECK_LABEL = "Check"
_PAGE_LABEL = "Source Page"
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass
class SheetRow:
    """One item of the derived sheet, with every page it is printed on."""

    path: tuple[str, ...]
    ref: str
    description: str
    unit: str
    quantity: str
    rate: str
    amount: str
    pages: list[int] = field(default_factory=list)

    def figure(self, measure: str) -> float | None:
        text = self.quantity if measure == MEASURE_QUANTITY else self.amount
        return g.parse_number(text)

    def has_figure(self) -> bool:
        return any(g.parse_number(t) is not None for t in (self.quantity, self.rate, self.amount))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


_PAGE_RE = re.compile(r"\d+(?:\.0+)?")


def _page(value: Any) -> int | None:
    """The page number a ``Source Page`` cell holds; ``None`` when it holds none.

    A blank or non-numeric cell is a row whose page the sheet did not record,
    not an error, so it is told apart by its text rather than by catching.
    """
    text = "" if value is None else str(value).strip()
    if not _PAGE_RE.fullmatch(text):
        return None
    return int(float(text))


def read_sheet_rows(xlsx_path: str) -> list[SheetRow]:
    """Every row of every sheet of a derived bill workbook, in sheet order.

    ``xlsx_path`` is a stored document: ``bill_derived`` writes the sheet
    through the byte store, which encrypts it when a data key is set, so the
    workbook is read from its plaintext (``file_crypto.open_plaintext``).
    """
    from app.core import file_crypto

    with file_crypto.open_plaintext(xlsx_path) as readable:
        return carry_open_heading(_read_workbook_rows(readable))


_REF_NUMBER_RE = re.compile(r"^(.*?)(\d+)\s*$")


def _ref_number(ref: str) -> tuple[str, int] | None:
    m = _REF_NUMBER_RE.match((ref or "").strip())
    return (m.group(1).strip().casefold(), int(m.group(2))) if m else None


def _continues(path: tuple[str, ...], open_path: tuple[str, ...]) -> bool:
    """``path`` is the heading open at the page break, re-printed at the head
    of the next page: equal to it, or a strict prefix of it (``(Bill 3,)``
    under ``(Bill 3, Manholes)``, the path cut short by a repeated outer
    heading). A path whose last label EXTENDS the open label (more words) is
    NOT a continuation: a new subsection whose heading extends the open label
    reads the same, and merging it would add its rows to the open section's
    count and sum. A true 'continued' line is therefore left under its own
    heading (under-counted, the safe side). Only the labels are compared; no
    word of any bill is named."""
    n = len(path)
    return 0 < n <= len(open_path) and path == open_path[:n]


def carry_open_heading(rows: Sequence[SheetRow]) -> list[SheetRow]:
    """Rows in print order, the rows a page break moved under another heading
    put back under the heading that was open when the page broke.

    The derived sheet groups rows by heading path, and the extractor's path
    carries over a page break; a label re-printed at the head of the next page
    (the bill's running heading, or the section's 'continued' line) changes
    the path of the rows that follow it, so the rest of the section lands in
    another sheet. Here the rows are put back in print order (page, then the
    order the sheets were first printed), and the FIRST block of a page (rows
    of one sheet on that page) takes the path open at the end of the previous
    page when its own path continues it (:func:`_continues`: equal to the
    open path or a strict prefix of it) AND the item numbers are the
    evidence: the block's first reference has the open section's last
    reference's prefix and the next number. A heading alone (the running
    heading) is not evidence: it cannot be told from a new section without
    naming a word. A label re-printed with more words is never carried: it
    cannot be told from a new subsection that extends the open label.
    Only reading changes: the stored sheet is not touched.
    """
    blocks: list[tuple[int, int, list[SheetRow]]] = []
    sheet = -1
    prev_path: tuple[str, ...] | None = None
    for row in rows:
        if row.path != prev_path:
            sheet, prev_path = sheet + 1, row.path
        page = row.pages[0] if row.pages else -1
        if blocks and blocks[-1][1] == sheet and (blocks[-1][2][-1].pages[:1] or [-1])[0] == page:
            blocks[-1][2].append(row)
        else:
            blocks.append((page, sheet, [row]))
    if any(page < 0 for page, _, _ in blocks):
        # A row with no page: print order cannot be rebuilt; read as stored.
        return [SheetRow(**{**r.__dict__, "pages": list(r.pages)}) for r in rows]
    blocks.sort(key=lambda b: (b[0], b[1]))
    out: list[SheetRow] = []
    open_path: tuple[str, ...] | None = None
    open_ref: str = ""
    last_page = -1
    for page, _, block in blocks:
        path = block[0].path
        if open_path is not None and page == last_page + 1 and _continues(path, open_path):
            first = next((r.ref for r in block if r.ref), "")
            a, b = _ref_number(open_ref), _ref_number(first)
            # Carried only on positive evidence: the block's first reference
            # is the next number of the open section's sequence. An empty or
            # number-less reference, or a sequence that restarts or jumps,
            # leaves the block where it was printed (a wrong carry adds rows
            # of another section to this one's sum).
            if a and b and a[0] == b[0] and b[1] == a[1] + 1:
                path = open_path
        out.extend(SheetRow(**{**r.__dict__, "path": path, "pages": list(r.pages)}) for r in block)
        last_page = page
        open_path = path
        open_ref = next((r.ref for r in reversed(block) if r.ref), open_ref)
    return out


def _read_workbook_rows(xlsx_path: str) -> list[SheetRow]:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    out: list[SheetRow] = []
    try:
        for ws in wb.worksheets:
            path: tuple[str, ...] = ()
            header: list[str] | None = None
            for raw in ws.iter_rows(values_only=True):
                cells = [_cell(v) for v in raw]
                if header is None:
                    if "Description" in cells:
                        header = cells
                    elif any(cells):
                        banner = next(c for c in cells if c)
                        path = tuple(p.strip() for p in banner.split(g._PATH_SEP.strip()) if p.strip())
                    continue
                if not any(cells):
                    continue
                named = dict(zip(header, cells))
                # The extra columns sit between the canonical columns and Check,
                # in the order the pages printed them; the reference is the first.
                extras = [h for h in header if h and h not in _CANONICAL
                          and h not in (_CHECK_LABEL, _PAGE_LABEL)]
                page = _page(named.get(_PAGE_LABEL))
                out.append(SheetRow(
                    path=path,
                    ref=named.get(extras[0], "") if extras else "",
                    description=named.get("Description", ""),
                    unit=named.get("Unit", ""),
                    quantity=named.get("Quantity", ""),
                    rate=named.get("Rate", ""),
                    amount=named.get("Amount", ""),
                    pages=[page] if page is not None else [],
                ))
    finally:
        wb.close()
    return out


def join_page_continuations(rows: Sequence[SheetRow]) -> list[SheetRow]:
    """Rejoin each item the page break split into two rows (see module doc)."""
    out: list[SheetRow] = []
    for row in rows:
        prev = out[-1] if out else None
        if (
            prev is not None
            and not row.ref
            and row.path == prev.path
            and row.pages and prev.pages
            and row.pages[0] == prev.pages[-1] + 1
            and not (row.has_figure() and prev.has_figure())
        ):
            prev.description = " ".join(t for t in (prev.description, row.description) if t)
            for attr in ("unit", "quantity", "rate", "amount"):
                if not getattr(prev, attr):
                    setattr(prev, attr, getattr(row, attr))
            prev.pages.append(row.pages[0])
            continue
        out.append(SheetRow(**{**row.__dict__, "pages": list(row.pages)}))
    return out


#: Length units a size is printed in, and their factor to metres (the base
#: unit). Units, not sizes: no size of any bill is listed here.
_LENGTH_UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
                 "مم": 0.001, "ملم": 0.001, "سم": 0.01, "م": 1.0, "كم": 1000.0}
#: Arabic-Indic and Eastern Arabic-Indic digits, and the Arabic decimal and
#: thousands separators, read as the Latin ones.
_DIGITS = str.maketrans({**{chr(0x0660 + i): str(i) for i in range(10)},
                         **{chr(0x06F0 + i): str(i) for i in range(10)},
                         "٫": ".", "٬": ","})
_NUM = r"\d+(?:[.,]\d+)*"
_UNIT = "(?:" + "|".join(sorted(map(re.escape, _LENGTH_UNITS), key=len, reverse=True)) + r")(?![^\W_])"
_MEMBER = rf"{_NUM}(?:\s*{_UNIT})?"
_SIZE_SEP = r"\s*(?:x|×|\*|by|في)\s*"
#: A size: one number, or numbers joined by ``x`` (a dimension), each with
#: its unit when printed. Not a number glued to a letter (a code: ``DN150``).
_SIZE_RE = re.compile(rf"(?<![^\W_])(?<!\.){_MEMBER}(?:{_SIZE_SEP}{_MEMBER})*", re.IGNORECASE)
_MEMBER_RE = re.compile(rf"({_NUM})\s*({_UNIT})?", re.IGNORECASE)


def _number(text: str) -> float:
    """``1.8`` and ``1,8`` -> 1.8; ``1,800`` and ``1,800.5`` -> thousands."""
    seps = re.findall(r"[.,]", text)
    if not seps:
        return float(text)
    last = seps[-1]
    head, _, tail = text.rpartition(last)
    if all(s == last for s in seps) and (len(seps) > 1 or (last == "," and len(tail) == 3)):
        # One mark repeated, or a lone comma before three digits: thousands.
        return float(text.replace(last, ""))
    # The last mark is the decimal point; any other mark groups thousands.
    return float(re.sub(r"[.,]", "", head) + "." + tail)


@dataclass(frozen=True)
class Size:
    """A size as printed (``text``), its members as the plain numbers printed
    (``printed``) and, when a unit is printed on any member, each member's
    length in metres (``metres``; ``None`` for a size with no unit)."""

    text: str
    printed: tuple[float, ...]
    metres: tuple[float, ...] | None

    def same(self, other: Size) -> bool:
        """Two sizes with a unit are one size when their lengths in metres
        are equal (``1.8 x 1.8 m`` is ``1800 x 1800 mm``). A size with no
        unit is the plain number it prints: it is the same as another size
        only when the printed numbers are equal, never at another magnitude
        (a bare ``1.8`` is not ``18``, ``180`` or ``1800``)."""
        if self.metres is not None and other.metres is not None:
            return self.metres == other.metres
        return self.printed == other.printed


def sizes_in(text: str) -> list[Size]:
    """Every size printed in ``text``.

    A member with no unit takes the unit printed on the next member, else on
    the one before (``1.8 x 1.8 m`` is metres throughout); a size with no unit
    on any member borrows none: it keeps only its printed numbers."""
    out: list[Size] = []
    for m in _SIZE_RE.finditer((text or "").translate(_DIGITS)):
        members = [(_number(n), (u or "").casefold()) for n, u in _MEMBER_RE.findall(m.group(0))]
        units = [u for _, u in members]
        metres: tuple[float, ...] | None = None
        if any(units):
            filled = []
            for i, u in enumerate(units):
                after = next((x for x in units[i:] if x), "")
                filled.append(_LENGTH_UNITS[u or after or next(x for x in reversed(units[:i]) if x)])
            metres = tuple(round(v * f, 9) for (v, _), f in zip(members, filled))
        out.append(Size(" ".join(m.group(0).split()), tuple(v for v, _ in members), metres))
    return out


def _without_sizes(text: str) -> str:
    return _SIZE_RE.sub(" ", (text or "").translate(_DIGITS))


def _words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD_RE.findall(text or "")}


def _terms(terms: Iterable[str] | str | None) -> set[str]:
    if isinstance(terms, str):
        return _words(terms)
    return set().union(*(_words(t) for t in terms)) if terms else set()


def _forms(word: str) -> set[str]:
    """A word and its singular spellings: ``pipes`` is also ``pipe``,
    ``trenches`` is also ``trench``. Two words agree when their forms meet."""
    out = {word}
    # A word ending in a double s (glass, boss) is not a plural of a word
    # ending in one s; it folds only through ``-es`` (glasses -> glass).
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        out.add(word[:-1])
        if word.endswith("es"):
            out.add(word[:-2])
    # An Arabic word and the same word without its article (``الفخار`` is
    # also ``فخار``, ``بالفخار`` too) or a one-letter prefix (``وفخار``), with
    # its letter variants made one. Forms are only ADDED (the word itself
    # stays), so a word that merely starts with a prefix letter still
    # matches itself.
    stems = {word}
    if word[:1] in _AR_PREFIX_LETTERS and len(word) - 1 >= 3:
        stems.add(word[1:])
    out |= {_bare(s) for s in stems}
    return out


def _contains(words: set[str], terms: set[str], fold: bool) -> bool:
    if not fold:
        return terms <= words
    have = set().union(*(_forms(w) for w in words)) if words else set()
    return all(_forms(t) & have for t in terms)


def matches(row: SheetRow, terms: set[str], section_terms: set[str], *, fold: bool = False,
            sizes: Sequence[Size] = ()) -> bool:
    """Every term is a word of the description, every section term a word of
    the heading path, and every size is printed (as a size, in any unit) in
    the heading path or the description."""
    if terms and not _contains(_words(_without_sizes(row.description)), terms, fold):
        return False
    if sizes:
        printed = sizes_in(" ".join(row.path)) + sizes_in(row.description)
        if not all(any(want.same(s) for s in printed) for want in sizes):
            return False
    return not section_terms or _contains(_words(_without_sizes(" ".join(row.path))), section_terms, fold)


#: Words of an ask that never pick rows: function words, the words of asking,
#: and the words that say WHAT is counted or summed rather than WHICH rows
#: (a total length is a sum of quantities; it does not select rows that print
#: the word "length"). General English; nothing here is a bill's word.
_ASK_STOPWORD_TEXT = """
a an the of to in on at by for from with within into onto under over and or nor not no
is are was were be been being am do does did done have has had having will would shall
should can could may might must there here this that these those it its it's i me my
we our us you your they their them what which who whom whose how when where why whether
all any each every both either some only just also please give tell show list find
calculate compute work out get let make need want know many much more most less least
total totals sum sums summed count counts counted number numbers figure figures overall
altogether together combined grand quantity quantities amount amounts value values
cost costs price prices priced money length lengths area areas volume volumes weight
weights item items row rows line lines entry entries bill bills boq boqs schedule
document documents section sections part parts heading headings page pages per
"""
#: The same for an Arabic ask (written without the article: :func:`_bare`
#: removes it from the ask's words before they are looked up).
_ASK_STOPWORD_TEXT_AR = """
ما ماذا كم كام هل هو هي هم في من على الى إلى عن مع او أو و ثم كل جميع كامل
مجموع اجمالي إجمالي جمله جملة عدد بند بنود صف صفوف سطر اسطر احسب أحسب اعطني أعطني
اعطيني أعطيني اذكر بين هذا هذه ذلك تلك التي الذي الذين لي لنا يوجد توجد موجود قسم
اقسام أقسام جدول كميات مستند وثيقه وثيقة صفحه صفحة
"""

_AR_ARTICLE = ("وال", "بال", "فال", "كال", "لل", "ال")
#: One-letter Arabic prefixes (and, with, for, so, like) a word can carry.
_AR_PREFIX_LETTERS = frozenset("وبلفك")
_AR_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})


def _bare(word: str) -> str:
    """An Arabic word without its article and with its letter variants made
    one (``التكلفة`` -> ``تكلفه``); any other word as it is. Used only to look
    an ask's word up in the small tables below, never to match the bill."""
    word = word.casefold().translate(_AR_LETTERS)
    for article in _AR_ARTICLE:
        if word.startswith(article) and len(word) - len(article) >= 2:
            return word[len(article):]
    return word


def _table(*texts: str) -> frozenset[str]:
    return frozenset(_bare(w) for text in texts for w in text.split())


#: Ask words that mean the money column; any of them makes the measure amount.
_AMOUNT_WORDS = _table(
    "amount amounts cost costs value values price prices priced money",
    "مبلغ مبالغ تكلفة تكلفه تكاليف كلفة كلفه سعر أسعار اسعار قيمة قيمه قيم ثمن",
)
#: Ask words that mean the measured column.
_QUANTITY_WORDS = _table(
    "quantity quantities length lengths area areas volume volumes weight weights",
    "كمية كميه كميات طول أطوال اطوال مساحة مساحه مساحات حجم أحجام احجام وزن أوزان اوزان",
)
_ASK_STOPWORDS = _table(_ASK_STOPWORD_TEXT, _ASK_STOPWORD_TEXT_AR) | _AMOUNT_WORDS | _QUANTITY_WORDS


def measure_from_ask(ask: str | None, default: str | None = None) -> str | None:
    """The column an ask totals: money words choose the amount, measure words
    the quantity (English or Arabic). An ask with neither returns ``default``
    (``None``: the column cannot be read from the ask; the caller totals the
    quantity and says so plainly, so a quantity is never passed off as a
    cost)."""
    words = {_bare(w) for w in _words(ask or "")}
    if words & _AMOUNT_WORDS:
        return MEASURE_AMOUNT
    if words & _QUANTITY_WORDS:
        return MEASURE_QUANTITY
    return default


def selection_from_ask(rows: Sequence[SheetRow], ask: str | None) -> dict[str, list[str]]:
    """Which words of the user's ask select rows of this bill, and how.

    The rule, applied the same way to every ask and every bill:

    * the ask's words, less :data:`_ASK_STOPWORDS`;
    * a word the bill prints in a heading path is a section word (the bill's
      own categories), else a word it prints in a description is a
      description word, else it selects nothing and is reported as ignored;
    * a plural and its singular agree (:func:`_forms`);
    * a number (with its unit, or a dimension ``a x b``) is a SIZE
      (:func:`sizes_in`): a size with a unit is read to metres on the ask
      and on the bill alike, so every spelling of it with a unit selects the
      same rows; a size with no unit is the plain number it prints, never
      read at another magnitude; a size the bill prints nowhere selects
      nothing and is reported as ignored.

    The result depends only on the ask and the bill, never on the model.
    """
    path_words: set[str] = set()
    desc_words: set[str] = set()
    printed: list[Size] = []
    for row in rows:
        path_words |= _words(_without_sizes(" ".join(row.path)))
        desc_words |= _words(_without_sizes(row.description))
        printed += sizes_in(" ".join(row.path)) + sizes_in(row.description)
    path_forms = set().union(*(_forms(w) for w in path_words)) if path_words else set()
    desc_forms = set().union(*(_forms(w) for w in desc_words)) if desc_words else set()
    terms: list[str] = []
    section: list[str] = []
    ignored: list[str] = []
    sizes: list[str] = []
    for size in sizes_in(ask or ""):
        bucket = sizes if any(size.same(p) for p in printed) else ignored
        if size.text not in bucket:
            bucket.append(size.text)
    for word in _WORD_RE.findall(_without_sizes(ask or "")):
        word = word.casefold()
        if _bare(word) in _ASK_STOPWORDS or word in terms or word in section or word in ignored:
            continue
        if _forms(word) & path_forms:
            section.append(word)
        elif _forms(word) & desc_forms:
            terms.append(word)
        else:
            ignored.append(word)
    return {"terms": sorted(terms), "section_terms": sorted(section), "sizes": sizes,
            "ignored": sorted(ignored)}


def aggregate(
    rows: Sequence[SheetRow],
    *,
    terms: Iterable[str] | str | None,
    section: Iterable[str] | str | None = None,
    measure: str = MEASURE_QUANTITY,
    fold: bool = False,
    sizes: Iterable[str] = (),
) -> dict[str, Any]:
    """Count and sum the items matching ``terms`` (and ``section``).

    ``fold`` lets a plural and its singular agree (the ask-derived selection).
    ``sizes`` are sizes as written (``1.8 x 1.8 m``); a row matches only when
    it prints each of them, in any spelling (:func:`sizes_in`).

    Quantities are summed per unit: adding cubic metres to metres is not a
    total, so a mixed-unit match returns one sum per unit and no single sum.
    A matching item without the figure is counted, listed, and named in
    ``without_figure``; it adds nothing to a sum.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    # A size in the call's own terms is matched as a size too, never as words.
    texts = [terms] if isinstance(terms, str) else list(terms or [])
    want, where = _terms([_without_sizes(t) for t in texts]), _terms(section)
    asked = [s for text in [*sizes, *texts] for s in sizes_in(text)]
    if not want and not where and not asked:
        raise ValueError("give description terms or a section to match")
    items = join_page_continuations(rows)
    used, without = [], []
    sums: dict[str, float] = {}
    # One entry per section (heading path) the matched rows sit in, in the
    # order the bill prints them: an ask whose words match rows of two
    # sections is told what each section adds, not only the total of all.
    sections: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in items:
        if not matches(row, want, where, fold=fold, sizes=asked):
            continue
        value = row.figure(measure)
        entry = {
            "ref": row.ref,
            "pages": row.pages,
            "section": list(row.path),
            "description": row.description,
            "unit": row.unit,
            measure: value,
        }
        used.append(entry)
        part = sections.setdefault(tuple(row.path), {"section": list(row.path), "count": 0, "sums": {}})
        part["count"] += 1
        if value is None:
            without.append({"ref": row.ref, "pages": row.pages, "description": row.description})
            continue
        key = row.unit if measure == MEASURE_QUANTITY else ""
        sums[key] = sums.get(key, 0.0) + value
        part["sums"][key] = part["sums"].get(key, 0.0) + value
    single = len(sums) <= 1
    return {
        "measure": measure,
        "terms": sorted(want),
        "section_terms": sorted(where),
        "sizes": [s.text for s in asked],
        "rows_scanned": len(items),
        "count": len(used),
        "sum": (next(iter(sums.values()), 0.0) if single else None),
        "unit": (next(iter(sums), "") if single and measure == MEASURE_QUANTITY else None),
        "sum_by_unit": ({u: s for u, s in sums.items()} if measure == MEASURE_QUANTITY else None),
        "without_figure": without,
        "rows": used,
        "by_section": _by_section(sections.values(), measure),
    }


def _by_section(parts: Iterable[dict[str, Any]], measure: str) -> dict[str, dict[str, Any]]:
    """Each section's total keyed by its heading path, in the order the bill
    prints them. An object, not a list of objects: a list of objects is a
    list of ROWS to the result's size cap (``runtime._row_lists``), which
    drops it from the never-cut outline; a section total is a figure."""
    out: dict[str, dict[str, Any]] = {}
    for part in parts:
        key = base = _section_name(part["section"])
        n = 2
        while key in out:  # two paths printed alike stay two sections
            key, n = f"{base} ({n})", n + 1
        out[key] = _section_total(part, measure)
    return out


def _section_total(part: dict[str, Any], measure: str) -> dict[str, Any]:
    sums = part["sums"]
    single = len(sums) <= 1
    return {
        "section": part["section"],
        "count": part["count"],
        "sum": (next(iter(sums.values()), 0.0) if single else None),
        "unit": (next(iter(sums), "") if single and measure == MEASURE_QUANTITY else None),
        "sum_by_unit": (dict(sums) if measure == MEASURE_QUANTITY else None),
    }


def _section_name(section: Sequence[str]) -> str:
    return " > ".join(t for t in section if t) or "(no heading)"


def _total_text(part: dict[str, Any], measure: str) -> str:
    if part.get("sum") is not None:
        unit = f" {part['unit']}" if part.get("unit") else ""
        return f"{_fmt(part['sum'])}{unit}"
    return ", ".join(f"{_fmt(v)} {u}".rstrip() for u, v in (part.get("sum_by_unit") or {}).items())


NO_DERIVED_SHEET = "no_derived_bill_sheet"
NO_READABLE_BYTES = "derived_sheet_has_no_readable_bytes"
UNREADABLE_SHEET = "derived_sheet_is_not_a_readable_workbook"
#: Every reason a bill (or the whole call) did not return; the tool declares
#: these, and they are the only error words of it a user is shown.
REASON_CODES = (NO_DERIVED_SHEET, NO_READABLE_BYTES, UNREADABLE_SHEET)


def _unreadable_sheet_errors() -> tuple[type[BaseException], ...]:
    """What reading a stored sheet raises when its bytes are not a workbook
    this key can open: not a zip (or a cut one), not an xlsx inside, or
    encrypted under another key. That is the bill's error, not the turn's."""
    import zipfile

    from openpyxl.utils.exceptions import InvalidFileException

    from app.core import file_crypto

    return (zipfile.BadZipFile, InvalidFileException, file_crypto.DecryptionError)


def derived_bill_documents(project_id: str, document_id: str | None = None) -> list[dict[str, Any]]:
    """The visible derived bill sheets of the project; with ``document_id``,
    only the one derived from (or being) that document."""
    from app.core import bill_derived, projects

    out = []
    for doc in projects.list_documents(project_id):
        if doc.get("retrieval_visible") is False:
            continue
        meta = projects.coerce_document_metadata(doc.get("metadata"))
        if meta.get("derivation") != bill_derived.DERIVATION:
            continue
        if document_id and document_id not in (doc.get("id"), meta.get("derived_from")):
            continue
        out.append(doc)
    return out


def _fmt(value: float) -> str:
    """``3000`` -> ``3,000``; ``103.5`` -> ``103.5``; at most three decimals."""
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _row_label(entry: dict[str, Any]) -> str:
    pages = entry.get("pages") or []
    where = ("p. " + "-".join(str(p) for p in (pages[0], pages[-1]) if p is not None)
             if len(pages) > 1 else (f"p. {pages[0]}" if pages else ""))
    name = entry.get("ref") or " ".join((entry.get("description") or "").split()[:6])
    return f"{name} ({where})" if where else name


#: How many rows a statement names before it states how many more there are.
#: A statement is never cut downstream (the result outline keeps it whole),
#: so its length must not grow with the bill: the rows themselves stay in
#: the result's ``rows``.
STATEMENT_ROWS_NAMED = 12


def _labels(entries: Sequence[dict[str, Any]]) -> str:
    """The labels of ``entries``, the first STATEMENT_ROWS_NAMED of them by
    name and the rest as a count, so the text is bounded for any bill."""
    named = "; ".join(_row_label(e) for e in entries[:STATEMENT_ROWS_NAMED])
    more = len(entries) - STATEMENT_ROWS_NAMED
    return f"{named}; and {more} more row(s), listed under rows" if more > 0 else named


def _quoted(words: Iterable[str]) -> str:
    return ", ".join(f'"{w}"' for w in words)


def statement(result: dict[str, Any]) -> str:
    """One sentence a user reads: which rows were counted and why, and the
    figure, built from ``result`` alone so it is the tool's own figure."""
    measure = result.get("measure") or MEASURE_QUANTITY
    why = []
    if result.get("terms"):
        why.append(f"whose description contains {_quoted(result['terms'])}")
    if result.get("section_terms"):
        why.append(f"in the section {_quoted(result['section_terms'])}")
    if result.get("sizes"):
        why.append(f"of size {_quoted(result['sizes'])}")
    ignored = result.get("ignored_ask_words") or []
    unused = (f" Words of the question found nowhere in this bill, so not used: {_quoted(ignored)}."
              if ignored else "")
    if result.get("selected_by") == "none":
        return ("No row was selected from the question: none of its words occurs in this "
                "bill's headings or descriptions. Ask again naming the items as the bill "
                f"names them.{unused}")
    rows = result.get("rows") or []
    head = f"{result.get('count', 0)} row(s) {' '.join(why)}".rstrip()
    if not rows:
        return f"{head}: no row of this bill matches.{unused}"
    listed = _labels(rows)
    if result.get("sum") is not None:
        unit = f" {result['unit']}" if result.get("unit") else ""
        figure = f"total {measure} {_fmt(result['sum'])}{unit}"
    else:
        figure = f"total {measure} by unit: " + ", ".join(
            f"{_fmt(v)} {u}".rstrip() for u, v in (result.get("sum_by_unit") or {}).items())
    text = f"{head}: {listed}. {figure[0].upper()}{figure[1:]}."
    parts = list((result.get("by_section") or {}).values())
    if len(parts) > 1:
        text += f" The rows sit in {len(parts)} sections: " + "; ".join(
            f"{_section_name(p['section'])}: {p['count']} row(s), total {measure} "
            f"{_total_text(p, measure)}" for p in parts) + (
            f"; all {len(parts)} sections together: {_total_text(result, measure)}.")
    if result.get("measure_unread"):
        text += (" The question names no column, so this is the total quantity, "
                 "not the amount (cost).")
    without = result.get("without_figure") or []
    if without:
        text += f" {len(without)} of them print no {measure} and add nothing: " + _labels(
            without) + "."
    return text + unused


def stated_figures(result: dict[str, Any]) -> dict[str, Any]:
    """The figures a bill result states as its total, in the shape the
    answer's exit check reads (``app.agents.answer_exit``): the answer must
    carry each of them. ``values`` and ``shown`` are parallel lists of plain
    values (never a list of objects, which the result's size cap would read
    as rows and drop from the never-cut outline)."""
    if result.get("selected_by") == "none" or not result.get("rows"):
        pairs: list[tuple[str, Any]] = []
    elif result.get("sum") is not None:
        pairs = [(result.get("unit") or "", result["sum"])]
    else:
        pairs = list((result.get("sum_by_unit") or {}).items())
    return {"values": [float(v) for _, v in pairs],
            "shown": [f"{_fmt(v)} {u}".rstrip() for u, v in pairs],
            "measure": result.get("measure")}


def aggregate_project_bills(
    project_id: str,
    *,
    terms: Iterable[str] | str | None,
    section: Iterable[str] | str | None = None,
    measure: str = MEASURE_QUANTITY,
    document_id: str | None = None,
    ask: str | None = None,
) -> dict[str, Any]:
    """:func:`aggregate` over every row of each derived bill sheet in scope,
    one result per bill (two bills are never added together).

    With ``ask`` (the user's own message, even an empty one) each bill's rows
    are selected by :func:`selection_from_ask` and the measure by
    :func:`measure_from_ask`, and ``terms``/``section``/``measure`` are NOT
    used: an ask that selects nothing in a bill says so (``selected_by``
    ``none``) and selects no row; the model's words never stand in for the
    user's. Only a caller with no ask at all (``ask=None``: not a chat turn)
    selects by ``terms``/``section`` (``selected_by`` ``call``)."""
    from app.core import projects

    docs = derived_bill_documents(project_id, document_id)
    if not docs:
        out = {"status": "error", "reason": NO_DERIVED_SHEET}
        _LOG.info("%s", selection_log_line(out, terms=terms, section=section, measure=measure))
        return out
    if measure not in MEASURES:
        # The call's column is checked even when the ask decides the column.
        raise ValueError(f"measure must be one of {MEASURES}")
    bills = []
    for doc in docs:
        path = projects.materialize_document_path(doc)
        meta = projects.coerce_document_metadata(doc.get("metadata"))
        head = {"document_id": doc.get("id"), "filename": doc.get("original_name"),
                "derived_from": meta.get("derived_from")}
        if not path:
            bills.append({**head, "status": "error", "reason": NO_READABLE_BYTES})
            continue
        try:
            rows = read_sheet_rows(path)
        except _unreadable_sheet_errors() as exc:
            _LOG.warning("derived bill sheet %s is not readable: %s", doc.get("id"), type(exc).__name__)
            bills.append({**head, "status": "error", "reason": UNREADABLE_SHEET})
            continue
        result = _aggregate_bill(rows, terms=terms, section=section, measure=measure, ask=ask)
        bills.append({**head, "status": "success", **result})
    out = {"status": "success", "bills": bills}
    _LOG.info("%s", selection_log_line(out, terms=terms, section=section, measure=measure))
    return out


def _call_texts(terms: Iterable[str] | str | None, section: Iterable[str] | str | None) -> list[str]:
    return [t for t in ([terms] if isinstance(terms, str) else list(terms or []))
            + ([section] if isinstance(section, str) else list(section or [])) if t]


def _named_forms(texts: Sequence[str]) -> set[str]:
    return set().union(*(_forms(w) for w in _terms([_without_sizes(t) for t in texts]))) if texts else set()


def _absent_part_of_ask(
    picked: dict[str, list[str]],
    terms: Iterable[str] | str | None,
    section: Iterable[str] | str | None,
) -> list[str]:
    """The user's words a call names when the call names ONLY words of the
    ask that this bill holds nowhere (``picked['ignored']``) and none of the
    words that select rows: that call asks about the absent part of a
    two-part question ('kerbs, and X?'), so it is answered as that part --
    no row -- not with the other part's rows. The call never adds a word of
    its own: only the ask's ignored words it names are returned."""
    # A call that names no word meets no word below: it returns [] (the
    # call's selection, not the absent part, is then read).
    named = _named_forms(_call_texts(terms, section))
    selecting = picked["terms"] + picked["section_terms"]
    if any(_forms(w) & named for w in selecting):
        return []
    return [w for w in picked["ignored"] if _forms(w) & named]


def _part_of_ask(
    picked: dict[str, list[str]],
    terms: Iterable[str] | str | None,
    section: Iterable[str] | str | None,
) -> dict[str, list[str]] | None:
    """The part of the ask's selection that a call names, for an ask whose
    words all together select no row (an ask of two parts: 'kerbs and
    ducts'). Only the USER's selecting words are kept, those the call's terms
    also name; the call never adds a word of its own. ``None`` when the call
    names none of them or all of them (nothing narrower to read)."""
    texts = _call_texts(terms, section)
    named = _named_forms(texts)
    called = [s for t in texts for s in sizes_in(t)]
    part = {
        "terms": [w for w in picked["terms"] if _forms(w) & named],
        "section_terms": [w for w in picked["section_terms"] if _forms(w) & named],
        "sizes": [t for t in picked["sizes"] if any(s.same(c) for s in sizes_in(t) for c in called)],
    }
    kept = sum(len(v) for v in part.values())
    whole = sum(len(picked[k]) for k in part)
    return part if 0 < kept < whole else None


def _aggregate_bill(
    rows: Sequence[SheetRow],
    *,
    terms: Iterable[str] | str | None,
    section: Iterable[str] | str | None,
    measure: str,
    ask: str | None,
) -> dict[str, Any]:
    if ask is None:
        result = aggregate(rows, terms=terms, section=section, measure=measure)
        result["selected_by"] = "call"
    else:
        picked = selection_from_ask(rows, ask)
        selects = bool(picked["terms"] or picked["section_terms"] or picked["sizes"])
        absent = _absent_part_of_ask(picked, terms, section) if selects else []
        if absent:
            read = measure_from_ask(ask)
            result = aggregate(rows, terms=absent, section=None,
                               measure=read or MEASURE_QUANTITY, fold=True)
            result["selected_by"] = "ask_absent"
            result["measure_unread"] = read is None
        elif selects:
            read = measure_from_ask(ask)
            result = aggregate(rows, terms=picked["terms"], section=picked["section_terms"],
                               measure=read or MEASURE_QUANTITY, fold=True, sizes=picked["sizes"])
            result["selected_by"] = "ask"
            part = _part_of_ask(picked, terms, section) if not result["rows"] else None
            if part is not None:
                result = aggregate(rows, terms=part["terms"], section=part["section_terms"],
                                   measure=read or MEASURE_QUANTITY, fold=True, sizes=part["sizes"])
                result["selected_by"] = "ask_part"
            # No column named: the quantity, and the statement says so.
            result["measure_unread"] = read is None
        else:
            result = {"measure": None, "terms": [], "section_terms": [], "sizes": [],
                      "rows_scanned": len(join_page_continuations(rows)), "count": 0, "sum": None,
                      "unit": None, "sum_by_unit": None, "without_figure": [], "rows": [],
                      "selected_by": "none"}
        result["ignored_ask_words"] = picked["ignored"]
    result["statement"] = statement(result)
    result["stated_figures"] = stated_figures(result)
    return result


def _digest(items: Any) -> str:
    """A short, stable fingerprint of ``items``: equal inputs give equal
    digests, so two calls can be compared from the log without the log
    holding any of the words or rows themselves."""
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _set_field(words: set[str]) -> str:
    return f"{len(words)}:{_digest(sorted(words)) if words else '-'}"


def selection_log_line(
    result: dict[str, Any],
    *,
    terms: Iterable[str] | str | None,
    section: Iterable[str] | str | None,
    measure: str,
) -> str:
    """One line naming which term set and which rows a call selected.

    The term sets are logged as size and digest of the casefolded word set
    (the matcher's own view of them, so word order and case do not change
    it); each bill as its status, rows scanned, rows counted and a digest of
    the selected rows (reference, pages and heading path). Two asks that
    selected different rows show different row digests; whether they also
    show different term digests says whether the term choice was the cause.
    No description text and no figure is written.
    """
    parts = [
        "bill_rows_aggregate",
        f"measure={measure}",
        f"terms={_set_field(_terms(terms))}",
        f"section={_set_field(_terms(section))}",
        f"status={result.get('status')}",
    ]
    if result.get("reason"):
        parts.append(f"reason={result['reason']}")
    bills = result.get("bills") or []
    parts.append(f"bills={len(bills)}")
    for bill in bills:
        if bill.get("status") != "success":
            parts.append(f"[{bill.get('document_id')} {bill.get('status')} reason={bill.get('reason')}]")
            continue
        selected = [[r.get("ref"), r.get("pages"), r.get("section")] for r in bill.get("rows") or []]
        parts.append(
            f"[{bill.get('document_id')} success by={bill.get('selected_by', 'call')} "
            f"used_terms={_set_field(set(bill.get('terms') or []))} "
            f"used_section={_set_field(set(bill.get('section_terms') or []))} "
            f"scanned={bill.get('rows_scanned')} "
            f"count={bill.get('count')} without_figure={len(bill.get('without_figure') or [])} "
            f"rows={_digest(selected) if selected else '-'}]"
        )
    return " ".join(parts)
