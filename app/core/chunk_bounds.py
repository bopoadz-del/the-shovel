"""A floor and a ceiling on every chunk the indexer writes (RQ-2).

The embedder sees at most a 512-token window (the cross-encoder reranker and
the BERT-family tokenizer the static embedder was distilled from both stop
there). At about four characters per English token that window is ~2,000
characters, so a longer chunk is embedded on its head only and its tail is
invisible to vector search. A very short chunk (a lone heading, one cell)
carries too little text to embed into a useful neighbourhood.

* **ceiling** -- a chunk longer than ``CHUNK_MAX_CHARS`` (default 2000) is
  split at the coarsest boundary that fits: blank line, then line (a
  spreadsheet or bill row), then sentence end, then whitespace. A hard cut
  is the last resort for a single unbroken run.
* **floor** -- a prose chunk shorter than ``CHUNK_MIN_CHARS`` (default 200)
  is joined to its next neighbour (the previous one for the last chunk) when
  the joined text still fits under the ceiling. A short chunk that cannot be
  joined without breaking the ceiling is kept as is: text is never dropped.
  Unit chunks -- one bill or spreadsheet row, one Contract Data particulars
  block, one drawing schedule -- are a deliberate unit of retrieval and get
  the ceiling only (``merge=False``): joining two rows would blur which row
  states the figure.

* **heading line** -- every piece of a split chunk starts with the heading
  line of the page it sits on (a drawing sheet: its title-block number and
  title; a text page: the page's first heading-like line), so a piece is
  findable by what it belongs to. A piece keeps the page it is located on,
  or its parent's page when it cannot be located on its own. Room for the
  heading (``HEADING_RESERVE``) is kept free when the split is cut, so a
  piece with its heading still fits under the ceiling.

Both values are read from the environment at call time; ``0`` turns the
rule off. Nothing here reads a document name or id: the rules look only at
the chunk text's own length and boundaries.
"""
from __future__ import annotations

import calendar
import itertools
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_CHARS = 2000
#: Characters kept free in each split piece for its page's heading line.
HEADING_RESERVE = 120

# One id per split parent, unique for the process: pieces of one parent share it.
_SPLIT_PARENT_IDS = itertools.count(1)


def new_split_parent() -> int:
    """A fresh split-parent id, for a caller whose own splitter cut the pieces."""
    return next(_SPLIT_PARENT_IDS)

# Coarsest boundary first. Each pattern matches the separator between pieces.
_BOUNDARIES = (
    re.compile(r"\n\s*\n"),          # paragraph
    re.compile(r"\n"),               # line / table row
    re.compile(r"(?<=[.!?;])\s+"),   # sentence end
    re.compile(r"\s+"),              # word
)


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, "").strip()))
    except ValueError:
        return default


def chunk_max_chars() -> int:
    """``CHUNK_MAX_CHARS`` (default 2000; 0 = no ceiling)."""
    return _env_int("CHUNK_MAX_CHARS", DEFAULT_MAX_CHARS)


def chunk_min_chars() -> int:
    """``CHUNK_MIN_CHARS`` (default 200; 0 = no floor)."""
    return _env_int("CHUNK_MIN_CHARS", DEFAULT_MIN_CHARS)


def _pack(pieces: list[str], sep: str, ceiling: int) -> list[str]:
    """Greedy: join consecutive pieces with ``sep`` while they fit."""
    out: list[str] = []
    cur = ""
    for piece in pieces:
        if not piece:
            continue
        joined = f"{cur}{sep}{piece}" if cur else piece
        if cur and len(joined) > ceiling:
            out.append(cur)
            cur = piece
        else:
            cur = joined
    if cur:
        out.append(cur)
    return out


def split_oversize(text: str, ceiling: int, _level: int = 0) -> list[str]:
    """``text`` cut into pieces of at most ``ceiling`` chars at natural boundaries."""
    if ceiling <= 0 or len(text) <= ceiling:
        return [text]
    if _level >= len(_BOUNDARIES):
        return [text[i:i + ceiling] for i in range(0, len(text), ceiling)]
    pattern = _BOUNDARIES[_level]
    sep = "\n\n" if _level == 0 else ("\n" if _level == 1 else " ")
    pieces = [p.strip() for p in pattern.split(text) if p and p.strip()]
    if len(pieces) <= 1:
        return split_oversize(text, ceiling, _level + 1)
    out: list[str] = []
    for packed in _pack(pieces, sep, ceiling):
        out.extend(split_oversize(packed, ceiling, _level + 1))
    return out


def _merge_pairs(
    pairs: list[tuple[str, int | None]], floor: int, ceiling: int,
    piece_ceiling: int | None = None,
) -> list[tuple[str, int | None]]:
    """Join each pair shorter than ``floor`` to a neighbour when it still fits.

    Pairs are ``(text, split_parent)``; a joined pair keeps a parent id.
    ``piece_ceiling`` (optional) bounds a join that keeps a parent id: that
    join is a split piece, which is still given its heading line, so it may
    not refill the room the cut kept free for that line.
    """
    def _limit(parent: int | None) -> int:
        if parent is not None and piece_ceiling is not None:
            return piece_ceiling
        return ceiling

    out: list[tuple[str, int | None]] = []
    carry: tuple[str, int | None] | None = None
    for chunk, parent in pairs:
        if carry is not None:
            joined = f"{carry[0]}\n{chunk}"
            kept = parent if parent is not None else carry[1]
            limit = _limit(kept)
            if limit <= 0 or len(joined) <= limit:
                chunk, parent = joined, kept
            else:
                out.append(carry)
            carry = None
        if len(chunk.strip()) < floor:
            carry = (chunk, parent)
        else:
            out.append((chunk, parent))
    if carry is not None:
        kept = carry[1]
        if out and out[-1][1] is not None:
            kept = out[-1][1]
        limit = _limit(kept)
        if out and (limit <= 0 or len(out[-1][0]) + 1 + len(carry[0]) <= limit):
            last, last_parent = out[-1]
            out[-1] = (
                f"{last}\n{carry[0]}",
                last_parent if last_parent is not None else carry[1],
            )
        else:
            out.append(carry)
    return out


def bound_chunks(
    chunks: Iterable[str],
    stats: dict[str, Any] | None = None,
    *,
    merge: bool = True,
    split_parents: list[int | None] | None = None,
) -> list[str]:
    """Every chunk under ``CHUNK_MAX_CHARS``; with ``merge``, over the floor too.

    The ceiling is applied first so a merge never rebuilds an oversize chunk.
    ``merge=False`` is for unit chunks (rows, particulars, schedules).
    ``stats`` (optional) receives ``split_from`` / ``merged`` counts.
    ``split_parents`` (optional) is extended with one entry per returned
    chunk: the id its split parent's pieces share, or None for an unsplit
    chunk. A caller that passes it puts a heading line on each piece
    (``attach_page_headings``), so the cut keeps ``HEADING_RESERVE`` free.
    """
    ceiling = chunk_max_chars()
    floor = chunk_min_chars() if merge else 0
    if ceiling and floor >= ceiling:
        floor = 0  # a floor at or above the ceiling cannot be honoured
    cut = ceiling
    if split_parents is not None and ceiling > 2 * HEADING_RESERVE:
        cut = ceiling - HEADING_RESERVE
    src = [c for c in chunks if c is not None]
    split: list[tuple[str, int | None]] = []
    split_from = 0
    for chunk in src:
        parts = split_oversize(chunk, cut) if ceiling and len(chunk) > ceiling else [chunk]
        if len(parts) > 1:
            split_from += 1
            parent = next(_SPLIT_PARENT_IDS)
            split.extend((part, parent) for part in parts)
        else:
            split.append((chunk, None))
    # A join that stays a split piece keeps the heading room the cut kept.
    out = _merge_pairs(
        split, floor, ceiling, piece_ceiling=cut if cut < ceiling else None,
    )
    if stats is not None:
        stats["split_from"] = stats.get("split_from", 0) + split_from
        stats["merged"] = stats.get("merged", 0) + (len(split) - len(out))
    if split_parents is not None:
        split_parents.extend(parent for _, parent in out)
    return [c for c, _ in out]


def bound_units_with_pages(
    chunks: list[str], pages: list[int | None],
) -> tuple[list[str], list[int | None]]:
    """Unit chunks under the ceiling with ``pages`` kept one entry per chunk.

    No merge (rows, schedules). Every piece of a split chunk keeps the page of
    the chunk it came from. A ``pages`` list of the wrong length gives no
    usable attribution, so every returned chunk then gets None.
    """
    if len(pages) != len(chunks):
        pages = [None] * len(chunks)
    out_chunks: list[str] = []
    out_pages: list[int | None] = []
    for chunk, page in zip(chunks, pages):
        pieces = bound_chunks([chunk], merge=False)
        out_chunks.extend(pieces)
        out_pages.extend([page] * len(pieces))
    return out_chunks, out_pages


# A heading-like line: short, no closing sentence punctuation, and either
# numbered ("3.2 Concrete"), a markdown heading, or mostly capitals.
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*|[A-Z])[.)]?\s+\S")
_MAX_HEADING_WORDS = 14
# Month names and abbreviations from the standard calendar table, lowercased.
_MONTH_WORDS = frozenset(
    m.lower() for m in (*calendar.month_name, *calendar.month_abbr) if m
)
_DAY_OR_YEAR = re.compile(r"^\d{1,4}(?:st|nd|rd|th)?$", re.IGNORECASE)


def _is_date_line(text: str) -> bool:
    """True when every word of ``text`` is a day, a year or a month name.

    A letter or a set of minutes opens on its date ("12 March 2026"), and a
    day number in front of a word passes the numbered heading rule. A date
    names when the page was written, not what it is about.
    """
    words = [w.strip(".,") for w in text.split()]
    words = [w for w in words if w]
    months = sum(1 for w in words if w.lower() in _MONTH_WORDS)
    numbers = sum(1 for w in words if _DAY_OR_YEAR.match(w))
    return months >= 1 and numbers >= 1 and months + numbers == len(words)


# The cell separator the table extractors write (" | "). Only the pipe:
# ";" and tabs, which chunk_noise.cells also splits on, occur in headings.
_TABLE_CELL_SEPARATOR = "|"
# Two lines of two cells each are as often two titles holding the separator
# as a table: a run of equal lines is taken as a table only with more shape
# than that, at least this many columns or this many rows.
_TABLE_MIN_COLUMNS = 3
_TABLE_MIN_ROWS = 3


def _separator_count(line: str) -> int:
    """The cell separators of ``line``; a markdown ``#`` heading holds none."""
    if line.lstrip().startswith("#"):
        return 0
    return line.count(_TABLE_CELL_SEPARATOR)


# How many lines ahead of a run's first line its width is looked for.
_TABLE_WIDTH_LOOKAHEAD = 8


def _table_units(counts: list[int], start: int, width: int) -> list[list[int]]:
    """The rows of a table of ``width`` separators read from ``start`` on.

    A row is one line holding ``width`` separators, or a row whose cell
    wrapped: the writer joins the cells with the separator, so a cell's line
    break splits the row over consecutive lines that each hold fewer, and
    whose separators add up to ``width``. Such a row opens on a line holding
    a separator (a line holding none is no piece of a row that starts there).
    Reading stops at the first line that fits neither.
    """
    units: list[list[int]] = []
    pos = start
    while pos < len(counts):
        if counts[pos] == width:
            units.append([pos])
            pos += 1
            continue
        if not 0 < counts[pos] < width:
            break
        total, end = 0, pos
        while end < len(counts) and counts[end] < width and total < width:
            total += counts[end]
            end += 1
        if total != width:
            break
        units.append(list(range(pos, end)))
        pos = end
    return units


def _table_row_indexes(lines: list[str]) -> set[int]:
    """Indexes of the lines of ``lines`` that are rows of a table.

    The extractors write every row of one table with one separator per
    column gap, blank cells kept, so a one-cell section row carries as many
    separators as its header, and a row whose cell wrapped is split over
    lines whose separators add up to that count (``_table_units``). A table
    is a run of such rows, at least one of them on a line of its own, of at
    least ``_TABLE_MIN_COLUMNS`` columns or ``_TABLE_MIN_ROWS`` rows. A line
    that opens on the separator and holds two or more is a markdown table
    row on its own. A title holding the separator, alone or above a table it
    does not fit, is no row, and a markdown ``#`` heading never is.
    """
    filled = [i for i, line in enumerate(lines) if line.strip()]
    counts = [_separator_count(lines[i]) for i in filled]
    rows = {
        i for i, n in zip(filled, counts)
        if n >= 2 and lines[i].lstrip().startswith(_TABLE_CELL_SEPARATOR)
    }
    start = 0
    while start < len(filled):
        best: list[list[int]] = []
        for width in sorted(set(counts[start:start + _TABLE_WIDTH_LOOKAHEAD])):
            if not width:
                continue
            units = _table_units(counts, start, width)
            whole = any(len(unit) == 1 for unit in units)
            if whole and len(units) >= 2 and (
                width + 1 >= _TABLE_MIN_COLUMNS or len(units) >= _TABLE_MIN_ROWS
            ) and sum(map(len, units)) > sum(map(len, best)):
                best = units
        if not best:
            start += 1
            continue
        rows.update(filled[k] for unit in best for k in unit)
        start = best[-1][-1] + 1
    return rows


def _is_heading_line(line: str) -> bool:
    text = line.strip().lstrip("#").strip()
    if not text or len(text) >= HEADING_RESERVE or text[-1] in ".,;":
        return False
    if len(text.split()) > _MAX_HEADING_WORDS:
        return False
    if _is_date_line(text):
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 3:
        return False
    if line.lstrip().startswith("#") or _NUMBERED_HEADING.match(text):
        return True
    return sum(ch.isupper() for ch in letters) / len(letters) >= 0.6


def _folded(text: str) -> str:
    return " ".join(text.split())


def _table_cell_copies(lines: list[str], rows: set[int]) -> set[int]:
    """Indexes of the lines that repeat cells of the page's own table rows.

    A PDF page carries its text layer and, after it, its tables as rows; the
    text layer holds the same cells, one or a few to a line. A line is such
    a copy when its text, spacing folded, is one cell of a row on this page
    or a run of neighbouring cells of one row (the text layer writes cells
    that sit close together on one line), and the page holds that text off
    the rows no more often than in them: one more occurrence is a line of
    its own (a heading the table repeats), and keeps its place.
    """
    runs: Counter[str] = Counter()
    for idx in rows:
        row = [
            _folded(cell) for cell in lines[idx].split(_TABLE_CELL_SEPARATOR)
            if cell.strip()
        ]
        for start in range(len(row)):
            for end in range(start + 1, len(row) + 1):
                runs[" ".join(row[start:end])] += 1
    # A wrapped row's continuation line can hold no separator; it is a row,
    # not a copy, so only lines off the rows are counted and taken.
    loose = Counter(
        _folded(line) for idx, line in enumerate(lines)
        if idx not in rows and _folded(line) in runs
    )
    copies: set[int] = set()
    for idx, line in enumerate(lines):
        text = _folded(line)
        if idx not in rows and text in runs and loose[text] <= runs[text]:
            copies.add(idx)
    return copies


def heading_line_for_page(
    page_text: str,
    title_block: Callable[[str], dict[str, Any]] | None = None,
    running: Callable[[str], bool] | None = None,
) -> str:
    """The line a piece of this page is filed under ("" when there is none).

    A drawing sheet is named by its title block (number and title, read by
    ``title_block``); any other page by its first heading-like line that is
    not page furniture (``running``: a line the document repeats on most of
    its pages). A page whose only heading-like lines are furniture keeps the
    first of them.
    """
    if not page_text or not page_text.strip():
        return ""
    if title_block is not None:
        fields = title_block(page_text) or {}
        named = [
            str(fields[k]).strip() for k in ("drawing_number", "drawing_title")
            if fields.get(k) and str(fields[k]).strip()
        ]
        if named:
            return " - ".join(named)[: HEADING_RESERVE - 1]
    first = ""
    lines = page_text.splitlines()
    # A table row (a header or a numbered item row) names cells, not a section.
    rows = _table_row_indexes(lines)
    copies = _table_cell_copies(lines, rows)
    for idx, line in enumerate(lines):
        if idx in rows or idx in copies or not _is_heading_line(line):
            continue
        if running is None or not running(line):
            return line.strip()
        first = first or line.strip()
    return first


def _running_heading_lines(page_texts: dict[int, str]) -> Callable[[str], bool] | None:
    """The document's page furniture among its heading-like lines, or None.

    The same repetition rule the bill reader strips running headers by
    (``bill_table_geometry.running_lines``): a line that repeats, digits
    masked, on more than half of at least ``MIN_RUNNING_PAGES`` pages. A
    running header names the document, not the page, so it cannot tell one
    page's pieces from another's.
    """
    from app.core.bill_table_geometry import _Label, running_lines

    labels = [
        _Label(text=line.strip(), left=0.0, height=0.0, page=page)
        for page, body in page_texts.items()
        # Every heading-like line, table rows too: a table header repeated
        # page after page is furniture, and is only known as such when its
        # repeats are counted.
        for line in body.splitlines()
        if _is_heading_line(line)
    ]
    found = running_lines(labels, len(page_texts))
    return found.holds if found.masks else None


def page_heading_lines(
    text: str,
    page_starts: Iterable[Iterable[int]] | None,
    title_block: Callable[[str], dict[str, Any]] | None = None,
) -> dict[int, str]:
    """``{page: heading line}`` for every page of ``text`` that has one."""
    starts = sorted(
        ((int(p), int(o)) for p, o in (page_starts or [])), key=lambda s: s[1]
    )
    bodies: dict[int, str] = {}
    for idx, (page, offset) in enumerate(starts):
        end = starts[idx + 1][1] if idx + 1 < len(starts) else len(text)
        bodies[page] = text[offset:end]
    running = _running_heading_lines(bodies)
    out: dict[int, str] = {}
    for page, body in bodies.items():
        line = heading_line_for_page(body, title_block, running)
        if line:
            out[page] = line
    return out


def attach_page_headings(
    chunks: list[str],
    pages: list[int | None],
    split_parents: list[int | None],
    headings: dict[int, str],
) -> tuple[list[str], list[int | None]]:
    """Each piece of a split chunk: its page (own, else parent's) and that page's heading.

    The parent's page is the page its first located piece starts on (the
    first piece starts where the parent did). Unsplit chunks are returned
    unchanged, a piece that already starts with its heading is not given it
    twice, and a heading that would push a piece over the ceiling is left off.
    """
    if len(split_parents) != len(chunks) or len(pages) != len(chunks):
        return chunks, pages
    ceiling = chunk_max_chars()
    parent_page: dict[int, int] = {}
    for parent, page in zip(split_parents, pages):
        if parent is not None and page is not None:
            parent_page.setdefault(parent, page)
    out_chunks: list[str] = []
    out_pages: list[int | None] = []
    for chunk, page, parent in zip(chunks, pages, split_parents):
        if parent is not None:
            if page is None:
                page = parent_page.get(parent)
            heading = headings.get(page) if page is not None else None
            first = chunk.lstrip().split("\n", 1)[0].strip()
            headed = f"{heading}\n{chunk}" if heading and first != heading else chunk
            # The ceiling is a promise: a heading that does not fit is left off.
            if not ceiling or len(headed) <= ceiling:
                chunk = headed
        out_chunks.append(chunk)
        out_pages.append(page)
    return out_chunks, out_pages


def bounds_summary(lengths: Iterable[int]) -> dict[str, Any]:
    """Counts of chunk lengths against the configured bounds, plus p50/p95/max."""
    vals = sorted(int(n) for n in lengths)
    ceiling = chunk_max_chars()
    floor = chunk_min_chars()

    def _pct(q: float) -> int | None:
        if not vals:
            return None
        return vals[min(len(vals) - 1, int(q * (len(vals) - 1) + 0.5))]

    return {
        "chunks": len(vals),
        "min_chars": floor,
        "max_chars": ceiling,
        "over_ceiling": sum(1 for v in vals if ceiling and v > ceiling),
        "under_floor": sum(1 for v in vals if floor and v < floor),
        "p50": _pct(0.5),
        "p95": _pct(0.95),
        "max": vals[-1] if vals else None,
    }
