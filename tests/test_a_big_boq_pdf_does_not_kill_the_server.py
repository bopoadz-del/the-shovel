"""A big BOQ PDF must not take the web server down.

Killer #2, found 21 Sep 2026. Live 38bfa4e, 19:11 UAE, project master_corpus:
"What is the quantity, rate and amount for D529.2?" then "If the quantity
increased by 10%, what would the new amount be?" -> the agent called
boq_processor on the priced bill -> /livez stalled 40 s -> Render killed the
instance as unhealthy. The same class OOM-killed it at 17:01-17:13.

Measured on that real bill (28.2 MB, 370 pages), with the platform's own
BOQProcessorBlock, before this fix:

    15 s   16 pages    559 MB
    75 s   37 pages  1,938 MB   <- the 2 GB instance is dead here
   225 s  158 pages  5,786 MB
   335 s  227 pages  still parsing

Three causes, one per test group below:
  * pdfplumber keeps every parsed page (43 MB/page on that bill; 1 MB/page when
    each page is closed after use);
  * the parse ran synchronously inside `async def`, so the single worker's event
    loop -- health checks, every other user -- froze for the whole parse;
  * no budget: 370 pages x ~3 s/page is ~19 minutes, far past any chat turn.
"""
import asyncio
import os
import tempfile
import time

import pytest

pdfplumber = pytest.importorskip("pdfplumber")
pdfium = pytest.importorskip("pypdfium2")

from app.blocks import boq_processor as bp


@pytest.fixture
def blank_pdf():
    made = []

    def make(pages):
        doc = pdfium.PdfDocument.new()
        for _ in range(pages):
            doc.new_page(595, 842)
        fd, path = tempfile.mkstemp(suffix="_BOQ.pdf")
        os.close(fd)
        doc.save(path)
        made.append(path)
        return path

    yield make
    for p in made:
        try:
            os.remove(p)
        except OSError:
            pass


def _run(path):
    return asyncio.run(bp.BOQProcessorBlock().process({"file_path": path}, {}))


class _ChildLog:
    """What the parse did, readable from the test whether the parse ran in
    this process or in the isolated child (a forked child inherits these
    patched methods but not this process's lists, so events go to a file)."""

    def __init__(self, path):
        self.path = path
        path.write_text("", encoding="utf-8")

    def add(self, *event):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(" ".join(map(str, event)) + "\n")

    def events(self):
        return [tuple(int(x) if x.isdigit() else x for x in line.split())
                for line in self.path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def child_log(tmp_path):
    return _ChildLog(tmp_path / "parse_events.log")


# ── budget: a bill too big for a chat turn is refused up front ─────────────

def test_a_pdf_with_too_many_pages_is_refused_before_any_page_is_parsed(blank_pdf, monkeypatch, child_log):
    monkeypatch.setenv("BOQ_PDF_MAX_PAGES", "3")
    monkeypatch.setattr(pdfplumber.page.Page, "extract_tables",
                        lambda self, *a, **k: child_log.add("parsed") or [])
    out = _run(blank_pdf(5))
    assert out["status"] == "error"
    assert out.get("boq_pdf_too_many_pages") is True
    assert out["page_count"] == 5 and out["max_pages"] == 3
    assert child_log.events() == [], "no page may be parsed once the bill is over budget"
    msg = out["error"].lower()
    assert "5 pages" in msg and ("xlsx" in msg or "specific item" in msg)


def test_a_parse_that_runs_past_its_time_budget_stops_without_partial_totals(blank_pdf, monkeypatch, child_log):
    """The budget is driven by a fake clock, not by real elapsed time: each
    page "costs" 0.3 s on the parse clock and nothing sleeps. A wall-clock
    version of this test failed whenever the machine was loaded, because the
    PDF writes and opens around the parse were timed too.

    With a 0.5 s budget the clock reads 0.0 before page 1, 0.3 before page 2
    and 0.6 before page 3, so exactly two pages are parsed and page 3 is
    where the budget stops the parse.
    """
    monkeypatch.setenv("BOQ_PDF_PARSE_SECONDS", "0.5")
    now = [0.0]
    monkeypatch.setattr(bp, "_parse_clock", lambda: now[0])
    def slow(self, *a, **k):
        child_log.add("parsed")
        now[0] += 0.3
        return []

    monkeypatch.setattr(pdfplumber.page.Page, "extract_tables", slow)
    out = _run(blank_pdf(8))
    assert len(child_log.events()) == 2, "the budget must actually stop the parse"
    assert out["status"] == "error" and out.get("boq_pdf_parse_timeout") is True
    assert "stopped at page 3" in out["error"]
    assert "total_cost" not in out and "line_items" not in out, "never ship partial totals"


def test_a_small_pdf_still_parses_normally(blank_pdf):
    out = _run(blank_pdf(2))
    # Blank pages hold no tables and no text: the existing "nothing extractable"
    # answer, not a budget refusal.
    assert out["status"] == "error"
    assert not out.get("boq_pdf_too_many_pages") and not out.get("boq_pdf_parse_timeout")


# ── memory: every page is released after it is read ─────────────────────────

def test_every_page_is_closed_before_the_next_page_is_read(blank_pdf, monkeypatch, child_log):
    # pdfplumber closes cached pages when the document itself is closed, so
    # "closed eventually" proves nothing: the bill is dead by then. Each page
    # must be released before the next one is read, in both passes.
    real_close = pdfplumber.page.Page.close
    real_tables = pdfplumber.page.Page.extract_tables
    real_text = pdfplumber.page.Page.extract_text

    def close(self):
        child_log.add("close", self.page_number)
        return real_close(self)

    def tables(self, *a, **k):
        child_log.add("read", self.page_number)
        return real_tables(self, *a, **k)

    def text(self, *a, **k):
        child_log.add("read", self.page_number)
        return real_text(self, *a, **k)

    monkeypatch.setattr(pdfplumber.page.Page, "close", close)
    monkeypatch.setattr(pdfplumber.page.Page, "extract_tables", tables)
    monkeypatch.setattr(pdfplumber.page.Page, "extract_text", text)
    _run(blank_pdf(4))
    events = child_log.events()
    reads = [i for i, e in enumerate(events) if e[0] == "read"]
    assert len(reads) == 8, events  # tables pass + (no tables) text pass
    for i, j in zip(reads, reads[1:]):
        page = events[i][1]
        assert ("close", page) in events[i + 1:j + 1], (
            f"page {page} still open when the next page was read: {events}")
    assert events[reads[-1] + 1:reads[-1] + 2] == [("close", events[reads[-1]][1])]


# ── the event loop: other users are served while a bill is parsed ──────────

# Measures the event loop's own latency: CI runs it alone, not beside xdist
# workers competing for the runner's cores (pytest.ini: serial, timing).
@pytest.mark.serial
@pytest.mark.timing
def test_the_event_loop_keeps_running_while_a_pdf_is_parsed(blank_pdf, monkeypatch):
    def slow(self, *a, **k):
        time.sleep(0.25)
        return []

    monkeypatch.setattr(pdfplumber.page.Page, "extract_tables", slow)
    monkeypatch.setattr(pdfplumber.page.Page, "extract_text", lambda self, *a, **k: "")
    path = blank_pdf(4)

    async def scenario():
        gaps, done = [], False

        async def heartbeat():
            last = time.perf_counter()
            while not done:
                await asyncio.sleep(0.02)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.05)
        await bp.BOQProcessorBlock().process({"file_path": path}, {})
        done = True
        await hb
        return max(gaps)

    worst = asyncio.run(scenario())
    # The parse itself is 4 x 0.25 s, so blocking shows as a gap near that
    # whole second; scheduler noise on a loaded machine reached exactly 0.20 s
    # (virgin-mode run, 23 Sep). The bound sits between the two, not on the noise.
    assert worst < 0.4, f"event loop frozen for {worst:.2f}s while a PDF was parsed"


# ── isolation: the table parse runs in its own process where fork exists ────

def test_a_pdf_table_parse_runs_in_the_isolated_child(blank_pdf, monkeypatch, child_log):
    """Its parse libraries' first import holds the GIL even off the loop
    (CI: 0.41 s after #1001); in the child it holds the child's GIL. Proven
    by the process id the parse itself records."""
    from app.core import extract_isolated

    if not extract_isolated.isolation_available():
        pytest.skip("no fork on this platform: the parse runs in-process here")
    monkeypatch.setattr(pdfplumber.page.Page, "extract_tables",
                        lambda self, *a, **k: child_log.add("pid", os.getpid()) or [])
    _run(blank_pdf(2))
    pids = {e[1] for e in child_log.events() if e[0] == "pid"}
    assert pids and os.getpid() not in pids, f"the PDF table parse ran in the server process: {pids}"


def test_the_time_budget_stops_the_parse_the_child_runs(blank_pdf, monkeypatch, child_log):
    """The isolated child runs BOQProcessorBlock._parse_pdf_sync; here it runs
    in this process, so its budget check is proven on the exact code the
    child executes (and coverage, which does not follow the forked child,
    sees it run)."""
    monkeypatch.setenv("BOQ_PDF_PARSE_SECONDS", "0.5")
    now = [0.0]
    monkeypatch.setattr(bp, "_parse_clock", lambda: now[0])

    def slow(self, *a, **k):
        child_log.add("parsed")
        now[0] += 0.3
        return []

    monkeypatch.setattr(pdfplumber.page.Page, "extract_tables", slow)
    out = bp.BOQProcessorBlock()._parse_pdf_sync(blank_pdf(8), {})
    assert len(child_log.events()) == 2, "the budget must stop the parse"
    assert out["status"] == "error" and out.get("boq_pdf_parse_timeout") is True
    assert "stopped at page 3" in out["error"]
    assert "total_cost" not in out and "line_items" not in out
