"""RQ-2: a floor and a ceiling on every chunk, and a read-only plan of the
stored chunks outside them. Every test builds its own text and rows; each
rule is decided on two inputs (one each side of the rule).
"""
from __future__ import annotations

import pytest

from app.core import chunk_bounds as cb
from app.core import ingest_status as ist
from app.core.chunk_bounds_plan import chunk_bounds_plan
from tests.test_ingest_reconcile import _reload_projects
from tests.test_legacy_guard_chunk_purge import _seed


@pytest.fixture(autouse=True)
def _bounds(monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_CHARS", "120")
    monkeypatch.setenv("CHUNK_MIN_CHARS", "30")


def _sentences(n: int) -> str:
    return " ".join(f"Clause {i} requires grade C{30 + i} concrete." for i in range(n))


def _rows(n: int) -> str:
    return "\n".join(f"{i}.1 | Excavation in rock | m3 | {100 + i}" for i in range(n))


@pytest.mark.parametrize(
    "text, pieces",
    [
        (_sentences(2), 1),   # under the ceiling: one chunk, untouched
        (_sentences(8), 3),   # over it: cut at sentence ends
    ],
)
def test_ceiling_splits_only_oversize_text(text, pieces):
    out = cb.bound_chunks([text])
    assert len(out) == pieces
    assert all(len(c) <= 120 for c in out)
    # Nothing lost or reordered: the words come back in the same order.
    assert " ".join(out).split() == text.split()


@pytest.mark.parametrize("text", [_rows(6), _sentences(8)])
def test_split_lands_on_a_row_or_sentence_boundary(text):
    out = cb.bound_chunks([text])
    assert len(out) > 1
    ends = (".",) if "|" not in text else tuple("0123456789")
    assert all(c.rstrip().endswith(ends) for c in out)
    if "|" in text:
        # A row is never cut: every output line is a whole input line.
        assert all(ln in text.splitlines() for c in out for ln in c.splitlines())


@pytest.mark.parametrize(
    "chunks, expected",
    [
        # a lone heading below the floor joins the chunk after it
        (["Section 4", _sentences(1)], [f"Section 4\n{_sentences(1)}"]),
        # both above the floor: left alone
        ([_sentences(1), _sentences(1)], [_sentences(1), _sentences(1)]),
    ],
)
def test_floor_merges_only_short_chunks(chunks, expected):
    assert cb.bound_chunks(chunks) == expected


@pytest.mark.parametrize(
    "chunks, expected_len",
    [
        # short tail joins the previous chunk when it fits
        ([_sentences(1), "End."], 1),
        # short chunk whose neighbour is full stays as is: text never dropped
        ([("word " * 24).strip(), "End."], 2),
    ],
)
def test_floor_never_breaks_the_ceiling(chunks, expected_len):
    out = cb.bound_chunks(chunks)
    assert len(out) == expected_len
    assert "End." in "\n".join(out)


@pytest.mark.parametrize("env, pieces", [("0", 1), ("120", 3)])
def test_ceiling_is_configurable_and_zero_turns_it_off(monkeypatch, env, pieces):
    monkeypatch.setenv("CHUNK_MAX_CHARS", env)
    assert len(cb.bound_chunks([_sentences(8)])) == pieces


@pytest.mark.parametrize(
    "raw, value",
    [("", cb.DEFAULT_MAX_CHARS), ("abc", cb.DEFAULT_MAX_CHARS), ("-5", 0), ("900", 900)],
)
def test_ceiling_env_parsing(monkeypatch, raw, value):
    monkeypatch.setenv("CHUNK_MAX_CHARS", raw)
    assert cb.chunk_max_chars() == value


@pytest.mark.parametrize(
    "merge, expected",
    [
        # prose: the short heading joins the next chunk
        (True, ["Item 4.1\n4.1 | Excavation in rock | m3 | 104"]),
        # unit chunks (rows): never joined, so each row stays its own chunk
        (False, ["Item 4.1", "4.1 | Excavation in rock | m3 | 104"]),
    ],
)
def test_unit_chunks_get_the_ceiling_only(merge, expected):
    out = cb.bound_chunks(["Item 4.1", "4.1 | Excavation in rock | m3 | 104"], merge=merge)
    assert out == expected


@pytest.mark.parametrize("floor, merged", [("30", True), ("500", False)])
def test_a_floor_at_or_above_the_ceiling_is_ignored(monkeypatch, floor, merged):
    monkeypatch.setenv("CHUNK_MIN_CHARS", floor)
    out = cb.bound_chunks(["Section 4", _sentences(1)])
    assert (len(out) == 1) is merged


@pytest.mark.parametrize(
    "lengths, expected",
    [
        ([10, 50, 60, 200], {"chunks": 4, "over_ceiling": 1, "under_floor": 1,
                             "p50": 60, "p95": 200, "max": 200}),
        ([], {"chunks": 0, "over_ceiling": 0, "under_floor": 0,
              "p50": None, "p95": None, "max": None}),
    ],
)
def test_bounds_summary(lengths, expected):
    got = cb.bounds_summary(lengths)
    assert {k: got[k] for k in expected} == expected
    assert (got["min_chars"], got["max_chars"]) == (30, 120)


@pytest.mark.parametrize(
    "page2_sentences, last_page",
    [(3, 2), (0, 1)],
)
def test_text_stage_pages_follow_the_bounded_chunks(page2_sentences, last_page):
    from app.core import doc_index

    page1 = _sentences(4)
    page2 = _sentences(page2_sentences).replace("Clause", "Article")
    text = f"{page1}\n{page2}" if page2 else page1
    starts = [[1, 0]] + ([[2, len(page1) + 1]] if page2 else [])
    chunks, pages = doc_index._text_chunks_with_pages(
        text, {"page_starts": starts}, "default", "spec.pdf",
    )
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)
    assert len(pages) == len(chunks)
    assert pages[0] == 1 and pages[-1] == last_page


@pytest.mark.parametrize("role, status", [("admin", None), ("member", 403)])
def test_route_is_admin_only(monkeypatch, tmp_path, role, status):
    from fastapi import HTTPException

    from app.routers import admin

    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("R", user_id="system", project_id="rproj")
    if status is None:
        out = admin.admin_corpus_chunk_bounds(project_id="rproj", auth={"role": role})
        assert out["project_id"] == "rproj" and out["executed"] is False
    else:
        with pytest.raises(HTTPException) as err:
            admin.admin_corpus_chunk_bounds(project_id="rproj", auth={"role": role})
        assert err.value.status_code == status


@pytest.mark.parametrize("pid, raises", [("", True), ("rproj", False)])
def test_plan_requires_a_project(monkeypatch, tmp_path, pid, raises):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("R", user_id="system", project_id="rproj")
    if raises:
        with pytest.raises(ValueError):
            chunk_bounds_plan(project_id=pid)
    else:
        assert chunk_bounds_plan(project_id=pid)["length"]["chunks"] == 0


@pytest.mark.parametrize(
    "name, pages_in_file, texts, expect",
    [
        ("spec.pdf", None, ["x" * 200, "short"],
         {"over_ceiling": 1, "under_floor": 1, "pdf_documents": 1,
          "pdf_documents_without_pages_in_file": 1, "pdf_chunks_without_page": 2}),
        ("bill.xlsx", 3, ["y" * 60, "z" * 70],
         {"over_ceiling": 0, "under_floor": 0, "pdf_documents": 0,
          "pdf_documents_without_pages_in_file": 0, "pdf_chunks_without_page": 0}),
    ],
)
def test_plan_counts_outside_bounds_and_missing_locators(
    monkeypatch, tmp_path, name, pages_in_file, texts, expect,
):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("B", user_id="system", project_id="bproj")
    doc = pm.add_document("bproj", name, size=10)
    pm.stamp_document_index(
        doc["id"], chunk_count=len(texts), ingest_status=ist.INDEXED,
        pages_in_file=pages_in_file,
    )
    _seed("bproj", doc["id"], texts)

    plan = chunk_bounds_plan(project_id="bproj")
    got = {
        "over_ceiling": plan["length"]["over_ceiling"],
        "under_floor": plan["length"]["under_floor"],
        **{k: plan[k] for k in (
            "pdf_documents", "pdf_documents_without_pages_in_file",
            "pdf_chunks_without_page",
        )},
    }
    assert got == expect
    assert plan["executed"] is False
    # read-only: the stored chunk count is unchanged
    assert pm.get_document(doc["id"])["chunk_count"] == len(texts)


# --- RQ-2 addition: each piece of a split page carries that page's heading ---


def _sheet(number: str, title: str, notes: int) -> str:
    body = " ".join(
        f"Note {i}: cover to reinforcement {40 + i} mm in grade C{30 + i} concrete."
        for i in range(notes)
    )
    return f"{body}\nDRAWING NO: {number}\nDRAWING TITLE: {title}"


def _stub_title_block(text: str) -> dict:
    import re

    out = {"drawing_number": None, "drawing_title": None}
    for key, label in (("drawing_number", "DRAWING NO"), ("drawing_title", "DRAWING TITLE")):
        m = re.search(label + r": ([^\n]+)", text)
        if m:
            out[key] = m.group(1).strip()
    return out


@pytest.mark.parametrize(
    "line, is_heading",
    [
        ("3.2 Concrete Works", True),
        ("SECTION 4 - DRAINAGE PIPES", True),
        ("The contractor shall place the concrete in layers.", False),
        ("Note 1: cover to reinforcement 40 mm in grade C30 concrete.", False),
    ],
)
def test_heading_line_shape(line, is_heading):
    assert cb._is_heading_line(line) is is_heading


@pytest.mark.parametrize(
    "page_text, use_title_block, expected",
    [
        # A drawing sheet is named by its title block, not its first line.
        (_sheet("XY-301", "Manhole details", 1), True, "XY-301 - Manhole details"),
        # A text page without a title block: its first heading-like line.
        ("Preamble text that is long.\n2.1 Pipe Bedding\nBed pipes on granular material.",
         True, "2.1 Pipe Bedding"),
        ("Only a sentence here.", False, ""),
    ],
)
def test_heading_line_for_page(page_text, use_title_block, expected):
    reader = _stub_title_block if use_title_block else None
    assert cb.heading_line_for_page(page_text, reader) == expected


@pytest.mark.parametrize("track, cut", [(True, 400 - cb.HEADING_RESERVE), (False, 400)])
def test_split_keeps_room_for_the_heading_only_when_headings_follow(monkeypatch, track, cut):
    monkeypatch.setenv("CHUNK_MAX_CHARS", "400")
    parents: list = []
    out = cb.bound_chunks([_sentences(30)], split_parents=parents if track else None)
    assert len(out) > 1
    assert all(len(c) <= cut for c in out)
    assert any(len(c) > cut - 60 for c in out)
    if track:
        assert len(parents) == len(out) and len(set(parents)) == 1 and parents[0]


@pytest.mark.parametrize("long_parent, tagged", [(True, True), (False, False)])
def test_only_pieces_of_a_split_chunk_are_tagged(monkeypatch, long_parent, tagged):
    monkeypatch.setenv("CHUNK_MAX_CHARS", "400")
    monkeypatch.setenv("CHUNK_MIN_CHARS", "0")
    parents: list = []
    text = _sentences(30) if long_parent else _sentences(3)
    cb.bound_chunks([text, _sentences(2)], split_parents=parents)
    assert (parents[0] is not None) is tagged
    assert parents[-1] is None


@pytest.mark.parametrize(
    "pages, expected_pages",
    [
        ([4, None, 5], [4, 4, 5]),   # an unlocated piece takes its parent's page
        ([4, 5, 5], [4, 5, 5]),      # a located piece keeps its own page
    ],
)
def test_piece_page_is_its_own_else_its_parent(monkeypatch, pages, expected_pages):
    monkeypatch.setenv("CHUNK_MAX_CHARS", "400")
    headings = {4: "XY-301 - Manhole details", 5: "XY-302 - Chamber details"}
    chunks, out_pages = cb.attach_page_headings(
        ["piece a", "piece b", "piece c"], pages, [7, 7, 7], headings,
    )
    assert out_pages == expected_pages
    assert [c.split("\n")[0] for c in chunks] == [headings[p] for p in expected_pages]


@pytest.mark.parametrize(
    "chunk, parent, expected",
    [
        ("piece body", 3, "H - T\npiece body"),          # a piece gets its heading
        ("H - T\npiece body", 3, "H - T\npiece body"),  # not twice
        ("unsplit body", None, "unsplit body"),          # an unsplit chunk is untouched
    ],
)
def test_heading_is_added_once_and_only_to_pieces(monkeypatch, chunk, parent, expected):
    monkeypatch.setenv("CHUNK_MAX_CHARS", "400")
    out, _ = cb.attach_page_headings([chunk], [2], [parent], {2: "H - T"})
    assert out == [expected]


@pytest.mark.parametrize("ceiling, headed", [("400", True), ("12", False)])
def test_heading_never_breaks_the_ceiling(monkeypatch, ceiling, headed):
    monkeypatch.setenv("CHUNK_MAX_CHARS", ceiling)
    out, _ = cb.attach_page_headings(["piece body"], [2], [3], {2: "H - T"})
    assert out[0].startswith("H - T\n") is headed


@pytest.mark.parametrize("notes, split", [(30, True), (2, False)])
def test_text_stage_pieces_of_a_sheet_carry_its_title_block_line(monkeypatch, notes, split):
    from app.core import doc_index

    monkeypatch.setenv("CHUNK_MAX_CHARS", "600")
    monkeypatch.setattr(doc_index, "_title_block_reader", lambda: _stub_title_block)
    cover = "3.1 General Notes\nAll dimensions are in millimetres unless noted."
    sheet = _sheet("XY-301", "Manhole details", notes)
    text = f"{cover}\n{sheet}"
    starts = [[1, 0], [2, len(cover) + 1]]
    chunks, pages = doc_index._text_chunks_with_pages(
        text, {"page_starts": starts}, "default", "drawings.pdf",
    )
    assert all(len(c) <= 600 for c in chunks)
    headed = [c.startswith("XY-301 - Manhole details\n") for c in chunks]
    if split:
        on_sheet = [h for h, p in zip(headed, pages) if p == 2]
        assert len(on_sheet) > 1 and all(on_sheet)
        assert not headed[0] and pages[0] == 1  # the cover page is not the sheet
    else:
        assert len(chunks) == 1 and not any(headed)
