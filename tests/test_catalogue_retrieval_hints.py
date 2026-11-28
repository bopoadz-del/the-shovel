"""The hints of the six retrieval-tool files under ``catalogue/rag`` state
facts about what ONE call of each tool returns. Each test here runs the real
tool handler on inputs it builds itself (two different inputs per fact) and
checks the fact the hint states against the result, so a hint sentence that
stops being true of the code turns a test red.

The last test is the ``when_not`` case owed from the catalogue files PR: an
approved hint with no neighbour and no ``not_for`` kind fills no ``when_not``
and leaves it among the entry's gaps."""
from __future__ import annotations

import json
import os
import uuid

import numpy as np
import pytest

from app.agents.core.tool_registry import ToolCall
from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "catalogue", "rag")


def _hints(name):
    with open(os.path.join(ROOT, name + ".json"), encoding="utf-8") as fh:
        return json.load(fh)["hints"]


def _call(name, args, project_id="p"):
    return ToolCall(agent=None, name=name, args=args, project_id=project_id)


# -- the files themselves ----------------------------------------------------

@pytest.mark.parametrize("name", ["search_project_documents", "search_general_knowledge", "fetch_document",
                                  "list_project_documents", "bill_section_map", "bill_rows_aggregate"])
def test_each_retrieval_hint_is_written_and_left_unapproved(name):
    hints = _hints(name)
    assert hints["purpose"].strip()
    assert hints["approved"] is False


# -- bill_rows_aggregate -----------------------------------------------------

def _row(path, page, ref, desc, unit="", qty="", amount=""):
    fields = {"description": desc}
    for k, v in (("unit", unit), ("quantity", qty), ("total", amount)):
        if v:
            fields[k] = v
    return g.BillRow(section=path[-1], source_page=page, fields=fields,
                     extra={"Item": ref} if ref else {}, path=tuple(path))


ONE = [_row(("Works", "Drains"), 2, "A1", "Pipe bedding granular", "m3", "10", "50"),
       _row(("Works", "Drains"), 2, "A2", "Pipe bedding concrete", "m", "4", "36"),
       _row(("Works", "Drains"), 3, "A3", "Pipe bedding item only", "m3", "", "")]
TWO = [_row(("Roads", "Base"), 7, "B1", "Pipe bedding under road", "m3", "5", "20"),
       _row(("Roads", "Base"), 8, "B2", "Kerb laying", "m", "80", "200")]


@pytest.fixture
def bills(tmp_path, monkeypatch):
    from app.core import bill_derived, projects

    derived = {"derivation": bill_derived.DERIVATION}
    docs = []
    for i, rows in enumerate((ONE, TWO), start=1):
        path = str(tmp_path / f"b{i}.xlsx")
        g.write_bill_workbook(rows, path)
        docs.append({"id": f"d{i}", "original_name": f"b{i}.bill.xlsx", "path": path,
                     "metadata": {**derived, "derived_from": f"s{i}"}})
    monkeypatch.setattr(projects, "list_documents", lambda pid, **_kw: docs)
    monkeypatch.setattr(projects, "materialize_document_path", lambda d: d["path"])


async def _aggregate(args):
    from app.agents.base.tools import handle_bill_rows_aggregate
    return await handle_bill_rows_aggregate(_call("bill_rows_aggregate", args))


@pytest.mark.asyncio
@pytest.mark.parametrize("args, per_bill", [
    # every derived sheet, one result each, never one added total
    ({"terms": "pipe bedding"}, [("d1", 3, {"m3": 10.0, "m": 4.0}, ["A3"]), ("d2", 1, {"m3": 5.0}, [])]),
    # only the sheet derived from the named document
    ({"terms": "kerb", "document_id": "s2"}, [("d2", 1, {"m": 80.0}, [])]),
], ids=["every-bill", "one-bill"])
async def test_bill_rows_one_call_counts_and_sums_every_row_per_bill(bills, args, per_bill):
    out = await _aggregate(args)
    assert out["ok"] is True
    got = [(b["document_id"], b["count"], b["sum_by_unit"], [r["ref"] for r in b["without_figure"]])
           for b in out["result"]["bills"]]
    assert got == per_bill
    for bill in out["result"]["bills"]:
        # each row used comes back with its reference and its page
        assert all(r["ref"] and r["pages"] for r in bill["rows"])
        assert len(bill["rows"]) == bill["count"]


@pytest.mark.asyncio
@pytest.mark.parametrize("units, single", [("pipe bedding granular", 10.0), ("pipe bedding", None)])
async def test_bill_rows_mixed_units_give_one_sum_per_unit_and_no_single_sum(bills, units, single):
    out = await _aggregate({"terms": units, "document_id": "d1"})
    assert out["result"]["bills"][0]["sum"] == single


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [{"terms": "pipe bedding"}, {"terms": "kerb", "measure": "amount"}])
async def test_bill_rows_the_same_call_again_returns_the_same_rows(bills, args):
    first = await _aggregate(args)
    again = await _aggregate(args)
    assert first == again
    assert first["result"]["bills"]


@pytest.mark.asyncio
@pytest.mark.parametrize("held, refs_pages", [
    ([(None, "", "7", "21"), (4, "", "3", "9")], [("", []), ("", [4])]),
    ([(6, "C1", "2", "8"), (None, "C2", "5", "10"), (6, "", "1", "4")], [("C1", [6]), ("C2", []), ("", [6])]),
], ids=["no-reference", "page-or-reference-missing"])
async def test_bill_rows_a_row_without_reference_or_page_is_still_counted(tmp_path, monkeypatch, held, refs_pages):
    """A row the sheet gives no reference or page for is counted and returned
    with that field empty: the reference and page come only where the sheet
    gives them."""
    from app.core import bill_derived, projects
    rows = [_row(("Works", "Drains"), page, ref, f"Pipe bedding {i}", "m3", qty, amount)
            for i, (page, ref, qty, amount) in enumerate(held)]
    path = str(tmp_path / "b.xlsx")
    g.write_bill_workbook(rows, path)
    docs = [{"id": "d1", "original_name": "b.bill.xlsx", "path": path,
             "metadata": {"derivation": bill_derived.DERIVATION, "derived_from": "s1"}}]
    monkeypatch.setattr(projects, "list_documents", lambda pid, **_kw: docs)
    monkeypatch.setattr(projects, "materialize_document_path", lambda d: d["path"])
    out = await _aggregate({"terms": "pipe bedding"})
    [bill] = out["result"]["bills"]
    assert bill["count"] == len(held)
    assert [(r["ref"], r["pages"]) for r in bill["rows"]] == refs_pages
    assert bill["sum_by_unit"] == {"m3": float(sum(int(q) for _p, _r, q, _a in held))}


# -- bill_section_map --------------------------------------------------------

def _index(project_id, doc_id, chunks):
    from app.core.rag.vector_store import get_store
    store = get_store()
    store.upsert_chunks(project_id, doc_id, chunks, np.zeros((len(chunks), store.dim), dtype="float32"))


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks, sections", [
    (["BILL NO. 1 - EARTHWORKS\nA1 Excavate m3 40", "BILL NO. 2 - CONCRETE\nB1 Slab m3 9"], ["1", "2"]),
    (["Bill Number 3 Drainage\nC1 Pipe m 20", "BILL # 4 - ROADS\nD1 Kerb m 80"], ["3", "4"]),
    (["A1 Excavate m3 40", "A2 Fill m3 12"], []),
    # a heading of another form is not read: the list is empty although the bill has sections
    (["SECTION 1 EARTHWORKS\nA1 Excavate m3 40", "SECTION 2 CONCRETE\nB1 Slab m3 9"], []),
    (["PART A - SITE WORKS\nA1 Clear m2 400", "PART B - ROADS\nB1 Kerb m 80"], []),
], ids=["two-sections", "number-and-hash", "no-heading", "section-headings", "part-headings"])
async def test_section_map_one_document_sections_in_order_and_no_rows(chunks, sections):
    from app.agents.base.bill_section_map import handle_bill_section_map
    tag = uuid.uuid4().hex[:8]
    project_id, doc_id = f"p-{tag}", f"d-{tag}"
    _index(project_id, doc_id, chunks)
    out = await handle_bill_section_map(_call("bill_section_map", {"document_id": doc_id}, project_id))
    assert out["ok"] is True
    assert [s["section"] for s in out["result"]["results"]] == sections
    # it returns sections only: no quantity, amount or row in any result
    for s in out["result"]["results"]:
        assert set(s) <= {"document_id", "filename", "section", "text"}


# -- list_project_documents ---------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("held, limit, returned", [(3, None, 3), (250, 500, 200)], ids=["few", "capped"])
async def test_document_list_counts_all_returns_up_to_the_cap_and_no_id(monkeypatch, held, limit, returned):
    from app.agents.base.tools import handle_list_project_documents
    from app.core import projects

    docs = [{"id": f"id{i}", "original_name": f"f{i}.pdf", "doc_type": "pdf", "size": i} for i in range(held)]
    monkeypatch.setattr(projects, "_master_corpus_source", lambda pid: None)
    asked = []

    def listed(pid, limit=100, **kw):
        asked.append(kw.get("newest_first"))
        return docs[:limit]

    monkeypatch.setattr(projects, "list_documents", listed)
    monkeypatch.setattr(projects, "count_documents", lambda pid: len(docs))
    args = {} if limit is None else {"limit": limit}
    out = await handle_list_project_documents(_call("list_project_documents", args))
    result = out["result"]
    assert (result["total_documents"], result["returned"]) == (held, returned)
    assert asked == [True]
    # each entry is file name, type and size: no text and no document id
    assert all(set(d) == {"filename", "doc_type", "size_bytes"} for d in result["documents"])


# -- fetch_document -----------------------------------------------------------

@pytest.fixture
def fetchable(monkeypatch):
    from app.agents import runtime
    from app.core import doc_index, projects
    from app.core.rag import retriever

    docs = [{"id": "a", "original_name": "wall spec.docx", "file_path": "x"},
            {"id": "b", "original_name": "wall schedule.docx", "file_path": "y"}]
    monkeypatch.setattr(projects, "list_documents", lambda pid, **_kw: docs)
    monkeypatch.setattr(retriever, "_general_knowledge_project_ids", lambda: [])
    monkeypatch.setattr(retriever, "_master_corpus_fallback_id", lambda: None)
    monkeypatch.setattr(retriever, "available", lambda: False)
    monkeypatch.setattr(runtime, "_partial_read_note_for_doc", lambda doc: None)
    monkeypatch.setattr(doc_index, "extract_document_text", lambda path, name: "word " * 50)
    return runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("cap, truncated, length", [(1000, False, 250), (250, False, 250), (249, True, 249),
                                                    (40, True, 40)])
async def test_fetch_returns_text_up_to_a_fixed_length_and_says_when_cut(fetchable, monkeypatch,
                                                                         cap, truncated, length):
    from app.agents.base.tools import handle_fetch_document
    monkeypatch.setattr(fetchable, "_effective_fetch_document_max_chars", lambda: cap)
    out = await handle_fetch_document(_call("fetch_document", {"filename": "wall spec.docx"}))
    assert out["ok"] is True
    assert (out["result"]["truncated"], len(out["result"]["content"])) == (truncated, length)


@pytest.mark.asyncio
@pytest.mark.parametrize("name, ok", [("wall", False), ("schedule", True)])
async def test_fetch_a_name_matching_several_documents_names_them_all(fetchable, name, ok):
    from app.agents.base.tools import handle_fetch_document
    out = await handle_fetch_document(_call("fetch_document", {"filename": name}))
    assert out["ok"] is ok
    if not ok:
        assert "wall spec.docx" in out["result"]["error"] and "wall schedule.docx" in out["result"]["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("held, asked, picked", [
    ([("a", "wall spec.docx"), ("b", "wall spec.docx")], "wall spec.docx", "b"),
    ([("a", "Wall Spec.docx"), ("b", "wall spec.DOCX"), ("c", "roof plan.pdf")], "WALL SPEC.docx", "b"),
    ([("a", "wall spec.docx"), ("b", "wall spec.docx.old")], "wall spec.docx", "a"),
], ids=["same-name", "same-name-other-case", "exact-beside-partial"])
async def test_fetch_a_name_equal_to_a_file_name_is_taken_without_asking(fetchable, monkeypatch, held, asked, picked):
    from app.agents.base.tools import handle_fetch_document
    from app.core import projects
    docs = [{"id": i, "original_name": n, "file_path": i} for i, n in held]
    monkeypatch.setattr(projects, "list_documents", lambda pid, **_kw: docs)
    out = await handle_fetch_document(_call("fetch_document", {"filename": asked}))
    assert (out["ok"], out["result"]["document_id"]) == (True, picked)


@pytest.mark.asyncio
@pytest.mark.parametrize("words, cap, windows", [(50, 2000, 1), (1500, 2000, 4), (3000, 4000, 4)])
async def test_fetch_longer_than_one_tool_message_comes_in_windows_read_with_char_offset(
        fetchable, monkeypatch, words, cap, windows):
    """The public path: the handler's result as the tool message carries it.
    Under the fetch length the result's own truncated stays False, and the
    message alone is marked truncated, with next_char_offset for the rest."""
    from app.agents.base.tools import handle_fetch_document
    from app.core import doc_index
    monkeypatch.setattr(doc_index, "extract_document_text", lambda path, name: "word " * words)
    monkeypatch.setattr(fetchable, "_effective_fetch_document_max_chars", lambda: 100_000)
    out = await handle_fetch_document(_call("fetch_document", {"document_id": "a"}))
    assert out["result"]["truncated"] is False
    whole = json.dumps(out["result"], default=str)
    message = json.loads(fetchable._tool_result_content(out["result"], max_chars=cap))
    if windows == 1:
        assert message == out["result"]
        return
    seen, offset, count = "", 0, 0
    while offset is not None:
        message = json.loads(fetchable._tool_result_content(out["result"], offset=offset, max_chars=cap))
        assert message["truncated"] is True and message["char_offset"] == offset
        assert len(json.dumps(message)) <= cap
        seen += message["preview"]
        offset = message["next_char_offset"]
        count += 1
    assert seen == whole
    assert count >= windows


@pytest.mark.asyncio
@pytest.mark.parametrize("note", [None, "pages 1-3 of 9 were read"])
async def test_fetch_says_when_only_part_of_a_scan_was_read(fetchable, monkeypatch, note):
    from app.agents.base.tools import handle_fetch_document
    monkeypatch.setattr(fetchable, "_partial_read_note_for_doc", lambda doc: note)
    out = await handle_fetch_document(_call("fetch_document", {"document_id": "b"}))
    assert out["ok"] is True
    assert out["result"]["partial_read"] == note


# -- search_project_documents -------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("args, asked", [({"query": "q"}, 5), ({"query": "q", "top_k": "3"}, 3),
                                         ({"query": "q", "top_k": "abc"}, 5)])
async def test_search_asks_for_top_k_passages_default_five(monkeypatch, args, asked):
    from app.agents.base.tools import handle_search_project_documents
    from app.core import doc_index

    seen = []

    async def fake(project_id, query, top_k, **_kw):
        seen.append(top_k)
        return [{"document_id": "d", "snippet": f"s{i}", "score": 1.0 - i / 10} for i in range(top_k)]

    monkeypatch.setattr(doc_index, "search_project_documents", fake)
    out = await handle_search_project_documents(_call("search_project_documents", args))
    assert seen == [asked]
    # the rows come back in the order the search ranked them
    assert [r["snippet"] for r in out["result"]["results"]] == [f"s{i}" for i in range(asked)]


@pytest.mark.parametrize("cap, kept", [(10_000, 4), (260, 1), (330, 2)])
def test_search_rows_over_one_message_are_left_out_whole_and_counted(cap, kept):
    from app.agents.base.tools import search_rows_within_cap
    rows = [{"document_id": "d", "snippet": "word " * 10} for _ in range(4)]
    payload = search_rows_within_cap(rows, cap)
    assert len(payload["results"]) == kept
    assert payload.get("rows_withheld", 0) == 4 - kept
    assert all(r == rows[0] for r in payload["results"])


@pytest.mark.parametrize("first_words, rows, cap", [(3000, 2, 800), (2000, 1, 500), (4000, 3, 1200)])
def test_search_a_top_row_too_long_alone_is_kept_shortened_from_the_end(first_words, rows, cap):
    from app.agents.base.tools import search_rows_within_cap
    first = " ".join(f"w{i}" for i in range(first_words))
    results = [{"document_id": "d0", "snippet": first}] + [
        {"document_id": f"d{i}", "snippet": "short"} for i in range(1, rows)]
    payload = search_rows_within_cap(results, cap)
    assert len(json.dumps(payload)) <= cap
    [kept] = payload["results"]
    assert kept["document_id"] == "d0"
    snippet = kept["snippet"]
    assert snippet.endswith(" ...") and first.startswith(snippet[:-4]) and len(snippet) < len(first)
    assert payload.get("rows_withheld", 0) == rows - 1
    assert "shortened" in payload["note"] and "every row shown is complete" not in payload["note"]


@pytest.mark.parametrize("cap, kept", [(10_000, 4), (260, 1)])
def test_search_rows_that_fit_whole_are_not_said_to_be_shortened(cap, kept):
    from app.agents.base.tools import search_rows_within_cap
    rows = [{"document_id": "d", "snippet": "word " * 10} for _ in range(4)]
    payload = search_rows_within_cap(rows, cap)
    assert len(payload["results"]) == kept
    assert "shortened" not in payload.get("note", "")


# -- the numbers each hint states, read from the hint and run -----------------

def _stated(name, word):
    import re
    found = re.search(word + r" (\d+)", _hints(name)["purpose"])
    assert found, (name, word)
    return int(found.group(1))


async def _asked_search(monkeypatch, top_k):
    from app.agents.base.tools import handle_search_project_documents
    from app.core import doc_index
    seen = []

    async def fake(project_id, query, top_k, **_kw):
        seen.append(top_k)
        return []

    monkeypatch.setattr(doc_index, "search_project_documents", fake)
    args = {"query": "q"} if top_k is None else {"query": "q", "top_k": top_k}
    await handle_search_project_documents(_call("search_project_documents", args))
    return seen[0]


async def _asked_general(monkeypatch, top_k):
    from app.agents.base.knowledge import handle_search_general_knowledge
    from app.core import doc_index
    from app.core.rag import retriever

    async def fake(project_id, query, k, **_kw):
        return [{"score": 1.0 - i / 1000} for i in range(k)]

    monkeypatch.setattr(retriever, "_general_knowledge_project_ids", lambda: ["gk1"])
    monkeypatch.setattr(doc_index, "search_project_documents", fake)
    args = {"query": "q"} if top_k is None else {"query": "q", "top_k": top_k}
    out = await handle_search_general_knowledge(_call("search_general_knowledge", args, None))
    return len(out["result"]["results"])


async def _asked_list(monkeypatch, limit):
    from app.agents.base.tools import handle_list_project_documents
    from app.core import projects
    docs = [{"id": f"i{n}", "original_name": f"f{n}.pdf", "doc_type": "pdf", "size": n} for n in range(500)]
    monkeypatch.setattr(projects, "_master_corpus_source", lambda pid: None)
    monkeypatch.setattr(projects, "list_documents", lambda pid, limit=100, **kw: docs[:limit])
    monkeypatch.setattr(projects, "count_documents", lambda pid: len(docs))
    args = {} if limit is None else {"limit": limit}
    out = await handle_list_project_documents(_call("list_project_documents", args))
    return out["result"]["returned"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name, run", [("search_project_documents", _asked_search),
                                       ("search_general_knowledge", _asked_general),
                                       ("list_project_documents", _asked_list)])
async def test_the_default_a_hint_states_is_what_a_call_without_it_gets(monkeypatch, name, run):
    assert await run(monkeypatch, None) == _stated(name, "default")


@pytest.mark.asyncio
@pytest.mark.parametrize("name, run", [("search_general_knowledge", _asked_general),
                                       ("list_project_documents", _asked_list)])
async def test_the_ceiling_a_hint_states_is_what_a_larger_ask_gets(monkeypatch, name, run):
    ceiling = _stated(name, "at most")
    assert await run(monkeypatch, ceiling + 1) == ceiling
    assert await run(monkeypatch, 10 * ceiling) == ceiling
    assert await run(monkeypatch, ceiling - 1) == ceiling - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limit, returned", [(-5, 1), (0, 100), ("abc", 100)])
async def test_document_list_limit_below_one_or_not_a_number(monkeypatch, limit, returned):
    assert await _asked_list(monkeypatch, limit) == returned


# -- search_general_knowledge -------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("libraries, top_k, scores", [
    (["gk1", "gk2"], None, [0.98, 0.97, 0.96, 0.95, 0.94]),
    (["gk1", "gk2"], 99, [round(0.98 - i / 100, 2) for i in range(20)]),
    (["gk2"], 3, [0.97, 0.95, 0.93]),
], ids=["default-five", "at-most-twenty", "one-library"])
async def test_general_knowledge_merges_every_library_best_first(monkeypatch, libraries, top_k, scores):
    from app.agents.base.knowledge import handle_search_general_knowledge
    from app.core import doc_index
    from app.core.rag import retriever

    # 15 hits per library, interleaved by score: the merge is best first across both.
    per = {"gk1": [round(0.98 - 2 * i / 100, 2) for i in range(15)],
           "gk2": [round(0.97 - 2 * i / 100, 2) for i in range(15)]}

    async def fake(project_id, query, k, **_kw):
        return [{"score": s} for s in per[project_id][:k]]

    monkeypatch.setattr(retriever, "_general_knowledge_project_ids", lambda: libraries)
    monkeypatch.setattr(doc_index, "search_project_documents", fake)
    args = {"query": "q"} if top_k is None else {"query": "q", "top_k": top_k}
    out = await handle_search_general_knowledge(_call("search_general_knowledge", args, None))
    got = out["result"]["results"]
    assert [h["score"] for h in got] == scores
    assert {h["layer"] for h in got} == {"general_knowledge"}


@pytest.mark.asyncio
@pytest.mark.parametrize("libraries, query, error", [
    ([], "q", "no general-knowledge library is configured"), (["gk1"], " ", "no query")])
async def test_general_knowledge_without_a_library_or_query_is_an_error(monkeypatch, libraries, query, error):
    from app.agents.base.knowledge import handle_search_general_knowledge
    from app.core.rag import retriever
    monkeypatch.setattr(retriever, "_general_knowledge_project_ids", lambda: libraries)
    out = await handle_search_general_knowledge(_call("search_general_knowledge", {"query": query}, None))
    assert (out["ok"], out["result"]["error"]) == (False, error)


# -- the when_not gap owed from the catalogue files PR ------------------------

@pytest.mark.parametrize("name", ["arithmetic_step", "beam_deflection_cantilever_point_load"])
def test_an_approved_hint_with_no_neighbour_leaves_when_not_a_gap(name):
    from app.agents.driver import catalogue, catalogue_files

    entry = next(e for e in catalogue.entries() if e["id"] == f"formula:{name}")
    bare = {**entry, "when_to_use": "", "when_not": "",
            "gaps": sorted(set(entry["gaps"]) | {"when_to_use", "when_not"})}
    doc = {"kind": "formula", "facts": catalogue_files.facts("formula", bare),
           "hints": {"approved": True, "purpose": f"Works out {name}.", "not_to_be_confused_with": []}}
    out = catalogue_files.enrich([bare], {bare["id"]: doc})[0]
    assert out["when_to_use"] == f"Works out {name}."
    assert out["when_not"] == ""
    assert "when_not" in out["gaps"] and "when_to_use" not in out["gaps"]


# -- the same file name on several documents ----------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("project_docs, library_docs, read", [
    # a re-upload: two project documents of one name
    ([("old", "BOQ.xlsx"), ("new", "BOQ.xlsx")], [], "new"),
    # a project file and a general-knowledge library file of one name
    ([("mine", "Concrete Spec.pdf")], [("lib1", "concrete spec.PDF")], "lib1"),
    ([("mine", "Concrete Spec.pdf"), ("other", "roof.pdf")], [("lib1", "Concrete Spec.pdf"), ("lib2", "x.pdf")], "lib1"),
], ids=["re-upload", "library-twin", "library-twin-among-others"])
async def test_fetch_by_a_name_several_documents_share_reads_one_and_the_others_are_out_of_reach(
        fetchable, monkeypatch, project_docs, library_docs, read):
    from app.agents.base.tools import handle_fetch_document
    from app.core import projects
    from app.core.rag import retriever

    held = {"p": [{"id": i, "original_name": n, "file_path": i} for i, n in project_docs],
            "lib": [{"id": i, "original_name": n, "file_path": i} for i, n in library_docs]}
    monkeypatch.setattr(projects, "list_documents", lambda pid, **_kw: held.get(pid, []))
    monkeypatch.setattr(retriever, "_general_knowledge_project_ids", lambda: ["lib"] if library_docs else [])
    sharing = [(i, n) for i, n in project_docs + library_docs if n.lower() == project_docs[0][1].lower()]
    assert len(sharing) >= 2
    got = set()
    for _id, name in sharing:
        out = await handle_fetch_document(_call("fetch_document", {"filename": name}))
        assert out["ok"] is True and "error" not in out["result"]
        got.add(out["result"]["document_id"])
    # Whichever spelling of the shared name is asked, the same one document is read.
    assert got == {read}
    assert "one is read and the others cannot be reached by name" in _hints("fetch_document")["purpose"]


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [["BOQ.xlsx", "BOQ.xlsx"], ["a.pdf", "A.pdf", "b.pdf"]])
async def test_document_list_can_carry_one_file_name_twice_and_gives_no_id_to_tell_them_apart(monkeypatch, names):
    from app.agents.base.tools import handle_list_project_documents
    from app.core import projects

    docs = [{"id": f"d{i}", "original_name": n, "doc_type": "x", "size": i + 1} for i, n in enumerate(names)]
    monkeypatch.setattr(projects, "_master_corpus_source", lambda pid: None)
    monkeypatch.setattr(projects, "list_documents", lambda pid, limit=100, **_kw: docs[:limit])
    monkeypatch.setattr(projects, "count_documents", lambda pid: len(docs))
    out = await handle_list_project_documents(_call("list_project_documents", {}))
    listed = out["result"]["documents"]
    assert [entry["filename"] for entry in listed] == names
    assert all(set(entry) == {"filename", "doc_type", "size_bytes"} for entry in listed)
    purpose = _hints("list_project_documents")["purpose"]
    assert "two entries can carry the same file name" in purpose and "accepts the file name" not in purpose
