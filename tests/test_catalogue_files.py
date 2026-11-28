"""The catalogue files (``app.agents.driver.catalogue_files``): one JSON file
per registered formula, kernel, knowledge entry and retrieval tool.

* drift: the shipped files agree with the code, and a file missing, a file
  too many or a fact changed is each named;
* the loader refuses what it does not know (an unknown field, a bad kind,
  bad JSON, a file in the wrong place, a neighbour without a file);
* chains built from the files alone (the call each file states, its input
  and output names) run through the working sheet: a quantity -> cost ->
  cost-per-area chain and a unit-conversion chain, each with two sets of
  givens, the figures checked against arithmetic done here.

Every input here is built by the test; nothing names a project or a document.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

from app.agents.driver import catalogue_files as cf
from app.agents.driver import sheet as sheet_mod


# -- fixtures --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shipped():
    """The shipped files, loaded once."""
    return cf.load()


@pytest.fixture(scope="module")
def capabilities():
    return cf.capabilities()


@pytest.fixture()
def copy_dir(tmp_path):
    """A private copy of the shipped files to break."""
    root = tmp_path / "catalogue"
    shutil.copytree(cf.files_dir(), root)
    return str(root)


def _read(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return json.load(f)


def _write(root, rel, doc):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def _a_file(shipped, kind, index=0):
    ids = sorted(i for i, d in shipped.items() if d["kind"] == kind)
    return shipped[ids[index]]


# -- an approved purpose is the index line's text --------------------------------------
# enrich, capabilities, catalogue.registry_entries / entries / _built,
# index_lines and index_block: what the model picks from.

def _approve(root, entry_id, purpose, approved=True):
    kind = entry_id.partition(":")[0]
    rel = cf.path_of(kind, entry_id)
    doc = _read(root, rel)
    doc["hints"].update(approved=approved, purpose=purpose)
    _write(root, rel, doc)
    return doc


def _line_of(lines, entry_id):
    return next(line for line in lines if line.startswith(entry_id + " | "))


def _ids_of(shipped, kind, count=2):
    return sorted(i for i, d in shipped.items() if d["kind"] == kind)[:count]


@pytest.mark.parametrize("index,purpose", [(0, "Weighs the steel of one bar mark."),
                                           (1, "Gives the area a crew covers in a shift.")])
def test_enrich_puts_an_approved_purpose_on_the_for_line_of_each_kind(shipped, capabilities, index, purpose):
    from app.agents.driver import catalogue
    held = {e["id"]: e for _k, e in capabilities}
    ids = [_ids_of(shipped, "formula")[index], _ids_of(shipped, "knowledge")[index]]
    files = json.loads(json.dumps(shipped))
    plain = [held[i] for i in ids]
    for entry_id in ids:
        files[entry_id]["hints"].update(approved=True, purpose=f"{purpose} ({entry_id})")
    out = cf.enrich(plain, files)
    assert [e["for"] for e in out] == [f"{purpose} ({i})" for i in ids]
    assert all("for" not in e["gaps"] for e in out)
    # An entry the code gave no for line has that gap; an approved purpose fills it.
    bare = [dict(e, **{"for": "", "gaps": ["for"] + e["gaps"]}) for e in plain]
    filled = cf.enrich(bare, {i: dict(files[i], facts=cf.facts(files[i]["kind"], b))
                              for i, b in zip(ids, bare)})
    assert [e["for"] for e in filled] == [f"{purpose} ({i})" for i in ids]
    assert all("for" not in e["gaps"] for e in filled)
    # The entries handed in are not changed: enrich returns copies.
    assert [e["for"] for e in plain] == [held[i]["for"] for i in ids]
    # A purpose longer than an index line is cut the way every for line is.
    long = (purpose + " ") * 20
    files[ids[0]]["hints"]["purpose"] = long
    cut = cf.enrich(plain, files)[0]["for"]
    assert cut == catalogue._one_line(long) and len(cut) == catalogue.FOR_CHARS and cut.endswith("…")


@pytest.mark.parametrize("index", [0, 1])
def test_enrich_leaves_the_for_line_when_the_hint_may_not_be_shown(shipped, capabilities, index):
    held = {e["id"]: e for _k, e in capabilities}
    ids = [_ids_of(shipped, "formula")[index], _ids_of(shipped, "knowledge")[index]]
    plain = [held[i] for i in ids]
    for change in ({"approved": False, "purpose": "Written, not approved."},
                   {"approved": True, "purpose": "   "},
                   {"approved": True, "purpose": "", "stale": True}):
        files = json.loads(json.dumps(shipped))
        for entry_id in ids:
            files[entry_id]["hints"].update(approved=change["approved"], purpose=change["purpose"])
            if change.get("stale"):
                files[entry_id]["hints"]["purpose"] = "Approved, but the facts moved on."
                files[entry_id]["facts"]["name"] = "a name the code does not give"
        assert [e["for"] for e in cf.enrich(plain, files)] == [held[i]["for"] for i in ids], change


@pytest.mark.parametrize("index,purpose", [(0, "Approved purpose one."), (1, "Approved purpose two.")])
def test_index_lines_and_index_block_show_the_approved_purpose(shipped, copy_dir, monkeypatch, index, purpose):
    from app.agents.driver import catalogue
    monkeypatch.setattr(cf, "files_dir", lambda: copy_dir)
    formula_id = _ids_of(shipped, "formula", 4)[index]
    knowledge_id = _ids_of(shipped, "knowledge", 4)[index]
    unapproved_id = _ids_of(shipped, "formula", 4)[index + 2]
    stale_id = _ids_of(shipped, "knowledge", 4)[index + 2]
    before = catalogue.index_lines()
    _approve(copy_dir, formula_id, purpose + " A formula.")
    _approve(copy_dir, knowledge_id, purpose + " A knowledge entry.")
    _approve(copy_dir, unapproved_id, "Not approved, never shown.", approved=False)
    stale = _approve(copy_dir, stale_id, "Approved over facts that moved on.")
    stale["facts"]["described_as"] = "what the code said once"
    _write(copy_dir, cf.path_of("knowledge", stale_id), stale)
    lines = catalogue.index_lines()
    block = catalogue.index_block().splitlines()
    for entry_id, text in ((formula_id, purpose + " A formula."), (knowledge_id, purpose + " A knowledge entry.")):
        line = _line_of(lines, entry_id)
        assert line.endswith(" | " + text) and line in block
        assert line != _line_of(before, entry_id)
    for entry_id in (unapproved_id, stale_id):
        assert _line_of(lines, entry_id) == _line_of(before, entry_id)
        assert _line_of(before, entry_id) in block
    assert len(lines) == len(before)


@pytest.mark.parametrize("index", [0, 1])
def test_capabilities_and_registry_entries_keep_the_codes_text(shipped, copy_dir, monkeypatch, index):
    from app.agents.driver import catalogue
    monkeypatch.setattr(cf, "files_dir", lambda: copy_dir)
    ids = [_ids_of(shipped, "formula")[index], _ids_of(shipped, "knowledge")[index]]
    code_for = {e["id"]: e["for"] for e in catalogue.registry_entries() if e["id"] in ids}
    for entry_id in ids:
        _approve(copy_dir, entry_id, f"An approved purpose for {entry_id}.")
    shown = {e["id"]: e["for"] for e in catalogue.entries() if e["id"] in ids}
    assert shown == {i: f"An approved purpose for {i}." for i in ids}
    # What the files are checked against is still the code's own text ...
    assert {e["id"]: e["for"] for e in catalogue.registry_entries() if e["id"] in ids} == code_for
    assert {e["id"]: e["for"] for _k, e in cf.capabilities() if e["id"] in ids} == code_for
    # ... so approving a hint never makes its own file drift from the code.
    assert not [line for line in cf.drift(copy_dir) if any(i in line for i in ids)]


# -- drift: one file for every capability, none extra, facts equal to code -----------

@pytest.mark.parametrize("kind", ["formula", "kernel", "knowledge", "rag"])
def test_capabilities_drift_shipped_files_agree_with_code(shipped, capabilities, kind):
    assert cf.drift() == []
    registered = {entry["id"] for k, entry in capabilities if k == kind}
    held = {i for i, d in shipped.items() if d["kind"] == kind}
    # One file per capability of this kind, none extra, each in its kind's directory.
    assert registered and held == registered
    for entry_id in held:
        assert cf.path_of(kind, entry_id).startswith(cf.DIRS[kind] + "/")


# -- the facts of each kind, read from the code -----------------------------------------

@pytest.mark.parametrize("kind,index,reader", [
    ("formula", 0, cf._formula_facts), ("formula", 5, cf._formula_facts),
    ("kernel", 0, cf._kernel_facts), ("kernel", 2, cf._kernel_facts),
    ("knowledge", 0, cf._knowledge_facts), ("knowledge", 7, cf._knowledge_facts),
    ("rag", 0, cf._rag_facts), ("rag", 4, cf._rag_facts)])
def test_each_kinds_facts_are_what_its_file_holds(shipped, capabilities, kind, index, reader):
    entry_id = _a_file(shipped, kind, index)["id"]
    entry = next(e for k, e in capabilities if e["id"] == entry_id)
    own = json.loads(json.dumps(reader(entry)))
    assert own, f"{entry_id}: the {kind} reader gave no facts"
    held = shipped[entry_id]["facts"]
    assert {key: held[key] for key in own} == own


@pytest.mark.parametrize("field", ["choices", "described_as"])
def test_rag_facts_carry_the_offered_choices_and_descriptions(shipped, capabilities, field):
    # A retrieval tool's inputs as offered to the model: what the offered
    # schema declares (an input's choices, its description) is carried.
    entries = {e["id"]: e for k, e in capabilities if k == "rag"}
    carrying = [i for i, e in entries.items()
                if any(item.get(field) for item in cf._rag_facts(e)["inputs"])]
    assert carrying, f"no retrieval tool carries {field}"
    for entry_id in carrying:
        assert cf._rag_facts(entries[entry_id])["inputs"] == shipped[entry_id]["facts"]["inputs"]


@pytest.mark.parametrize("bad_kind", ["recipe", ""])
def test_facts_refuses_an_unknown_kind(capabilities, bad_kind):
    entry = capabilities[0][1]
    with pytest.raises(cf.CatalogueFileError, match="unknown file kind"):
        cf.facts(bad_kind, entry)


@pytest.mark.parametrize("bound", [float("inf"), float("nan")])
def test_facts_refuses_a_figure_json_cannot_hold(capabilities, bound):
    kind, entry = next((k, e) for k, e in capabilities if k == "kernel")
    entry = dict(entry, inputs=[{"name": "x", "range": [0, bound]}])
    with pytest.raises(ValueError):
        cf.facts(kind, entry)


@pytest.mark.parametrize("kind,index", [("formula", 0), ("knowledge", 3)])
def test_drift_names_a_missing_file(copy_dir, shipped, kind, index):
    doc = _a_file(shipped, kind, index)
    rel = cf.path_of(kind, doc["id"])
    os.remove(os.path.join(copy_dir, rel))
    assert cf.drift(copy_dir) == [f"missing file: {rel} for {doc['id']}"]


@pytest.mark.parametrize("kind,index", [("formula", 1), ("rag", 0)])
def test_drift_names_an_extra_file(copy_dir, shipped, kind, index):
    doc = json.loads(json.dumps(_a_file(shipped, kind, index)))
    doc["id"] = f"{doc['id']}_not_registered"
    rel = cf.path_of(kind, doc["id"])
    _write(copy_dir, rel, doc)
    assert cf.drift(copy_dir) == [f"extra file: {rel} names {doc['id']}, which the code does not register"]


@pytest.mark.parametrize("kind,index,field", [("formula", 2, "inputs"), ("kernel", 0, "described_as")])
def test_drift_names_the_facts_that_differ(copy_dir, shipped, kind, index, field):
    doc = json.loads(json.dumps(_a_file(shipped, kind, index)))
    doc["facts"][field] = [] if isinstance(doc["facts"][field], list) and doc["facts"][field] else "changed"
    _write(copy_dir, cf.path_of(kind, doc["id"]), doc)
    assert cf.drift(copy_dir) == [f"{doc['id']}: facts differ from the code in: {field}"]


@pytest.mark.parametrize("first_kind,second_kind,gone_kind,gone_hints", [
    ("formula", "kernel", "rag",
     {"approved": False, "purpose": "", "not_to_be_confused_with": [], "question_kinds": []}),
    ("knowledge", "rag", "formula",
     {"approved": False, "purpose": "", "not_to_be_confused_with": []})])
def test_write_rebuilds_facts_and_keeps_hints(copy_dir, shipped, first_kind, second_kind, gone_kind, gone_hints):
    first, second = _a_file(shipped, first_kind, 0), _a_file(shipped, second_kind, 1)
    gone = _a_file(shipped, gone_kind, 0)
    for doc, purpose in ((first, "What the first one is for."), (second, "What the second one is for.")):
        changed = json.loads(json.dumps(doc))
        changed["facts"]["described_as"] = "stale"
        changed["hints"]["purpose"] = purpose
        _write(copy_dir, cf.path_of(doc["kind"], doc["id"]), changed)
    os.remove(os.path.join(copy_dir, cf.path_of(gone_kind, gone["id"])))
    report = cf.write(copy_dir)
    assert sorted(report["written"]) == sorted([cf.path_of(first_kind, first["id"]),
                                                cf.path_of(second_kind, second["id"]),
                                                cf.path_of(gone_kind, gone["id"])])
    assert report["extra"] == [] and report["removed"] == []
    assert cf.drift(copy_dir) == []
    assert _read(copy_dir, cf.path_of(first_kind, first["id"]))["hints"]["purpose"] == "What the first one is for."
    assert _read(copy_dir, cf.path_of(second_kind, second["id"]))["hints"]["purpose"] == "What the second one is for."
    # The file written for the missing capability carries the empty hint of its kind.
    assert _read(copy_dir, cf.path_of(gone_kind, gone["id"]))["hints"] == gone_hints
    # A second write changes nothing.
    again = cf.write(copy_dir)
    assert again["written"] == [] and again["unchanged"] == len(shipped)


@pytest.mark.parametrize("kind,index", [("knowledge", 0), ("formula", 4)])
def test_write_reports_extra_and_prunes_only_when_asked(copy_dir, shipped, kind, index):
    doc = json.loads(json.dumps(_a_file(shipped, kind, index)))
    doc["id"] = doc["id"] + "_gone"
    rel = cf.path_of(kind, doc["id"])
    _write(copy_dir, rel, doc)
    assert cf.write(copy_dir)["extra"] == [rel]
    assert os.path.exists(os.path.join(copy_dir, rel))
    assert cf.write(copy_dir, prune=True)["removed"] == [rel]
    assert not os.path.exists(os.path.join(copy_dir, rel))


@pytest.mark.parametrize("kind,index,text", [("formula", 3, "{not json"), ("kernel", 1, "[1,")])
def test_write_never_overwrites_an_unreadable_file(copy_dir, shipped, kind, index, text):
    rel = cf.path_of(kind, _a_file(shipped, kind, index)["id"])
    with open(os.path.join(copy_dir, rel), "w", encoding="utf-8") as f:
        f.write(text)
    with pytest.raises(cf.CatalogueFileError, match="unreadable, so not rewritten"):
        cf.write(copy_dir)
    with open(os.path.join(copy_dir, rel), encoding="utf-8") as f:
        assert f.read() == text


# -- loader refusals --------------------------------------------------------------------

@pytest.mark.parametrize("where,field", [("top", "notes"), ("hints", "if_the_user_says")])
def test_load_refuses_an_unknown_field(copy_dir, shipped, where, field):
    doc = json.loads(json.dumps(_a_file(shipped, "formula", 4)))
    (doc if where == "top" else doc["hints"])[field] = "x"
    rel = cf.path_of("formula", doc["id"])
    _write(copy_dir, rel, doc)
    with pytest.raises(cf.CatalogueFileError) as err:
        cf.load(copy_dir)
    assert rel in str(err.value) and field in str(err.value)


@pytest.mark.parametrize("bad_kind", ["recipe", "Formula"])
def test_load_refuses_a_bad_kind(copy_dir, shipped, bad_kind):
    doc = json.loads(json.dumps(_a_file(shipped, "formula", 5)))
    doc["kind"] = bad_kind
    rel = cf.path_of("formula", doc["id"])
    _write(copy_dir, rel, doc)
    with pytest.raises(cf.CatalogueFileError, match=rel.replace(".", r"\.") + ": kind"):
        cf.load(copy_dir)


@pytest.mark.parametrize("text", ["{not json", ""])
def test_load_refuses_bad_json(copy_dir, shipped, text):
    rel = cf.path_of("knowledge", _a_file(shipped, "knowledge", 1)["id"])
    with open(os.path.join(copy_dir, rel), "w", encoding="utf-8") as f:
        f.write(text)
    with pytest.raises(cf.CatalogueFileError, match=rel.replace(".", r"\.") + ": unreadable"):
        cf.load(copy_dir)


@pytest.mark.parametrize("rel", ["formulas/readme.txt", "recipes/a.json"])
def test_load_refuses_a_file_out_of_place(copy_dir, rel):
    path = os.path.join(copy_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{}")
    with pytest.raises(cf.CatalogueFileError, match="not a catalogue file"):
        cf.load(copy_dir)


@pytest.mark.parametrize("from_kind,to_kind", [("formula", "kernel"), ("rag", "knowledge")])
def test_load_refuses_a_file_in_another_kinds_folder(copy_dir, shipped, from_kind, to_kind):
    doc = _a_file(shipped, from_kind, 0)
    wrong = cf.DIRS[to_kind] + "/" + cf.path_of(from_kind, doc["id"]).split("/")[1]
    _write(copy_dir, wrong, doc)
    with pytest.raises(cf.CatalogueFileError, match="does not belong in this file"):
        cf.load(copy_dir)


@pytest.mark.parametrize("neighbour", ["formula:not_registered_here", "knowledge:nowhere"])
def test_load_refuses_a_neighbour_without_a_file(copy_dir, shipped, neighbour):
    doc = json.loads(json.dumps(_a_file(shipped, "formula", 6)))
    doc["hints"]["not_to_be_confused_with"] = [{"id": neighbour, "difference": "another thing"}]
    _write(copy_dir, cf.path_of("formula", doc["id"]), doc)
    with pytest.raises(cf.CatalogueFileError, match=f"names '{neighbour}', which has no file"):
        cf.load(copy_dir)


def test_load_refuses_a_missing_directory(tmp_path):
    for missing in (tmp_path / "none", tmp_path / "also_none"):
        with pytest.raises(cf.CatalogueFileError, match="no catalogue files directory"):
            cf.load(str(missing))


@pytest.mark.parametrize("index,other_index", [(0, 1), (3, 5)])
def test_problems_names_self_reference_duplicates_and_question_kinds(shipped, index, other_index):
    doc = json.loads(json.dumps(_a_file(shipped, "rag", index)))
    assert cf.problems(doc) == []
    other = _a_file(shipped, "rag", other_index)["id"]
    doc["hints"]["not_to_be_confused_with"] = [{"id": doc["id"], "difference": "the file itself, named as its own neighbour"}]
    assert cf.problems(doc) == ["hints/not_to_be_confused_with: names the file itself"]
    doc["hints"]["not_to_be_confused_with"] = [{"id": other, "difference": "a different retrieval tool"},
                                                {"id": other, "difference": "the same tool named again"}]
    assert cf.problems(doc) == ["hints/not_to_be_confused_with: names an id twice"]
    doc["hints"]["not_to_be_confused_with"] = []
    kinds = [{"kind": k, "role": "leads", "how": "reads the passages for it."} for k in cf.QUESTION_KINDS]
    doc["hints"]["question_kinds"] = kinds
    assert cf.problems(doc) == []
    doc["hints"]["question_kinds"] = kinds[:-1]
    assert cf.problems(doc)[0].startswith("hints/question_kinds: each question kind is spoken to exactly once")
    doc["hints"]["question_kinds"] = kinds + kinds[:1]
    assert cf.problems(doc)[0].startswith("hints/question_kinds")


@pytest.mark.parametrize("kind,entry_id", [("recipe", "formula:a"), ("formula", "formula:has space")])
def test_path_of_refuses_what_has_no_file_name(kind, entry_id):
    with pytest.raises(cf.CatalogueFileError, match="has no file name"):
        cf.path_of(kind, entry_id)


@pytest.mark.parametrize("kind,entry_id,rel", [
    ("formula", "formula:a_b", "formulas/a_b.json"), ("rag", "tool:search_x", "rag/search_x.json"),
    ("kernel", "kernel:hat.one", "kernels/hat.one.json"), ("knowledge", "knowledge:k-2", "knowledge/k-2.json")])
def test_path_of_places_each_kind_in_its_directory(kind, entry_id, rel):
    assert cf.path_of(kind, entry_id) == rel


@pytest.mark.parametrize("hints", [None, "a sentence, not an object"])
def test_problems_stops_at_the_schema_when_the_hints_are_not_an_object(shipped, hints):
    # A file the schema refuses is named by the schema alone; the later
    # checks, which read the hints, are not reached.
    doc = json.loads(json.dumps(_a_file(shipped, "rag", 0)))
    if hints is None:
        del doc["hints"]
    else:
        doc["hints"] = hints
    found = cf.problems(doc)
    assert found and all(not line.startswith("hints/not_to_be_confused_with") for line in found)


@pytest.mark.parametrize("change", ["rewrite", "remove"])
def test_identity_changes_when_a_file_changes(copy_dir, shipped, change):
    before = cf.identity(copy_dir)
    assert before == cf.identity(copy_dir) and len(before) == len(shipped) + 1  # + schema.json
    rel = cf.path_of("formula", _a_file(shipped, "formula", 0)["id"])
    if change == "rewrite":
        with open(os.path.join(copy_dir, rel), "a", encoding="utf-8") as f:
            f.write("  ")  # two more bytes: a rewrite
        after = cf.identity(copy_dir)
        assert len(after) == len(before) and after != before
    else:
        os.remove(os.path.join(copy_dir, rel))
        after = cf.identity(copy_dir)
        assert [r for r, _m, _s in after] == [r for r, _m, _s in before if r != rel]


@pytest.mark.parametrize("f_index,r_index", [(0, 0), (9, 2)])
def test_hint_gaps_lists_files_without_a_written_hint(copy_dir, shipped, f_index, r_index):
    before = set(cf.hint_gaps(copy_dir))
    unwritten = {i for i, d in shipped.items()
                 if not d["hints"]["purpose"].strip()
                 or (d["kind"] == "rag" and not d["hints"]["question_kinds"])}
    assert before == unwritten
    formula, rag = _a_file(shipped, "formula", f_index), _a_file(shipped, "rag", r_index)
    kinds = [{"kind": k, "role": "supports", "how": "What it does for this kind."} for k in cf.QUESTION_KINDS]
    for doc in (formula, rag):
        changed = json.loads(json.dumps(doc))
        changed["hints"]["purpose"] = "What it is for."
        if doc["kind"] == "rag":
            changed["hints"]["question_kinds"] = []
        _write(copy_dir, cf.path_of(doc["kind"], doc["id"]), changed)
    after = set(cf.hint_gaps(copy_dir))
    # A formula with a purpose is written; a retrieval tool also needs its question kinds.
    assert after == (before - {formula["id"]}) | {rag["id"]}
    changed["hints"]["question_kinds"] = kinds
    _write(copy_dir, cf.path_of(rag["kind"], rag["id"]), changed)
    assert set(cf.hint_gaps(copy_dir)) == before - {formula["id"], rag["id"]}


@pytest.mark.parametrize("f_index,blank", [(1, "   "), (7, chr(10) + chr(9))])
def test_hint_gaps_counts_a_blank_purpose_as_unwritten(copy_dir, shipped, f_index, blank):
    formula = _a_file(shipped, "formula", f_index)
    changed = json.loads(json.dumps(formula))
    changed["hints"]["purpose"] = "What it is for."
    _write(copy_dir, cf.path_of("formula", formula["id"]), changed)
    assert formula["id"] not in cf.hint_gaps(copy_dir)
    changed["hints"]["purpose"] = blank
    _write(copy_dir, cf.path_of("formula", formula["id"]), changed)
    assert formula["id"] in cf.hint_gaps(copy_dir)


# -- using a file: call_for, hint_text, enrich ------------------------------------------

@pytest.mark.parametrize("values", [{"length_m": 12, "width_m": 4, "depth_m": 2},
                                    {"length_m": 7, "width_m": 3, "depth_m": 1.5}])
def test_call_for_builds_the_files_call_and_refuses_by_name(shipped, values):
    doc = shipped["formula:excavation_volume"]
    call = cf.call_for(doc, values)
    assert call == {"tool": doc["facts"]["call"]["tool"],
                    "arguments": {**doc["facts"]["call"]["arguments"], "params": values}}
    with pytest.raises(cf.CatalogueFileError, match="has no input named colour"):
        cf.call_for(doc, {"length_m": 1, "width_m": 1, "depth_m": 1, "colour": "red"})
    for blank in (None, " "):
        with pytest.raises(cf.CatalogueFileError, match="needs depth_m"):
            cf.call_for(doc, {"length_m": 1, "width_m": 1, "depth_m": blank})
    with pytest.raises(cf.CatalogueFileError, match="the values are an object"):
        cf.call_for(doc, [1, 2, 3])


@pytest.mark.parametrize("a,b", [("formula:excavation_volume", "formula:unit_cost_total"),
                                 ("formula:cost_per_area", "formula:excavation_volume")])
def test_hint_text_and_enrich_use_only_an_approved_file_that_agrees(shipped, capabilities, a, b):
    entries = {e["id"]: e for _k, e in capabilities}
    files = json.loads(json.dumps(shipped))
    for entry_id, purpose in ((a, "First purpose."), (b, "Second purpose.")):
        files[entry_id]["hints"].update(purpose=purpose, not_to_be_confused_with=[
            {"id": a if entry_id == b else b, "difference": "another figure"}])
    assert cf.hint_text(files[a]) == ("First purpose.", f"Not to be confused with {b} (another figure).")
    plain = [dict(entries[a], when_to_use="", when_not=""), dict(entries[b], when_to_use="", when_not="")]
    # Not approved: nothing reaches the entries.
    assert cf.enrich(plain, files) == plain
    files[a]["hints"]["approved"] = True
    out = cf.enrich(plain, files)
    assert out[0]["when_to_use"] == "First purpose." and out[1]["when_to_use"] == ""
    # Approved but the facts no longer agree with the code: not applied.
    files[a]["facts"]["described_as"] = "stale"
    assert cf.enrich(plain, files)[0]["when_to_use"] == ""
    # A registry's own declaration is never replaced.
    files[a]["facts"] = shipped[a]["facts"]
    declared = [dict(plain[0], when_to_use="Declared by the registry.")]
    assert cf.enrich(declared, files)[0]["when_to_use"] == "Declared by the registry."


@pytest.mark.parametrize("purpose", ["First purpose.", "Another purpose."])
def test_enriched_reads_the_loaded_files_and_keeps_entries_when_they_fail(shipped, capabilities,
                                                                         monkeypatch, purpose):
    entry_id = "formula:excavation_volume"
    plain = [dict(e, when_to_use="", when_not="") for _k, e in capabilities if e["id"] == entry_id]
    files = json.loads(json.dumps(shipped))
    files[entry_id]["hints"].update(approved=True, purpose=purpose)
    monkeypatch.setattr(cf, "load", lambda root=None: files)
    assert cf.enriched(plain)[0]["when_to_use"] == purpose

    def broken(root=None):
        raise cf.CatalogueFileError(f"no catalogue files directory: {purpose}")
    monkeypatch.setattr(cf, "load", broken)
    assert cf.enriched(plain) is plain


# -- chains built from the files, run through the working sheet --------------------------

def _plan(files, steps):
    """A sheet plan built only from the files: the entry the file's call
    runs, its inputs checked against the file's own input names
    (``call_for``), and its output read from the file's outputs (the main
    one, or the one the step names, which must be listed in the file)."""
    plan = []
    for sid, entry_id, inputs, output in steps:
        doc = files[entry_id]
        call = cf.call_for(doc, inputs)  # refuses a name the file does not list
        entry = f"formula:{call['arguments']['calculation']}"
        listed = [o["name"] for o in doc["facts"]["outputs"]]
        main = [o["name"] for o in doc["facts"]["outputs"] if o["main"]]
        # The sheet's own rule: the step's output, else the main one, else
        # the only output that is not a unit.
        numeric = [n for n in listed if not n.endswith("_unit")]
        chosen = output or (main[0] if main else (numeric[0] if len(numeric) == 1 else ""))
        assert chosen in listed, f"{entry_id} lists no output {chosen!r}"
        step = {"id": sid, "entry": entry, "inputs": inputs}
        if output:
            step["output"] = output
        plan.append(step)
    return plan


def _run(givens, plan):
    sheet = sheet_mod.new_sheet("t1", "a chained question")
    sheet, problems = sheet_mod.apply_update(sheet, {"givens": givens, "plan": plan})
    assert problems == []
    for step in plan:
        sheet = sheet_mod.run_step(sheet, step["id"])
    return {s["id"]: s for s in sheet["steps"]}, sheet


def _given(gid, value, unit):
    return {"id": gid, "what": gid, "value": value, "unit": unit}


@pytest.mark.parametrize("length,width,depth,rate,area", [(12, 4, 2, 35, 48), (7.5, 3, 1.2, 18.4, 22.5)])
def test_chain_quantity_rate_cost_from_the_files(shipped, length, width, depth, rate, area):
    plan = _plan(shipped, [
        ("s1", "formula:excavation_volume", {"length_m": "g1", "width_m": "g2", "depth_m": "g3"}, "bank_volume_m3"),
        ("s2", "formula:unit_cost_total", {"quantity": "s1", "unit_rate": "g4"}, ""),
        ("s3", "formula:cost_per_area", {"total_cost": "s2", "area": "g5"}, "cost_per_area"),
    ])
    steps, sheet = _run([_given("g1", length, "m"), _given("g2", width, "m"), _given("g3", depth, "m"),
                         _given("g4", rate, "AED"), _given("g5", area, "m2")], plan)
    volume = length * width * depth
    assert [steps[s]["status"] for s in ("s1", "s2", "s3")] == ["done"] * 3
    assert steps["s1"]["result"] == {"name": "bank_volume_m3", "value": pytest.approx(volume), "unit": "m3"}
    assert steps["s2"]["result"]["value"] == pytest.approx(volume * rate)
    assert steps["s3"]["result"]["value"] == pytest.approx(volume * rate / area, rel=1e-6)
    assert sheet["done"] is True


@pytest.mark.parametrize("feet,width,depth", [(30, 13, 7), (100, 2.5, 0.8)])
def test_chain_unit_conversion_then_quantity_from_the_files(shipped, feet, width, depth):
    plan = _plan(shipped, [
        ("s1", "formula:pe_unit_convert", {"value": "g1", "from_unit": "ft", "to_unit": "m"}, "value_out"),
        ("s2", "formula:excavation_volume", {"length_m": "s1", "width_m": "g2", "depth_m": "g3"},
         "bank_volume_m3"),
    ])
    steps, sheet = _run([_given("g1", feet, "-"), _given("g2", width, "m"), _given("g3", depth, "m")], plan)
    metres = feet * 0.3048
    assert steps["s1"]["result"]["value"] == pytest.approx(metres)
    assert steps["s2"]["result"]["value"] == pytest.approx(metres * width * depth, rel=1e-6)
    assert sheet["done"] is True


@pytest.mark.parametrize("bad", [{"length": "g1", "width_m": "g2", "depth_m": "g3"},
                                 {"length_m": "g1", "width_m": "g2"}])
def test_chain_refuses_a_step_the_file_does_not_describe(shipped, bad):
    with pytest.raises(cf.CatalogueFileError):
        _plan(shipped, [("s1", "formula:excavation_volume", bad, "bank_volume_m3")])


@pytest.mark.parametrize("output", ["volume_m3", "total"])
def test_chain_refuses_an_output_the_file_does_not_list(shipped, output):
    with pytest.raises(AssertionError, match="lists no output"):
        _plan(shipped, [("s1", "formula:excavation_volume",
                         {"length_m": "g1", "width_m": "g2", "depth_m": "g3"}, output)])


# -- the files read into the catalogue's own entries ------------------------------------

def _unhinted_formula_ids(shipped, capabilities):
    """Formula ids whose registry declares no when_to_use and whose file exists."""
    return sorted(e["id"] for k, e in capabilities
                  if k == "formula" and not e["when_to_use"] and e["id"] in shipped)


def _entry_of(entry_id):
    from app.agents.driver import catalogue
    return next(e for e in catalogue.entries() if e["id"] == entry_id)


@pytest.mark.parametrize("index,purpose,later", [
    (0, "First approved purpose.", "A rewritten purpose, longer than the first one."),
    (1, "Another approved purpose, for a second formula.", "Short rewrite."),
])
def test_entries_carry_an_approved_files_hint_and_follow_a_rewrite(shipped, capabilities, copy_dir,
                                                                   monkeypatch, index, purpose, later):
    entry_id = _unhinted_formula_ids(shipped, capabilities)[index]
    rel = cf.path_of("formula", entry_id)
    monkeypatch.setattr(cf, "files_dir", lambda: copy_dir)
    doc = _read(copy_dir, rel)
    doc["hints"].update(approved=True, purpose=purpose)
    _write(copy_dir, rel, doc)
    entry = _entry_of(entry_id)
    assert entry["when_to_use"] == purpose
    assert "when_to_use" not in entry["gaps"]
    # A rewritten file is a new state: the catalogue reads it, never a stale copy.
    doc["hints"].update(purpose=later)
    _write(copy_dir, rel, doc)
    assert _entry_of(entry_id)["when_to_use"] == later


@pytest.mark.parametrize("index", [0, 1])
def test_entries_ignore_an_unapproved_files_hint(shipped, capabilities, copy_dir, monkeypatch, index):
    entry_id = _unhinted_formula_ids(shipped, capabilities)[index]
    rel = cf.path_of("formula", entry_id)
    monkeypatch.setattr(cf, "files_dir", lambda: copy_dir)
    doc = _read(copy_dir, rel)
    doc["hints"].update(approved=False, purpose="Written but not approved.")
    _write(copy_dir, rel, doc)
    entry = _entry_of(entry_id)
    assert entry["when_to_use"] == ""
    assert "when_to_use" in entry["gaps"]


@pytest.mark.parametrize("fault", ["{not json", "[]"])
def test_entries_stay_whole_when_the_files_cannot_load(shipped, capabilities, copy_dir, monkeypatch, fault):
    from app.agents.driver import catalogue
    entry_id = _unhinted_formula_ids(shipped, capabilities)[0]
    rel = cf.path_of("formula", entry_id)
    monkeypatch.setattr(cf, "files_dir", lambda: copy_dir)
    doc = _read(copy_dir, rel)
    doc["hints"].update(approved=True, purpose="Read only while every file loads.")
    _write(copy_dir, rel, doc)
    assert _entry_of(entry_id)["when_to_use"] == "Read only while every file loads."
    before = [e["id"] for e in catalogue.entries()]
    with open(os.path.join(copy_dir, "formulas", "zz_broken.json"), "w", encoding="utf-8") as f:
        f.write(fault)
    assert [e["id"] for e in catalogue.entries()] == before
    assert _entry_of(entry_id)["when_to_use"] == ""
