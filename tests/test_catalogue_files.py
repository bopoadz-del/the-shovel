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


# -- drift: one file for every capability, none extra, facts equal to code -----------

def test_capabilities_drift_shipped_files_agree_with_code(shipped, capabilities):
    assert cf.drift() == []
    registered = {entry["id"] for _kind, entry in capabilities}
    assert set(shipped) == registered
    # One file per capability, each in its own kind's directory.
    for kind, entry in capabilities:
        assert shipped[entry["id"]]["kind"] == kind
    kinds = {kind for kind, _e in capabilities}
    assert kinds == set(cf.DIRS)


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


def test_write_rebuilds_facts_and_keeps_hints(copy_dir, shipped):
    first, second = _a_file(shipped, "formula", 0), _a_file(shipped, "kernel", 1)
    for doc, purpose in ((first, "What the first one is for."), (second, "What the second one is for.")):
        changed = json.loads(json.dumps(doc))
        changed["facts"]["described_as"] = "stale"
        changed["hints"]["purpose"] = purpose
        _write(copy_dir, cf.path_of(doc["kind"], doc["id"]), changed)
    os.remove(os.path.join(copy_dir, cf.path_of("rag", _a_file(shipped, "rag", 0)["id"])))
    report = cf.write(copy_dir)
    assert sorted(report["written"]) == sorted([cf.path_of("formula", first["id"]),
                                                cf.path_of("kernel", second["id"]),
                                                cf.path_of("rag", _a_file(shipped, "rag", 0)["id"])])
    assert report["extra"] == [] and report["removed"] == []
    assert cf.drift(copy_dir) == []
    assert _read(copy_dir, cf.path_of("formula", first["id"]))["hints"]["purpose"] == "What the first one is for."
    assert _read(copy_dir, cf.path_of("kernel", second["id"]))["hints"]["purpose"] == "What the second one is for."
    # The file written for the missing capability carries the empty hint of its kind.
    assert _read(copy_dir, cf.path_of("rag", _a_file(shipped, "rag", 0)["id"]))["hints"] == {
        "approved": False, "purpose": "", "not_to_be_confused_with": [], "question_kinds": []}
    # A second write changes nothing.
    again = cf.write(copy_dir)
    assert again["written"] == [] and again["unchanged"] == len(shipped)


def test_write_reports_extra_and_prunes_only_when_asked(copy_dir, shipped):
    doc = json.loads(json.dumps(_a_file(shipped, "knowledge", 0)))
    doc["id"] = doc["id"] + "_gone"
    rel = cf.path_of("knowledge", doc["id"])
    _write(copy_dir, rel, doc)
    assert cf.write(copy_dir)["extra"] == [rel]
    assert os.path.exists(os.path.join(copy_dir, rel))
    assert cf.write(copy_dir, prune=True)["removed"] == [rel]
    assert not os.path.exists(os.path.join(copy_dir, rel))


def test_write_never_overwrites_an_unreadable_file(copy_dir, shipped):
    rel = cf.path_of("formula", _a_file(shipped, "formula", 3)["id"])
    with open(os.path.join(copy_dir, rel), "w", encoding="utf-8") as f:
        f.write("{not json")
    with pytest.raises(cf.CatalogueFileError, match="unreadable, so not rewritten"):
        cf.write(copy_dir)
    with open(os.path.join(copy_dir, rel), encoding="utf-8") as f:
        assert f.read() == "{not json"


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


def test_problems_names_self_reference_duplicates_and_question_kinds(shipped):
    doc = json.loads(json.dumps(_a_file(shipped, "rag", 0)))
    assert cf.problems(doc) == []
    other = _a_file(shipped, "rag", 1)["id"]
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


def test_path_of_places_each_kind_in_its_directory():
    assert cf.path_of("formula", "formula:a_b") == "formulas/a_b.json"
    assert cf.path_of("rag", "tool:search_x") == "rag/search_x.json"


def test_hint_gaps_lists_files_without_a_written_hint(copy_dir, shipped):
    before = cf.hint_gaps(copy_dir)
    assert len(before) == len(shipped)  # nothing is written yet
    formula, rag = _a_file(shipped, "formula", 0), _a_file(shipped, "rag", 0)
    for doc in (formula, rag):
        changed = json.loads(json.dumps(doc))
        changed["hints"]["purpose"] = "What it is for."
        _write(copy_dir, cf.path_of(doc["kind"], doc["id"]), changed)
    after = cf.hint_gaps(copy_dir)
    # A formula with a purpose is written; a retrieval tool also needs its question kinds.
    assert formula["id"] not in after and rag["id"] in after
    assert len(after) == len(before) - 1


# -- using a file: call_for, hint_text, enrich ------------------------------------------

def test_call_for_builds_the_files_call_and_refuses_by_name(shipped):
    doc = shipped["formula:excavation_volume"]
    for values in ({"length_m": 12, "width_m": 4, "depth_m": 2}, {"length_m": 7, "width_m": 3, "depth_m": 1.5}):
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


def test_hint_text_and_enrich_use_only_an_approved_file_that_agrees(shipped, capabilities):
    entries = {e["id"]: e for _k, e in capabilities}
    files = json.loads(json.dumps(shipped))
    a, b = "formula:excavation_volume", "formula:unit_cost_total"
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
