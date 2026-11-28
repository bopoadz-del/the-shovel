"""Kernel packs: the shipped packs agree with the code, and each check of
``kernel_packs`` refuses what it must (every test builds its own input from
the shipped pack and breaks one thing; two inputs per decision)."""
from __future__ import annotations

import copy
import json
import os

import pytest

from app.agents.driver import kernel_packs as kp


@pytest.fixture(scope="module")
def entries():
    return kp._entries()


@pytest.fixture(scope="module")
def shipped():
    return kp.load()


@pytest.fixture()
def pack(shipped):
    return copy.deepcopy(shipped["kernel:quantities"])


def _chain(pack, chain_id):
    return next(c for c in pack["chains"] if c["id"] == chain_id)


def _write_root(tmp_path, packs):
    """A packs directory of its own: the shipped schema and ``packs``."""
    with open(os.path.join(kp.packs_dir(), kp.SCHEMA_FILE), encoding="utf-8") as f:
        (tmp_path / kp.SCHEMA_FILE).write_text(f.read(), encoding="utf-8")
    for pack in packs:
        name = pack["kernel"].split(":", 1)[1] + ".json"
        (tmp_path / name).write_text(json.dumps(pack), encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize("root", ["shipped", "copy"])
def test_shipped_packs_agree_with_the_code(tmp_path, shipped, root):
    where = None if root == "shipped" else _write_root(tmp_path, shipped.values())
    assert kp.check(where) == []


@pytest.mark.parametrize("break_it,expect", [
    ("stale", "formulas/formula:rebar_weight: stale"),
    ("example", "chains/slab_concrete_to_price/example: step"),
])
def test_check_reports_a_stale_line_and_a_wrong_figure(tmp_path, pack, break_it, expect):
    # both are faults ``problems`` does not see: only ``check`` finds them
    if break_it == "stale":
        line = next(x for x in pack["formulas"] if x["id"] == "formula:rebar_weight")
        line["fingerprint"] = "sha256:" + "0" * 16
    else:
        figure = _chain(pack, "slab_concrete_to_price")["example"][-1]
        figure["value"] = round(figure["value"] * 1.5, 2)
    assert kp.problems(pack) == []
    found = kp.check(_write_root(tmp_path, [pack]))
    assert any(p.startswith("kernel:quantities: " + expect) for p in found), found


@pytest.mark.parametrize("part", ["formulas", "skills"])
def test_shipped_pack_has_every_formula_and_skill_of_its_kernel(shipped, entries, part):
    for kernel, pack in shipped.items():
        owned = entries[kernel]["source"][part]
        assert owned, f"{kernel} owns no {part}"
        assert sorted(line["id"] for line in pack[part]) == sorted(owned)
        assert not any(line["approved"] for line in pack[part]), "nothing is approved yet"


@pytest.mark.parametrize("part,index", [("formulas", 0), ("formulas", 5), ("skills", 1)])
def test_a_missing_line_of_an_owned_id_is_refused(pack, entries, part, index):
    gone = pack[part].pop(index)["id"]
    assert any(f"no line for {gone}" in p for p in kp.problems(pack, entries))


@pytest.mark.parametrize("part,stranger", [("formulas", "formula:concrete_volume"),
                                           ("skills", "skill:planning.p1")])
def test_a_line_for_an_id_the_kernel_does_not_own_is_refused(pack, entries, part, stranger):
    if stranger not in entries:
        stranger = next(i for i, e in entries.items() if e["kind"] == part[:-1]
                        and i not in [line["id"] for line in pack[part]])
    pack[part].append(dict(copy.deepcopy(pack[part][0]), id=stranger))
    assert any(f"a line for {stranger}" in p for p in kp.problems(pack, entries))


@pytest.mark.parametrize("where", ["knowledge", "neighbour", "step"])
def test_every_id_a_pack_names_must_exist(pack, entries, where):
    ghost = "formula:no_such_formula_x" if where != "knowledge" else "knowledge:no_such_entry_x"
    if where == "knowledge":
        pack["knowledge"][0]["id"] = ghost
    elif where == "neighbour":
        pack["formulas"][1]["not_to_be_confused_with"][0]["id"] = ghost
    else:
        pack["chains"][1]["steps"][1]["formula"] = ghost
    assert any(ghost in p for p in kp.problems(pack, entries))


@pytest.mark.parametrize("harmless", ["none", "lines_reordered", "all_approved"])
def test_the_shipped_pack_has_no_problem_before_it_is_broken(pack, entries, harmless):
    # the negative of every refusal above: the unbroken input passes, and so
    # do changes no check is about (line order, the approved flag)
    if harmless == "lines_reordered":
        for part in kp.LINE_PARTS:
            pack[part].reverse()
    elif harmless == "all_approved":
        for _part, line in kp._lines(pack):
            line["approved"] = True
    assert kp.problems(pack, entries) == []


@pytest.mark.parametrize("break_it,where", [
    ("extra_key", "pack: Additional properties"),
    ("approved_word", "formulas/0/approved: "),
])
def test_a_pack_off_the_schema_gives_the_schema_fault_alone(pack, entries, break_it, where):
    # a pack off the schema is not read further: a later fault it also has
    # (a missing line of an owned formula) is not reported until the schema
    # fault is put right
    gone = pack["formulas"].pop(3)["id"]
    assert any(f"no line for {gone}" in p for p in kp.problems(pack, entries))
    if break_it == "extra_key":
        pack["notes"] = "free text"
    else:
        pack["formulas"][0]["approved"] = "yes"
    found = kp.problems(pack, entries)
    assert found == kp.schema_problems(pack) and found, found
    assert any(p.startswith(where) for p in found), found
    assert not any(gone in p for p in found), found


@pytest.mark.parametrize("chain_id,step,name,output", [
    ("slab_concrete_to_price", 1, "quantity_m3", "net_volume_m3"),         # m3 -> m3: still agrees
    ("slab_rebar_to_price", 2, "quantity_kg", "total_mass_kg"),             # kg -> kg: agrees
])
def test_linked_units_that_agree_pass(pack, entries, chain_id, step, name, output):
    chain = _chain(pack, chain_id)
    chain["steps"][step]["inputs"][name] = {"from": {"step": step - 1, "output": output}}
    assert kp._chain_problems(chain, entries) == []


@pytest.mark.parametrize("chain_id,step,name,output,unit_out", [
    ("slab_rebar_to_price", 2, "quantity_kg", "total_mass_t", "t"),        # t into kg
    ("slab_concrete_to_price", 1, "quantity_m3", "value", "currency"),     # money into m3
])
def test_linked_units_that_disagree_are_refused(pack, entries, chain_id, step, name, output, unit_out):
    chain = _chain(pack, chain_id)
    chain["steps"][step]["inputs"][name] = {"from": {"step": step - 1, "output": output}}
    found = kp._chain_problems(chain, entries)
    assert any(f"unit {unit_out!r} of {output} does not agree" in p for p in found), found


@pytest.mark.parametrize("chain_id,step,name,output", [
    ("slab_rebar_to_price", 2, "quantity_kg", "quantity"),                 # output with no unit
    ("slab_concrete_to_price", 1, "quantity_m3", "waste_factor"),          # output with no unit
])
def test_a_link_with_an_undeclared_unit_cannot_be_checked(pack, entries, chain_id, step, name, output):
    chain = _chain(pack, chain_id)
    chain["steps"][step]["inputs"][name] = {"from": {"step": step - 1, "output": output}}
    assert any("not declared on both sides" in p for p in kp._chain_problems(chain, entries))


@pytest.mark.parametrize("chain_id,src", [("slab_rebar_to_price", 2), ("slab_concrete_to_price", 1)])
def test_a_link_from_the_same_or_a_later_step_is_refused(pack, entries, chain_id, src):
    chain = _chain(pack, chain_id)
    last = len(chain["steps"]) - 1
    name = next(k for k, v in chain["steps"][last]["inputs"].items() if isinstance(v, dict))
    chain["steps"][last]["inputs"][name]["from"]["step"] = src
    assert any("not an earlier step" in p for p in kp._chain_problems(chain, entries))


@pytest.mark.parametrize("chain_id,bad_input", [("slab_rebar_to_price", "area_mm2"),
                                                ("slab_concrete_to_price", "length_ft")])
def test_an_input_the_formula_does_not_take_is_refused(pack, entries, chain_id, bad_input):
    chain = _chain(pack, chain_id)
    chain["steps"][0]["inputs"][bad_input] = 1
    assert any(f"takes no input {bad_input!r}" in p for p in kp._chain_problems(chain, entries))


@pytest.mark.parametrize("chain_id", ["slab_concrete_to_price", "slab_rebar_to_price"])
def test_a_chain_of_one_formula_is_refused_by_the_schema(pack, chain_id):
    index = [c["id"] for c in pack["chains"]].index(chain_id)
    pack["chains"][index]["steps"] = pack["chains"][index]["steps"][:1]
    found = kp.schema_problems(pack)
    assert [p for p in found if p.startswith("chains/")] and         all(p.startswith(f"chains/{index}/steps") for p in found if p.startswith("chains/")), found


@pytest.mark.parametrize("keep", [0, 1])
def test_a_pack_with_fewer_than_two_chains_is_refused(pack, entries, keep):
    # the design: two or three worked chains per pack; the shipped two pass
    assert kp.schema_problems(pack) == []
    pack["chains"] = pack["chains"][:keep]
    found = kp.problems(pack, entries)
    assert any(p.startswith("chains: ") for p in found), found


@pytest.mark.parametrize("ghost", ["formula:no_such_formula_x", "knowledge:"])
def test_running_a_chain_with_a_step_that_is_not_a_formula_raises(pack, entries, ghost):
    if ghost == "knowledge:":
        ghost = pack["knowledge"][0]["id"]                     # exists, but not a formula
    chain = copy.deepcopy(_chain(pack, "slab_rebar_to_price"))
    chain["steps"][0]["formula"] = ghost
    with pytest.raises(kp.KernelPackError, match=r"steps/0: .* is not a catalogue formula"):
        kp.run_chain(chain, entries)


@pytest.mark.parametrize("kind", ["missing", "a_file"])
def test_load_without_a_packs_directory_raises(tmp_path, kind):
    root = tmp_path / "nowhere"
    if kind == "a_file":
        root.write_text("{}", encoding="utf-8")
    with pytest.raises(kp.KernelPackError, match="no kernel packs directory"):
        kp.load(str(root))


@pytest.mark.parametrize("chain_id", ["slab_concrete_to_price", "slab_rebar_to_price"])
def test_example_figures_are_what_the_formulas_give(pack, entries, chain_id):
    chain = _chain(pack, chain_id)
    assert kp.example_problems(chain, entries) == []
    last = chain["example"][-1]
    last["value"] = round(last["value"] * 1.01, 2)            # one per cent off
    found = kp.example_problems(chain, entries)
    assert len(found) == 1 and f"{last['output']} is written" in found[0]


@pytest.mark.parametrize("chain_id,figure,factor", [("slab_concrete_to_price", 0, 2.0),
                                                     ("slab_rebar_to_price", 1, 0.5)])
def test_a_changed_input_changes_the_figures(pack, entries, chain_id, figure, factor):
    chain = _chain(pack, chain_id)
    first = chain["steps"][0]["inputs"]
    key = next(k for k, v in first.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
               and k not in ("waste_factor", "bar_diameter_mm"))
    first[key] = first[key] * factor
    found = kp.example_problems(chain, entries)
    assert any(f"step {chain['example'][figure]['step']} {chain['example'][figure]['output']}" in p
               for p in found), found


def test_a_refused_step_names_the_step(pack, entries):
    chain = _chain(pack, "slab_rebar_to_price")
    chain["steps"][0]["inputs"]["spacing_mm"] = -5
    found = kp.example_problems(chain, entries)
    assert len(found) == 1 and "steps/0" in found[0]


@pytest.mark.parametrize("written,computed,agrees", [
    (6094.0, 6094.35, True),     # rounded to four significant digits
    (6096.0, 6094.35, False),    # off by more than half the fourth digit
    (1065.38, 1065.38, True),
    (1066.0, 1065.38, False),
    (0.0, 0.0, True),
    (0.1, 0.0, False),
])
def test_rounding_of_a_written_figure(written, computed, agrees):
    assert kp._agrees(written, computed) is agrees


def test_a_figure_that_is_not_a_number_never_agrees():
    assert kp._agrees(1.0, True) is False
    assert kp._agrees(1.0, None) is False


def _changed_facts(entries, target):
    changed = dict(entries)
    entry = copy.deepcopy(entries[target])
    entry["name"] = entry["name"] + " (renamed in code)"
    changed[target] = entry
    return changed


@pytest.mark.parametrize("target,stale_line", [
    ("formula:rebar_weight", "formulas/formula:rebar_weight"),
    ("skill:quantities.q5", "skills/skill:quantities.q5"),
])
def test_a_line_whose_facts_changed_is_stale_and_drops_back(pack, entries, target, stale_line):
    assert kp.stale(pack, entries) == []
    for _part, line in kp._lines(pack):
        line["approved"] = True
    changed = _changed_facts(entries, target)
    gone = kp.stale(pack, changed)
    assert stale_line in gone
    assert stale_line not in kp.approved_lines(pack, changed)
    assert stale_line in kp.approved_lines(pack, entries)
    restamped = kp.stamp(pack, changed)
    assert stale_line in restamped and kp.stale(pack, changed) == []
    assert stale_line not in kp.approved_lines(pack, changed), "a re-stamp clears approved"


def _changed_field(entries, target, field, value):
    changed = dict(entries)
    entry = copy.deepcopy(entries[target])
    entry[field] = value
    changed[target] = entry
    return changed


@pytest.mark.parametrize("target,field,value,stale_line", [
    # what a skill line restates: when the skill is not to be used
    ("skill:quantities.q2", "when_not", "nothing; it re-measures every quantity itself.",
     "skills/skill:quantities.q2"),
    ("skill:quantities.q5", "when_to_use", "another use.", "skills/skill:quantities.q5"),
    # the same fields of a formula and of a knowledge entry
    ("formula:rebar_weight", "when_not", "never.", "formulas/formula:rebar_weight"),
    ("formula:rebar_weight", "when_to_use", "another use.", "formulas/formula:rebar_weight"),
    ("formula:rebar_weight", "standard", "another standard", "formulas/formula:rebar_weight"),
    ("formula:rebar_weight", "examples", [{"said": "x"}], "formulas/formula:rebar_weight"),
    ("knowledge:boq_units_of_measurement", "when_not", "never.", "knowledge/knowledge:boq_units_of_measurement"),
    # what feeds what is a fact of the id, for a skill as it already is for a formula
    ("skill:quantities.q2", "fed_by", ["formula:a_new_formula"], "skills/skill:quantities.q2"),
    ("skill:quantities.q5", "feeds", ["formula:a_new_formula"], "skills/skill:quantities.q5"),
])
def test_a_line_goes_stale_when_anything_the_catalogue_shows_for_its_id_changes(pack, entries, target, field,
                                                                                value, stale_line):
    _approve_all(pack)
    assert entries[target][field] != value
    changed = _changed_field(entries, target, field, value)
    assert stale_line in kp.stale(pack, changed)
    assert stale_line not in kp.approved_lines(pack, changed)
    assert stale_line in kp.approved_lines(pack, entries)
    assert stale_line in kp.stamp(pack, changed), "a re-stamp clears approved"
    assert stale_line not in kp.approved_lines(pack, changed)


@pytest.mark.parametrize("target,field,value", [
    # kept ABOUT the entry: a line is not unapproved by them
    ("formula:rebar_weight", "stamps", {"verified": "x", "tested": "y", "approved": "z"}),
    ("formula:rebar_weight", "gaps", ["another gap"]),
    ("skill:quantities.q2", "stamps", {"verified": "x", "tested": None, "approved": None}),
    ("skill:quantities.q5", "gaps", []),
])
def test_a_line_stays_approved_when_only_what_is_kept_about_its_id_changes(pack, entries, target, field, value):
    _approve_all(pack)
    assert entries[target][field] != value
    changed = _changed_field(entries, target, field, value)
    assert kp.stale(pack, changed) == []
    assert kp.approved_lines(pack, changed) == kp.approved_lines(pack, entries)


def _procedure(hat_name, proc_id):
    from app.agents import catalog

    for hat in catalog.list_hats():
        if hat.discipline.value == hat_name:
            for proc in hat.playbook.procedures:
                if proc.id == proc_id:
                    return proc
    raise AssertionError((hat_name, proc_id))


@pytest.mark.parametrize("skill,field,value", [
    ("q2", "steps", ["re-measure every item from the drawings"]),
    ("q2", "done_when", "never."),
    ("q5", "steps", []),
    ("q5", "refusals", "nothing; it corrects every row itself."),
])
def test_a_skill_line_goes_stale_when_the_procedure_behind_it_changes(monkeypatch, pack, entries, skill, field,
                                                                     value):
    """The manifest's procedure holds more than the catalogue entry shows
    (its steps, when it is done); a skill line restates them."""
    _approve_all(pack)
    where = f"skills/skill:quantities.{skill}"
    proc = _procedure("quantities", skill)
    assert getattr(proc, field) != value and kp.stale(pack, entries) == []
    monkeypatch.setattr(proc, field, value)
    assert kp.stale(pack, entries) == [where]
    assert where not in kp.approved_lines(pack, entries)
    assert kp.stamp(pack, entries) == [where]
    assert where not in kp.approved_lines(pack, entries), "a re-stamp clears approved"


@pytest.mark.parametrize("entry,behind", [
    ({"id": "skill:quantities.q2", "code": {"hat": "quantities", "procedure": "q2"}}, "q2"),
    ({"id": "skill:quantities.q5", "code": {"hat": "quantities", "procedure": "q5"}}, "q5"),
    ({"id": "formula:rebar_weight", "code": {"tool": "construction_calc"}}, None),
    ({"id": "x", "code": None}, None),
    ({"id": "x"}, None),
])
def test_procedure_behind_is_the_manifests_procedure_whole_or_none(entry, behind):
    found = kp.procedure_behind(entry)
    if behind is None:
        assert found is None
    else:
        proc = _procedure("quantities", behind)
        assert found == proc.model_dump(mode="json")
        assert found["id"] == behind and found["steps"] and found["refusals"]


@pytest.mark.parametrize("code", [{"hat": "quantities", "procedure": "no_such_procedure"},
                                  {"hat": "no_such_hat", "procedure": "q2"},
                                  {"procedure": "q2"}])
def test_procedure_behind_refuses_a_procedure_the_hats_do_not_hold(code):
    with pytest.raises(kp.KernelPackError, match="the hats hold no procedure"):
        kp.procedure_behind({"id": "skill:x", "code": code})


@pytest.mark.parametrize("target", ["formula:rebar_weight", "skill:quantities.q2",
                                    "knowledge:boq_units_of_measurement", "kernel:quantities"])
def test_code_facts_hold_everything_the_catalogue_shows_for_the_id(entries, target):
    entry = entries[target]
    facts = kp.code_facts(entry)
    assert set(facts["catalogue"]) == set(entry) - set(kp.NOT_FACTS)
    assert all(facts["catalogue"][k] == json.loads(json.dumps(entry[k])) for k in facts["catalogue"])
    assert ("procedure" in facts) == (entry["kind"] == "skill")
    assert facts["inputs"] == json.loads(json.dumps(entry["inputs"]))


def test_a_chain_goes_stale_when_any_of_its_formulas_changes(pack, entries):
    changed = _changed_facts(entries, "formula:cost_buildup_rebar")
    gone = kp.stale(pack, changed)
    assert "chains/slab_rebar_to_price" in gone
    assert "chains/slab_concrete_to_price" not in gone


@pytest.mark.parametrize("ids", [["formula:rebar_weight"], ["formula:rebar_by_area", "formula:rebar_weight"]])
def test_fingerprint_follows_the_facts(entries, ids):
    first = kp.fingerprint(entries, ids)
    assert first == kp.fingerprint(entries, ids)
    assert first != kp.fingerprint(_changed_facts(entries, ids[-1]), ids)


def test_fingerprint_of_an_unknown_id_raises(entries):
    with pytest.raises(kp.KernelPackError):
        kp.fingerprint(entries, ["formula:no_such_formula_x"])


@pytest.mark.parametrize("name,body", [("quantities.json", {"schema": "kernel-pack/1"}),
                                       ("planning.json", None)])
def test_load_refuses_a_broken_or_misplaced_pack(tmp_path, shipped, name, body):
    shutil_src = kp.packs_dir()
    with open(os.path.join(shutil_src, kp.SCHEMA_FILE), encoding="utf-8") as f:
        (tmp_path / kp.SCHEMA_FILE).write_text(f.read(), encoding="utf-8")
    doc = body if body is not None else shipped["kernel:quantities"]
    (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(kp.KernelPackError) as err:
        kp.load(str(tmp_path))
    assert name in str(err.value)


# -- neighbours are of the line's kind (KIND_OF) ---------------------------------------

@pytest.mark.parametrize("part,index,neighbour,kind", [
    ("formulas", 0, "kernel:quantities", "formula"),
    ("formulas", 0, "skill:quantities.q2", "formula"),
    ("formulas", 3, "knowledge:boq_units_of_measurement", "formula"),
    ("knowledge", 0, "formula:rebar_weight", "knowledge"),
    ("skills", 1, "formula:rebar_weight", "skill"),
])
def test_a_neighbour_of_another_kind_is_refused(pack, entries, part, index, neighbour, kind):
    line = pack[part][index]
    line["not_to_be_confused_with"] = [{"id": neighbour, "difference": "one difference."}]
    found = kp.problems(pack, entries)
    assert f"{part}/{line['id']}: neighbour {neighbour!r} is not a catalogue {kind}" in found, found


@pytest.mark.parametrize("part,index,neighbour", [
    ("knowledge", 0, "knowledge:earthworks.swelling_factor"),
    ("skills", 0, "skill:quantities.q5"),
    ("formulas", 2, "formula:rebar_weight"),
])
def test_a_neighbour_of_the_lines_kind_passes(pack, entries, part, index, neighbour):
    pack[part][index]["not_to_be_confused_with"] = [{"id": neighbour, "difference": "one difference."}]
    assert kp.problems(pack, entries) == []


@pytest.mark.parametrize("part,index", [("formulas", 4), ("knowledge", 1)])
def test_a_line_naming_itself_as_a_neighbour_is_refused(pack, entries, part, index):
    line = pack[part][index]
    line["not_to_be_confused_with"] = [{"id": line["id"], "difference": "one difference."}]
    assert f"{part}/{line['id']}: names itself as a neighbour" in kp.problems(pack, entries)


# -- what a line was approved as: its words and every id it names ------------------------

def _approve_all(pack):
    for _part, line in kp._lines(pack):
        line["approved"] = True


@pytest.mark.parametrize("changed_id,line_id,untouched", [
    ("formula:rebar_weight", "formulas/formula:cost_buildup_rebar", "formulas/formula:crane_cost_estimate"),
    ("formula:crane_cost_estimate", "formulas/formula:mobilization_cost_estimate",
     "formulas/formula:cost_buildup_rebar"),
])
def test_a_line_drops_back_when_a_neighbour_it_names_changes(pack, entries, changed_id, line_id, untouched):
    _approve_all(pack)
    changed = _changed_facts(entries, changed_id)
    assert line_id in kp.stale(pack, changed)
    assert line_id not in kp.approved_lines(pack, changed)
    # a line that does not name the changed formula stays approved
    assert untouched in kp.approved_lines(pack, changed)


def _edit(pack, what):
    if what == "purpose":
        pack["formulas"][2]["purpose"] += " Edited."
        return "formulas/" + pack["formulas"][2]["id"]
    if what == "difference":
        pack["formulas"][0]["not_to_be_confused_with"][0]["difference"] += " Edited."
        return "formulas/" + pack["formulas"][0]["id"]
    if what == "skill_purpose":
        pack["skills"][0]["purpose"] = "Another purpose."
        return "skills/" + pack["skills"][0]["id"]
    chain = _chain(pack, "slab_rebar_to_price")
    if what == "chain_input":
        # a real edit: another input, and the example re-worked to what the formulas give for it
        chain["steps"][0]["inputs"]["area_m2"] = 121
        results = kp.run_chain(chain)
        for figure in chain["example"]:
            figure["value"] = round(results[figure["step"]][figure["output"]], 4)
    else:
        # the figure re-written, still what the formulas give to SIGNIFICANT_DIGITS
        chain["example"][0]["value"] = 1200.2
    return "chains/slab_rebar_to_price"


@pytest.mark.parametrize("what", ["purpose", "difference", "skill_purpose", "chain_input", "chain_example"])
def test_an_approved_line_whose_words_change_drops_back_until_re_stamped(pack, entries, what):
    _approve_all(pack)
    assert kp.stale(pack, entries) == [], "approving alone changes no fingerprint"
    where = _edit(pack, what)
    assert kp.stale(pack, entries) == [where]
    assert where not in kp.approved_lines(pack, entries)
    assert kp.stamp(pack, entries) == [where]
    assert kp.stale(pack, entries) == [] and where not in kp.approved_lines(pack, entries)
    for part, line in kp._lines(pack):
        if f"{part}/{line['id']}" == where:
            line["approved"] = True
    assert where in kp.approved_lines(pack, entries), "re-approved after it was read again"


@pytest.mark.parametrize("what", ["purpose", "difference"])
def test_check_names_a_line_whose_words_changed(tmp_path, pack, what):
    where = _edit(pack, what)
    found = kp.check(_write_root(tmp_path, [pack]))
    assert found == [f"kernel:quantities: {where}: stale (its words or the code facts it names "
                     "changed since it was stamped; read it again and re-stamp)"], found


@pytest.mark.parametrize("ghost_part,ghost", [("knowledge", "knowledge:no_such_x"),
                                              ("formulas", "formula:no_such_x")])
def test_stale_lists_a_line_naming_an_id_the_code_lacks_without_raising(pack, entries, ghost_part, ghost):
    pack[ghost_part][0]["id"] = ghost
    assert f"{ghost_part}/{ghost}" in kp.stale(pack, entries)


def test_a_formula_gone_from_the_code_takes_its_line_its_neighbours_and_chain(pack, entries):
    fewer = {i: e for i, e in entries.items() if i != "formula:rebar_weight"}
    gone = kp.stale(pack, fewer)
    for where in ("formulas/formula:rebar_weight", "formulas/formula:cost_buildup_rebar",
                  "formulas/formula:rebar_by_area", "chains/slab_rebar_to_price"):
        assert where in gone
    assert "chains/slab_concrete_to_price" not in gone


# -- stamp writes nothing for a pack with a fault ------------------------------------------

@pytest.mark.parametrize("ghost_at", ["step", "neighbour"])
def test_stamp_refuses_a_pack_with_a_fault_and_leaves_it_as_it_was(pack, entries, ghost_at):
    if ghost_at == "step":
        _chain(pack, "slab_rebar_to_price")["steps"][1]["formula"] = "formula:rebar_weight_gone"
    else:
        pack["formulas"][1]["not_to_be_confused_with"][0]["id"] = "formula:rebar_weight_gone"
    pack["formulas"][0]["fingerprint"] = "sha256:" + "0" * 16
    before = copy.deepcopy(pack)
    with pytest.raises(kp.KernelPackError) as err:
        kp.stamp(pack, entries)
    assert "formula:rebar_weight_gone" in str(err.value)
    assert pack == before


def _run_cli(monkeypatch, root, argv):
    import importlib.util

    script = os.path.join(os.path.dirname(kp.packs_dir()), "scripts", "check_kernel_packs.py")
    spec = importlib.util.spec_from_file_location("check_kernel_packs_under_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(kp, "packs_dir", lambda: root)
    return mod.main(argv)


@pytest.mark.parametrize("broken", ["wrong_kernel", "ghost_step"])
def test_cli_stamp_lists_faults_and_writes_no_pack(tmp_path, monkeypatch, capsys, pack, broken):
    good = copy.deepcopy(pack)
    good["formulas"][0]["fingerprint"] = "sha256:" + "0" * 16      # would be re-stamped
    bad = copy.deepcopy(pack)
    bad["kernel"] = "kernel:safety"                                # its file sorts after quantities
    if broken == "ghost_step":
        _chain(bad, "slab_rebar_to_price")["steps"][1]["formula"] = "formula:rebar_weight_gone"
    root = _write_root(tmp_path, [good, bad])
    before = {n: (tmp_path / n).read_bytes() for n in ("quantities.json", "safety.json")}
    assert _run_cli(monkeypatch, root, ["--stamp"]) == 1
    out = capsys.readouterr().out
    assert "kernel:safety: " in out and "no pack was stamped" in out and "Traceback" not in out
    assert {n: (tmp_path / n).read_bytes() for n in before} == before


@pytest.mark.parametrize("index", [0, 6])
def test_cli_stamp_rewrites_a_clean_pack(tmp_path, monkeypatch, capsys, pack, index):
    pack["formulas"][index]["fingerprint"] = "sha256:" + "0" * 16
    pack["formulas"][index]["approved"] = True
    root = _write_root(tmp_path, [pack])
    assert _run_cli(monkeypatch, root, ["--stamp"]) == 0
    written = json.loads((tmp_path / "quantities.json").read_text(encoding="utf-8"))
    assert written["formulas"][index]["approved"] is False
    assert kp.check(root) == []


@pytest.mark.parametrize("body", ["{not json", "[]"])
def test_cli_stamp_on_a_pack_it_cannot_load_exits_1_without_a_traceback(tmp_path, monkeypatch, capsys, body):
    root = _write_root(tmp_path, [])
    (tmp_path / "quantities.json").write_text(body, encoding="utf-8")
    assert _run_cli(monkeypatch, root, ["--stamp"]) == 1
    out = capsys.readouterr().out
    assert "quantities.json" in out and "not stamped" in out


# -- an example figure the formulas no longer give: stamp, --stamp and approved_lines -------

def _off_by_one_percent(chain):
    figure = chain["example"][-1]
    figure["value"] = round(figure["value"] * 1.01, 4)
    return figure["output"]


@pytest.mark.parametrize("chain_id", ["slab_concrete_to_price", "slab_rebar_to_price"])
def test_faults_holds_a_wrong_example_figure_and_a_clean_pack_has_none(pack, entries, chain_id):
    assert kp.faults(pack, entries) == []
    output = _off_by_one_percent(_chain(pack, chain_id))
    found = kp.faults(pack, entries)
    assert len(found) == 1 and f"chains/{chain_id}/example: step" in found[0] and output in found[0], found


@pytest.mark.parametrize("ghost_at", ["step", "neighbour"])
def test_faults_of_a_pack_with_problems_are_the_problems_only(pack, entries, ghost_at):
    if ghost_at == "step":
        _chain(pack, "slab_rebar_to_price")["steps"][1]["formula"] = "formula:rebar_weight_gone"
    else:
        pack["formulas"][1]["not_to_be_confused_with"][0]["id"] = "formula:rebar_weight_gone"
    assert kp.faults(pack, entries) == kp.problems(pack, entries) != []


@pytest.mark.parametrize("chain_id", ["slab_concrete_to_price", "slab_rebar_to_price"])
def test_stamp_refuses_a_wrong_example_figure_and_leaves_the_pack_as_it_was(pack, entries, chain_id):
    pack["formulas"][0]["fingerprint"] = "sha256:" + "0" * 16       # would be re-stamped
    _off_by_one_percent(_chain(pack, chain_id))
    before = copy.deepcopy(pack)
    with pytest.raises(kp.KernelPackError) as err:
        kp.stamp(pack, entries)
    assert f"chains/{chain_id}/example" in str(err.value)
    assert pack == before


@pytest.mark.parametrize("chain_id", ["slab_concrete_to_price", "slab_rebar_to_price"])
def test_cli_stamp_with_a_wrong_example_figure_exits_1_and_writes_nothing(tmp_path, monkeypatch, capsys,
                                                                         pack, chain_id):
    pack["formulas"][0]["fingerprint"] = "sha256:" + "0" * 16
    _off_by_one_percent(_chain(pack, chain_id))
    root = _write_root(tmp_path, [pack])
    before = (tmp_path / "quantities.json").read_bytes()
    assert _run_cli(monkeypatch, root, ["--stamp"]) == 1
    out = capsys.readouterr().out
    assert f"chains/{chain_id}/example" in out and "no pack was stamped" in out
    assert (tmp_path / "quantities.json").read_bytes() == before


def _scale_output(monkeypatch, entries, formula, output, factor):
    """The calculation behind ``formula`` gives ``output`` times ``factor``;
    its catalogue facts (and so every fingerprint) stay as they were."""
    from app.lib import construction_formulas

    real = construction_formulas.run_calculation
    name = entries[formula]["code"]["calculation"]

    def changed(calc, params, *a, **kw):
        envelope = real(calc, params, *a, **kw)
        if calc == name and isinstance(envelope.get("result"), dict) and output in envelope["result"]:
            envelope["result"][output] = envelope["result"][output] * factor
        return envelope

    monkeypatch.setattr(construction_formulas, "run_calculation", changed)


@pytest.mark.parametrize("formula,output,dropped,kept,line", [
    ("formula:rebar_weight", "total_mass_kg", "chains/slab_rebar_to_price", "chains/slab_concrete_to_price",
     "formulas/formula:rebar_weight"),
    ("formula:concrete_volume", "net_volume_m3", "chains/slab_concrete_to_price", "chains/slab_rebar_to_price",
     "formulas/formula:cost_buildup_concrete"),
])
def test_a_chain_whose_calculation_changed_is_no_longer_approved_and_is_not_stamped(
        monkeypatch, pack, entries, formula, output, dropped, kept, line):
    for _, each in kp._lines(pack):
        each["approved"] = True
    assert dropped in kp.approved_lines(pack, entries)
    _scale_output(monkeypatch, entries, formula, output, 1.10)
    assert kp.stale(pack, entries) == []                # the facts did not change ...
    approved = kp.approved_lines(pack, entries)
    assert dropped not in approved and kept in approved  # ... but the figures did
    assert line in approved                              # a formula line carries no figure
    with pytest.raises(kp.KernelPackError) as err:
        kp.stamp(pack, entries)
    assert dropped + "/example" in str(err.value)


@pytest.mark.parametrize("part,line,ids", [
    ("chains", {"id": "c", "steps": [{"formula": "formula:a"}, {"formula": "formula:b"},
                                     {"formula": "formula:a"}]}, ["formula:a", "formula:b"]),
    ("formulas", {"id": "formula:a", "not_to_be_confused_with": [{"id": "formula:b"}, {"id": "formula:b"},
                                                                 {"id": "formula:a"}]},
     ["formula:a", "formula:b"]),
])
def test_a_line_names_each_id_once_in_the_order_first_named(part, line, ids):
    assert kp._line_ids(part, line) == ids


# -- check reports a directory it cannot read, never raises, never passes empty -----------

@pytest.mark.parametrize("name,body,expect", [
    ("quantities.json", "{not json", "packs: quantities.json: unreadable"),
    ("README.md", "notes", "packs: README.md: not a pack"),
    ("notes.txt", "notes", "packs: notes.txt: not a pack"),
])
def test_check_reports_an_unreadable_or_stray_file(tmp_path, shipped, name, body, expect):
    root = _write_root(tmp_path, [] if name == "quantities.json" else shipped.values())
    (tmp_path / name).write_text(body, encoding="utf-8")
    found = kp.check(root)
    assert any(p.startswith(expect) for p in found), found
    with pytest.raises(kp.KernelPackError, match=name.replace(".", r"\.")):
        kp.load(root)


@pytest.mark.parametrize("with_schema", [True, False])
def test_check_of_a_directory_with_no_pack_is_a_fault(tmp_path, shipped, with_schema):
    empty, full = tmp_path / "empty", tmp_path / "full"
    empty.mkdir()
    full.mkdir()
    root = _write_root(empty, []) if with_schema else str(empty)
    assert kp.check(root) == [f"packs: no kernel pack in {root}"]
    assert kp.check(_write_root(full, shipped.values())) == []


# -- each refusal of ``problems`` and ``_chain_problems``, two inputs each ---------------

@pytest.mark.parametrize("part,stranger,kind", [
    ("knowledge", "formula:rebar_weight", "knowledge"),
    ("knowledge", "skill:quantities.q5", "knowledge"),
    ("skills", "knowledge:boq_units_of_measurement", "skill"),
])
def test_a_line_of_another_kind_is_refused(pack, entries, part, stranger, kind):
    pack[part].append(dict(copy.deepcopy(pack[part][0]), id=stranger, not_to_be_confused_with=[]))
    assert f"{part}/{stranger}: not a catalogue {kind}" in kp.problems(pack, entries)


@pytest.mark.parametrize("part,index", [("formulas", 0), ("skills", 1)])
def test_a_line_given_twice_is_refused(pack, entries, part, index):
    pack[part].append(copy.deepcopy(pack[part][index]))
    assert f"{part}: more than one line for {pack[part][index]['id']}" in kp.problems(pack, entries)


@pytest.mark.parametrize("keep", [0, 1])
def test_two_chains_sharing_an_id_are_refused(pack, entries, keep):
    pack["chains"][1 - keep]["id"] = pack["chains"][keep]["id"]
    assert "chains: two chains share an id" in kp.problems(pack, entries)


@pytest.mark.parametrize("kernel", ["formula:rebar_weight", "kernel:nope"])
def test_a_pack_for_something_that_is_not_a_kernel_is_refused(pack, entries, kernel):
    pack["kernel"] = kernel
    assert kp.problems(pack, entries) == [f"kernel: {kernel!r} is not a discipline kernel"]


@pytest.mark.parametrize("chain_id,step", [("slab_rebar_to_price", 3), ("slab_concrete_to_price", 2),
                                           ("slab_rebar_to_price", 7)])
def test_an_example_naming_a_step_the_chain_lacks_is_refused(pack, entries, chain_id, step):
    _chain(pack, chain_id)["example"][0]["step"] = step
    assert (f"chains/{chain_id}/example: names step {step}, which the chain does not have"
            in kp.problems(pack, entries))


@pytest.mark.parametrize("chain_id,output", [("slab_rebar_to_price", "nope"),
                                             ("slab_concrete_to_price", "bars_per_mm")])
def test_an_example_naming_an_output_its_step_lacks_is_refused(pack, entries, chain_id, output):
    _chain(pack, chain_id)["example"][0]["output"] = output
    assert f"chains/{chain_id}/example: step 0 gives no output {output!r}" in kp.problems(pack, entries)


@pytest.mark.parametrize("chain_id,step,name,formula", [
    ("slab_rebar_to_price", 1, "total_length_m", "formula:rebar_weight"),
    ("slab_concrete_to_price", 1, "quantity_m3", "formula:cost_buildup_concrete"),
])
def test_a_link_into_an_input_with_no_declared_unit_cannot_be_checked(monkeypatch, pack, entries,
                                                                     chain_id, step, name, formula):
    real = kp.code_facts

    def blank_input_unit(entry):
        facts = real(entry)
        if entry["id"] == formula:
            for item in facts["inputs"]:
                if item["name"] == name:
                    item["unit"] = ""
        return facts

    monkeypatch.setattr(kp, "code_facts", blank_input_unit)
    found = kp.problems(pack, entries)
    assert any(p.startswith(f"chains/{chain_id}/steps/{step}/{name}: the unit is not declared on both sides")
               for p in found), found


@pytest.mark.parametrize("chain_id,step", [("slab_rebar_to_price", 1), ("slab_concrete_to_price", 1)])
def test_running_a_link_from_an_output_the_step_did_not_give_raises(pack, entries, chain_id, step):
    chain = _chain(pack, chain_id)
    for value in chain["steps"][step]["inputs"].values():
        if isinstance(value, dict):
            value["from"]["output"] = "nope"
    with pytest.raises(kp.KernelPackError, match=r"step 0 gave no 'nope'"):
        kp.run_chain(chain, entries)


@pytest.mark.parametrize("written,computed,agrees", [
    (6094.85, 6094.35, True),       # exactly half a unit of the fourth digit
    (6094.86, 6094.35, False),
    (6094.5, 6094.0, True),
    (6094.5001, 6094.0, False),
    (float("inf"), float("inf"), False),
    (float("nan"), float("nan"), False),
    (1.0, float("nan"), False),
    (1.0, float("inf"), False),
])
def test_rounding_boundary_and_figures_that_are_not_finite(written, computed, agrees):
    assert kp._agrees(written, computed) is agrees
