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
def test_a_pack_off_the_schema_gives_the_schema_fault_first(pack, entries, break_it, where):
    # these faults are invisible to every later check: only the schema sees them
    if break_it == "extra_key":
        pack["notes"] = "free text"
    else:
        pack["formulas"][0]["approved"] = "yes"
    found = kp.problems(pack, entries)
    assert found and all(p == q for p, q in zip(found, kp.schema_problems(pack)))
    assert any(p.startswith(where) for p in found), found


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
