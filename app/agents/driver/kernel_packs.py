"""One pack per discipline kernel (``catalogue_packs/<kernel>.json``): what the
model reads when it takes that kernel.

A pack is hand-written data in four parts, each line carrying ``approved``:

* ``formulas``: one line per formula the kernel owns (what it is for, the
  neighbours it must not be confused with and the one difference);
* ``knowledge``: knowledge entries to read alongside the kernel's formulas,
  one line each;
* ``skills``: one line per procedure the kernel's manifest declares;
* ``chains``: worked chains, each two or more formula steps, which output
  feeds which input, and one worked example's figures.

Code checks a pack, it never writes a line's words (``problems``):
every formula and skill the kernel owns has its line and nothing else does;
every id a pack names is a catalogue entry of the kind its place takes
(``KIND_OF``); a chain's linked output and input carry the same declared
unit; and the example's figures are what the formulas give when the chain
is run (``run_chain``). Each line also carries a fingerprint of what it was
approved as (``line_fingerprint``): its own words and the code facts of
every id it names (its own, its neighbours', its chain's formulas). A line
whose words or named facts have changed since is ``stale`` and counts as
not approved (``approved_lines``) until it is read again and re-stamped
(``stamp``, which also clears its ``approved``). Nothing here reads a
question or picks a formula.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_NAME = "kernel-pack/1"
SCHEMA_FILE = "schema.json"
DIR_NAME = "catalogue_packs"

#: The pack's parts that hold one line per catalogue id, and the field of
#: that id. A chain's fingerprint covers every formula among its steps.
LINE_PARTS = ("formulas", "knowledge", "skills")

#: The catalogue kind each part's lines are about. A line's neighbours
#: ("not to be confused with") are of the same kind as the line: a formula
#: is confused with another formula, a skill with another skill.
KIND_OF = {"formulas": "formula", "knowledge": "knowledge", "skills": "skill"}

#: The fields of a line that are not its words: the stamp itself and the
#: owner's flag. Every other field is part of what was approved.
NOT_WORDS = ("fingerprint", "approved")

#: How close a computed example figure must be to the written one: the
#: written figure may be rounded to this many significant digits.
SIGNIFICANT_DIGITS = 4


class KernelPackError(ValueError):
    """A pack, or the set of them, is not what the schema and the code say."""


def packs_dir() -> str:
    """The directory the shipped packs live in."""
    from app.agents.driver import catalogue

    return os.path.join(catalogue._ROOT, DIR_NAME)


# -- what the code says ------------------------------------------------------------

def _entries() -> Dict[str, Dict[str, Any]]:
    """Every catalogue entry and every kernel entry, by id."""
    from app.agents.driver import catalogue, catalogue_files

    out = {e["id"]: e for e in catalogue.entries()}
    out.update({e["id"]: e for e in catalogue_files.kernel_entries()})
    return out


def code_facts(entry: Dict[str, Any]) -> Dict[str, Any]:
    """What a line about ``entry`` is written against: for a formula,
    knowledge entry or kernel the facts of its catalogue file; for anything
    else the entry's name, purpose, inputs, outputs, call and source."""
    from app.agents.driver import catalogue_files

    kind = entry["kind"]
    if kind in ("formula", "knowledge", "kernel"):
        return catalogue_files.facts(kind, entry)
    keys = ("name", "for", "inputs", "outputs", "code", "source")
    return json.loads(json.dumps({k: entry.get(k) for k in keys}, allow_nan=False))


def fingerprint(entries: Dict[str, Dict[str, Any]], ids: List[str],
                words: Optional[Dict[str, Any]] = None) -> str:
    """The fingerprint of the code facts of ``ids`` (in the order given)
    and of ``words`` (a line's own fields): ``sha256:`` and the first 16
    hex digits of the hash of their canonical JSON. An id with no entry
    raises ``KernelPackError``."""
    missing = [i for i in ids if i not in entries]
    if missing:
        raise KernelPackError(f"no catalogue entry for {', '.join(missing)}")
    blob = json.dumps([[[i, code_facts(entries[i])] for i in ids], words], sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _line_ids(part: str, line: Dict[str, Any]) -> List[str]:
    """The ids whose facts a line is written against: a chain's formulas;
    for any other line its own id and each neighbour it names (the
    neighbour's ``difference`` states a fact about that neighbour)."""
    if part == "chains":
        named = [step["formula"] for step in line["steps"]]
    else:
        named = [line["id"]] + [n["id"] for n in line.get("not_to_be_confused_with") or []]
    seen: List[str] = []
    for i in named:
        if i not in seen:
            seen.append(i)
    return seen


def _words(line: Dict[str, Any]) -> Dict[str, Any]:
    """A line's own words: every field but ``NOT_WORDS``."""
    return {k: v for k, v in line.items() if k not in NOT_WORDS}


def line_fingerprint(entries: Dict[str, Dict[str, Any]], part: str, line: Dict[str, Any]) -> str:
    """What a line is stamped with: its words and the facts of the ids it names."""
    return fingerprint(entries, _line_ids(part, line), _words(line))


def _lines(pack: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return [(part, line) for part in LINE_PARTS + ("chains",) for line in pack.get(part) or []]


# -- reading a pack -----------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _validator() -> Any:
    import jsonschema

    with open(os.path.join(packs_dir(), SCHEMA_FILE), encoding="utf-8") as f:
        return jsonschema.Draft7Validator(json.load(f))


def schema_problems(pack: Any) -> List[str]:
    """Why ``pack`` is not of the pack schema, each as ``where: what``."""
    return [f"{'/'.join(str(p) for p in e.absolute_path) or 'pack'}: {e.message[:200]}"
            for e in sorted(_validator().iter_errors(pack),
                            key=lambda e: [str(p) for p in e.absolute_path])]


def _unit_of(facts: Dict[str, Any], side: str, name: str) -> Optional[str]:
    for item in facts[side]:
        if item["name"] == name:
            return str(item.get("unit") or "")
    return None


def _chain_problems(chain: Dict[str, Any], entries: Dict[str, Dict[str, Any]]) -> List[str]:
    """A chain's links against the formulas' call cards: each step a
    formula; each input one the formula takes; each link from an earlier
    step's output that formula gives, with the same declared unit on both
    sides; and the example's figures named by step and output."""
    where = f"chains/{chain['id']}"
    out: List[str] = []
    cards: List[Optional[Dict[str, Any]]] = []
    for n, step in enumerate(chain["steps"]):
        entry = entries.get(step["formula"])
        if entry is None or entry["kind"] != "formula":
            out.append(f"{where}/steps/{n}: {step['formula']!r} is not a catalogue formula")
            cards.append(None)
            continue
        card = code_facts(entry)
        cards.append(card)
        for name, value in step["inputs"].items():
            unit_in = _unit_of(card, "inputs", name)
            if unit_in is None:
                out.append(f"{where}/steps/{n}: {step['formula']} takes no input {name!r}")
                continue
            if not isinstance(value, dict):
                continue
            src, output = value["from"]["step"], value["from"]["output"]
            if not 0 <= src < n:
                out.append(f"{where}/steps/{n}/{name}: fed from step {src}, which is not an earlier step")
                continue
            if cards[src] is None:
                continue
            unit_out = _unit_of(cards[src], "outputs", output)
            if unit_out is None:
                out.append(f"{where}/steps/{n}/{name}: step {src} "
                           f"({chain['steps'][src]['formula']}) gives no output {output!r}")
            elif not unit_out or not unit_in:
                out.append(f"{where}/steps/{n}/{name}: the unit is not declared on both sides "
                           f"({output} {unit_out!r} -> {name} {unit_in!r}), so it cannot be checked")
            elif unit_out != unit_in:
                out.append(f"{where}/steps/{n}/{name}: unit {unit_out!r} of {output} does not "
                           f"agree with unit {unit_in!r} of {name}")
    for figure in chain["example"]:
        n = figure["step"]
        if not 0 <= n < len(cards):
            out.append(f"{where}/example: names step {n}, which the chain does not have")
        elif cards[n] is not None and _unit_of(cards[n], "outputs", figure["output"]) is None:
            out.append(f"{where}/example: step {n} gives no output {figure['output']!r}")
    return out


def problems(pack: Any, entries: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """Why ``pack`` is not a valid pack of its kernel, each as ``where:
    what``; empty when it is valid. The schema first; then: the kernel is a
    kernel; its formulas and skills each have exactly one line and no
    other formula or skill has one; every id named (a line, a neighbour, a
    chain step) is a catalogue entry of the part's kind; a chain's links
    agree in unit (see ``_chain_problems``). Whether a chain's example
    figures are what the formulas give is ``example_problems``."""
    found = schema_problems(pack)
    if found:
        return found
    entries = entries if entries is not None else _entries()
    kernel = entries.get(pack["kernel"])
    if kernel is None or kernel["kind"] != "kernel":
        return [f"kernel: {pack['kernel']!r} is not a discipline kernel"]
    owned = {"formulas": kernel["source"]["formulas"], "skills": kernel["source"]["skills"]}
    kind_of = KIND_OF
    for part in LINE_PARTS:
        ids = [line["id"] for line in pack[part]]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            found.append(f"{part}: more than one line for {', '.join(dupes)}")
        for i in ids:
            entry = entries.get(i)
            if entry is None or entry["kind"] != kind_of[part]:
                found.append(f"{part}/{i}: not a catalogue {kind_of[part]}")
        if part in owned:
            missing = [i for i in owned[part] if i not in ids]
            extra = [i for i in ids if i not in owned[part]]
            if missing:
                found.append(f"{part}: no line for {', '.join(missing)}, which the kernel owns")
            if extra:
                found.append(f"{part}: a line for {', '.join(extra)}, which the kernel does not own")
        for line in pack[part]:
            for neighbour in line.get("not_to_be_confused_with") or []:
                other = entries.get(neighbour["id"])
                if neighbour["id"] == line["id"]:
                    found.append(f"{part}/{line['id']}: names itself as a neighbour")
                elif other is None or other["kind"] != kind_of[part]:
                    found.append(f"{part}/{line['id']}: neighbour {neighbour['id']!r} "
                                 f"is not a catalogue {kind_of[part]}")
    chain_ids = [c["id"] for c in pack["chains"]]
    if len(set(chain_ids)) != len(chain_ids):
        found.append("chains: two chains share an id")
    for chain in pack["chains"]:
        found.extend(_chain_problems(chain, entries))
    return found


def stale(pack: Dict[str, Any], entries: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """The lines whose fingerprint is not that of their words and named
    code facts now (the line was edited, or an id it names changed its
    facts in code, since it was stamped), each as ``part/id``. A line
    naming an id the code no longer has is stale. A stale line counts as
    not approved."""
    entries = entries if entries is not None else _entries()
    out = []
    for part, line in _lines(pack):
        ids = _line_ids(part, line)
        if any(i not in entries for i in ids) or line["fingerprint"] != line_fingerprint(entries, part, line):
            out.append(f"{part}/{line['id']}")
    return out


def approved_lines(pack: Dict[str, Any],
                   entries: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """``part/id`` of every line that is approved and not stale."""
    gone = set(stale(pack, entries))
    return [f"{part}/{line['id']}" for part, line in _lines(pack)
            if line["approved"] and f"{part}/{line['id']}" not in gone]


# -- running a chain ----------------------------------------------------------------

def run_chain(chain: Dict[str, Any],
              entries: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Run a chain's steps in order with the calculator every formula call
    goes through, by the calculation each formula's catalogue entry names,
    each linked input taking the earlier step's output. Returns each step's
    result (the calculator's ``result``). A step that is not a formula, or
    that the calculator refuses, raises ``KernelPackError`` naming it."""
    from app.lib.construction_formulas import run_calculation

    entries = entries if entries is not None else _entries()
    results: List[Dict[str, Any]] = []
    for n, step in enumerate(chain["steps"]):
        entry = entries.get(step["formula"])
        if entry is None or entry["kind"] != "formula":
            raise KernelPackError(f"chains/{chain['id']}/steps/{n}: {step['formula']!r} is not a catalogue formula")
        params = {}
        for name, value in step["inputs"].items():
            if isinstance(value, dict):
                src = value["from"]
                if not 0 <= src["step"] < n or src["output"] not in results[src["step"]]:
                    raise KernelPackError(f"chains/{chain['id']}/steps/{n}/{name}: "
                                          f"step {src['step']} gave no {src['output']!r}")
                value = results[src["step"]][src["output"]]
            params[name] = value
        envelope = run_calculation(entry["code"]["calculation"], copy.deepcopy(params))
        if envelope.get("status") != "success" or not isinstance(envelope.get("result"), dict):
            raise KernelPackError(f"chains/{chain['id']}/steps/{n}: {step['formula']} did not run: "
                                  f"{envelope.get('error') or envelope.get('status')}")
        results.append(envelope["result"])
    return results


def _agrees(written: float, computed: Any) -> bool:
    if not isinstance(computed, (int, float)) or isinstance(computed, bool):
        return False
    if not math.isfinite(written):
        return False
    if written == computed:
        return True
    if computed == 0 or not math.isfinite(computed):
        return False
    scale = 10 ** (math.floor(math.log10(abs(computed))) - SIGNIFICANT_DIGITS + 1)
    return abs(written - computed) <= scale / 2


def example_problems(chain: Dict[str, Any],
                     entries: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """Where a chain's written example figures are not what running the
    chain gives (to ``SIGNIFICANT_DIGITS``); a chain that does not run is
    one problem naming the step."""
    where = f"chains/{chain['id']}/example"
    try:
        results = run_chain(chain, entries)
    except KernelPackError as exc:
        return [str(exc)]
    out = []
    for figure in chain["example"]:
        n, name = figure["step"], figure["output"]
        computed = results[n].get(name) if 0 <= n < len(results) else None
        if not _agrees(figure["value"], computed):
            out.append(f"{where}: step {n} {name} is written {figure['value']!r}, the formulas give {computed!r}")
    return out


# -- the shipped packs --------------------------------------------------------------

def load(root: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Every pack under ``root`` by kernel id, each checked against the
    shipped schema (the one in ``packs_dir()``, whatever ``root`` holds)
    and its file name (``<kernel name>.json``). Raises
    ``KernelPackError`` naming every fault; nothing is dropped quietly."""
    root = root or packs_dir()
    if not os.path.isdir(root):
        raise KernelPackError(f"no kernel packs directory at {root}")
    out: Dict[str, Dict[str, Any]] = {}
    faults: List[str] = []
    for name in sorted(os.listdir(root)):
        if name == SCHEMA_FILE:
            continue
        if not name.endswith(".json"):
            faults.append(f"{name}: not a pack (a pack is <kernel name>.json)")
            continue
        try:
            with open(os.path.join(root, name), encoding="utf-8") as f:
                pack = json.load(f)
        except (OSError, ValueError) as exc:
            faults.append(f"{name}: unreadable: {exc}")
            continue
        found = schema_problems(pack)
        if not found and f"kernel:{name[:-5]}" != pack["kernel"]:
            found = [f"kernel {pack['kernel']!r} does not belong in this file"]
        faults.extend(f"{name}: {p}" for p in found)
        if not found:
            out[pack["kernel"]] = pack
    if faults:
        raise KernelPackError("\n".join(faults))
    return out


def check(root: Optional[str] = None) -> List[str]:
    """Every fault of every shipped pack and every stale line, each as
    ``<kernel>: ...``; empty when all agree with the code. A packs
    directory that cannot be loaded, or holds no pack, is reported as a
    fault (``packs: ...``), never raised and never passed."""
    try:
        packs = load(root)
    except KernelPackError as exc:
        return [f"packs: {p}" for p in str(exc).splitlines()]
    if not packs:
        return [f"packs: no kernel pack in {root or packs_dir()}"]
    entries = _entries()
    out: List[str] = []
    for kernel, pack in packs.items():
        found = problems(pack, entries)
        if not found:
            found = [p for chain in pack["chains"] for p in example_problems(chain, entries)]
            found += [f"{line}: stale (its words or the code facts it names changed since it was stamped; read it again and re-stamp)"
                      for line in stale(pack, entries)]
        out.extend(f"{kernel}: {p}" for p in found)
    return out


def stamp(pack: Dict[str, Any], entries: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """Give every line the fingerprint of its words and named code facts
    now, in place. A line whose fingerprint changes loses its
    ``approved``: what was approved is no longer what the line says or
    what the code says. Returns ``part/id`` of the lines changed. A pack
    with any ``problems`` is not stamped: ``KernelPackError`` names every
    one and the pack is left as it was."""
    entries = entries if entries is not None else _entries()
    found = problems(pack, entries)
    if found:
        raise KernelPackError("\n".join(found))
    changed = []
    for part, line in _lines(pack):
        now = line_fingerprint(entries, part, line)
        if line["fingerprint"] != now:
            line["fingerprint"] = now
            line["approved"] = False
            changed.append(f"{part}/{line['id']}")
    return changed
