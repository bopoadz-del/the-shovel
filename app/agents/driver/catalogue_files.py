"""One JSON file per formula, discipline kernel, knowledge entry and retrieval
tool (``catalogue/<formulas|kernels|knowledge|rag>/<name>.json``).

Each file has two parts and they have two writers:

* ``facts`` is written by code (``write``, run by
  ``scripts/build_catalogue_json.py``) from the same registries the
  catalogue is built from: the name, what the registry's own declaration
  says, every input with its unit and range, the outputs, the tool call that
  runs it, what an answer can cite and where the facts were read. ``drift``
  says where a file and the code disagree; nothing here edits a formula, a
  range or a unit.
* ``hints`` is written by people, as data in the file: what the capability
  is for, the near neighbours it must not be confused with and, for a
  retrieval tool, what it does for each kind of question. A hint carries
  ``approved``; only an approved hint whose file still agrees with the code
  is shown to the model (``enrich``), in the entry's own ``when_to_use`` /
  ``when_not``. Nothing here reads a question and nothing picks a tool.

``load`` validates every file against ``catalogue/schema.json`` and raises
on anything it does not know: an unknown field, a file in the wrong place, a
neighbour that has no file.

A discipline kernel is a hat manifest (``app/agents/manifests``): what
``select_hat`` puts on. A retrieval tool is a registered tool whose result
the registry declares as passages (kind ``search``) or that the platform
offers only inside a project (it reads that project's stored documents).
"""
from __future__ import annotations

import copy
import functools
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_NAME = "catalogue-file/1"
SCHEMA_FILE = "schema.json"

#: A file's kind -> its directory under the files' root, in file order.
DIRS = {"formula": "formulas", "kernel": "kernels", "knowledge": "knowledge", "rag": "rag"}

#: The kinds of question a retrieval tool's hints speak to, each exactly once.
QUESTION_KINDS = ("single_passage", "totals_across_bill_rows",
                  "specification_and_bill", "specification_and_drawing")

_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

_LOG = logging.getLogger(__name__)


class CatalogueFileError(ValueError):
    """A catalogue file, or the set of them, is not what the schema and the
    code say it must be."""


def files_dir() -> str:
    """The directory the shipped files live in."""
    from app.agents.driver import catalogue

    return os.path.join(catalogue._ROOT, "catalogue")


# -- what has a file ----------------------------------------------------------------

def kernel_entries() -> List[Dict[str, Any]]:
    """One entry, in the catalogue's one format, per discipline kernel (hat
    manifest), in discipline order: what ``select_hat`` puts on, and the
    formulas, tools and skills that belong to it, each by its catalogue id."""
    from app.agents import catalog
    from app.agents.core import tool_registry
    from app.agents.driver import catalogue, tools
    from app.lib import formula_registry

    tool_registry.load()
    out = []
    for hat in sorted(catalog.list_hats(), key=lambda h: h.discipline.value):
        discipline = hat.discipline.value
        procedures = (hat.playbook.procedures or []) if hat.playbook else []
        entry = catalogue._entry(
            "kernel", discipline, name=hat.name,
            code={"tool": tools.SELECT_HAT, "hat": discipline},
            source={"registry": "kernel", "file": f"app/agents/manifests/{hat.id}.json",
                    "manifest": hat.id,
                    "formulas": [f"formula:{s.name}" for s in formula_registry.all_specs()
                                 if s.owner == discipline],
                    "tools": [f"tool:{n}" for n in sorted(tool_registry.tools_of(discipline))],
                    "skills": [f"skill:{discipline}.{p.id}" for p in procedures]})
        entry["for"] = catalogue._one_line(hat.description)
        entry["gaps"] = catalogue._gaps(entry)
        out.append(entry)
    return out


def _offered(project_id: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """The platform's own tool schemas for a turn with or without a project
    (an agent with no blocks and no delegation), by tool name."""
    import types

    from app.agents.core import tools as core_tools

    agent = types.SimpleNamespace(allowed_blocks=[], can_delegate=False)
    return {t["function"]["name"]: t["function"]
            for t in core_tools.tool_definitions(agent, project_id=project_id)}


def rag_tool_names() -> List[str]:
    """The registered tools that retrieve from stored documents, sorted: a
    tool whose result the registry declares as passages (kind ``search``),
    or one the platform offers only to a turn inside a project."""
    from app.agents.core import tool_registry

    tool_registry.load()
    inside, outside = _offered("a project"), _offered(None)
    return sorted(name for name, spec in tool_registry._TOOLS.items()
                  if spec.kind == "search" or (name in inside and name not in outside))


def capabilities() -> List[Tuple[str, Dict[str, Any]]]:
    """``(file kind, catalogue entry)`` for everything that must have exactly
    one file: every registered formula, every discipline kernel, every
    knowledge entry and every retrieval tool, by kind then id."""
    from app.agents.driver import catalogue

    held = catalogue.entries()
    rag = {f"tool:{name}" for name in rag_tool_names()}
    out = [(e["kind"], e) for e in held if e["kind"] in ("formula", "knowledge")]
    out += [("rag", e) for e in held if e["id"] in rag]
    out += [("kernel", e) for e in kernel_entries()]
    return sorted(out, key=lambda pair: (list(DIRS).index(pair[0]), pair[1]["id"]))


# -- the facts of one file ------------------------------------------------------------

def _call(code: Dict[str, Any], inputs_as: str = "") -> Dict[str, Any]:
    call: Dict[str, Any] = {"tool": code["tool"],
                            "arguments": {k: v for k, v in code.items() if k != "tool"}}
    if inputs_as:
        call["inputs_as"] = inputs_as
    return call


def _formula_facts(entry: Dict[str, Any]) -> Dict[str, Any]:
    from app.lib import formula_registry

    spec = formula_registry.get(entry["code"]["calculation"])
    return {"described_as": spec.description,
            "required_groups": [list(group) for group in spec.required_groups],
            # The calculator takes a formula's inputs in ``params``.
            "call": _call(entry["code"], "params"),
            "cites": {"standard": entry["standard"], "credit": spec.display_name},
            "feeds": entry["feeds"], "fed_by": entry["fed_by"]}


def _kernel_facts(entry: Dict[str, Any]) -> Dict[str, Any]:
    from app.agents import catalog

    return {"described_as": catalog.get_hat_by_id(entry["source"]["manifest"]).description}


def _knowledge_facts(entry: Dict[str, Any]) -> Dict[str, Any]:
    source = entry["source"]
    cites = {"document": source.get("file", ""), "title": entry["name"]}
    cites.update({k: source[k] for k in ("entry", "provenance", "credibility_tier") if k in source})
    # The knowledge base tool takes an evaluated entry's inputs in ``values``.
    return {"call": _call(entry["code"], "values" if entry["inputs"] else ""), "cites": cites}


def _rag_facts(entry: Dict[str, Any]) -> Dict[str, Any]:
    from app.agents.core import tool_registry
    from app.agents.driver import sources

    name = entry["code"]["tool"]
    spec = tool_registry.get(name)
    # The schema the model is offered: the tool's own, or the platform's.
    offered = ((spec.schema or {}).get("function") or {}) or _offered("a project").get(name) or {}
    properties = (offered.get("parameters") or {}).get("properties") or {}
    inputs = []
    for item in entry["inputs"]:
        declared = properties.get(item["name"]) or {}
        item = dict(item)
        if declared.get("enum"):
            item["choices"] = list(declared["enum"])
        if declared.get("description"):
            item["described_as"] = str(declared["description"])
        inputs.append(item)
    return {"described_as": " ".join(str(offered.get("description") or "").split()),
            "inputs": inputs,
            "cites": {"result": spec.kind, "passages_are_sources": name in sources.RETRIEVAL_TOOLS,
                      "reason_codes": list(spec.reason_codes)}}


_FACTS = {"formula": _formula_facts, "kernel": _kernel_facts,
          "knowledge": _knowledge_facts, "rag": _rag_facts}


def facts(kind: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """The ``facts`` of the file of kind ``kind`` for a catalogue entry, read
    from the entry and the registry behind it, as plain JSON values. A figure
    JSON cannot hold (an infinite range bound) raises ``ValueError``."""
    if kind not in _FACTS:
        raise CatalogueFileError(f"unknown file kind {kind!r}; one of {', '.join(DIRS)}")
    out = {"name": entry["name"], "described_as": entry["for"], "inputs": entry["inputs"],
           "outputs": entry["outputs"], "call": _call(entry["code"]), "cites": {},
           "source": entry["source"]}
    out.update(_FACTS[kind](entry))
    return json.loads(json.dumps(out, allow_nan=False))


def path_of(kind: str, entry_id: str) -> str:
    """The file of an entry, relative to the files' root, with ``/``."""
    key = str(entry_id).partition(":")[2]
    if kind not in DIRS or not _KEY.fullmatch(key):
        raise CatalogueFileError(f"{entry_id!r} of kind {kind!r} has no file name")
    return f"{DIRS[kind]}/{key}.json"


# -- reading the files ------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _validator() -> Any:
    import jsonschema

    with open(os.path.join(files_dir(), SCHEMA_FILE), encoding="utf-8") as f:
        return jsonschema.Draft7Validator(json.load(f))


def problems(doc: Any) -> List[str]:
    """Why ``doc`` is not a valid catalogue file, each as ``where: what``;
    empty when it is valid. The schema first (an unknown or missing field, a
    wrong type); then what the schema cannot say: a retrieval tool's question
    kinds are each spoken to exactly once, and a file's neighbours name
    neither the file itself nor one id twice."""
    found = [f"{'/'.join(str(p) for p in error.absolute_path) or 'file'}: {error.message[:200]}"
             for error in sorted(_validator().iter_errors(doc),
                                 key=lambda e: [str(p) for p in e.absolute_path])]
    if found:
        return found
    hints = doc["hints"]
    kinds = [q["kind"] for q in hints.get("question_kinds") or []]
    if kinds and sorted(kinds) != sorted(QUESTION_KINDS):
        found.append("hints/question_kinds: each question kind is spoken to exactly once ("
                     + ", ".join(QUESTION_KINDS) + ")")
    neighbours = [n["id"] for n in hints["not_to_be_confused_with"]]
    if doc["id"] in neighbours:
        found.append("hints/not_to_be_confused_with: names the file itself")
    if len(set(neighbours)) != len(neighbours):
        found.append("hints/not_to_be_confused_with: names an id twice")
    return found


def _listing(root: str) -> List[str]:
    """Every file under ``root``, relative with ``/``, sorted."""
    out = []
    for folder, _dirs, names in os.walk(root):
        rel = os.path.relpath(folder, root).replace(os.sep, "/")
        out.extend(name if rel == "." else f"{rel}/{name}" for name in names)
    return sorted(out)


def identity(root: Optional[str] = None) -> Tuple[Tuple[str, int, int], ...]:
    """What a ``load`` would read now: each file's name, mtime and size. It
    changes when a file is added, removed or rewritten."""
    root = root or files_dir()
    state = []
    for rel in _listing(root):
        st = os.stat(os.path.join(root, rel))
        state.append((rel, st.st_mtime_ns, st.st_size))
    return tuple(state)


def load(root: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Every catalogue file under ``root`` (the shipped files by default), by
    id, each validated. Raises ``CatalogueFileError`` naming every fault: a
    missing directory, a file that is not ``<kind directory>/<name>.json``,
    unreadable JSON, a schema fault (``problems``), an id that does not
    belong in its file, or a neighbour that has no file. Nothing is dropped
    quietly."""
    root = root or files_dir()
    if not os.path.isdir(root):
        raise CatalogueFileError(f"no catalogue files directory at {root}")
    kind_of = {folder: kind for kind, folder in DIRS.items()}
    out: Dict[str, Dict[str, Any]] = {}
    faults: List[str] = []
    for rel in _listing(root):
        if rel == SCHEMA_FILE:
            continue
        folder, _, filename = rel.rpartition("/")
        if folder not in kind_of or not filename.endswith(".json"):
            faults.append(f"{rel}: not a catalogue file (a file is "
                          f"<{'|'.join(DIRS.values())}>/<name>.json)")
            continue
        try:
            with open(os.path.join(root, rel), encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as exc:
            faults.append(f"{rel}: unreadable: {exc}")
            continue
        found = problems(doc)
        if not found and (doc["kind"] != kind_of[folder] or path_of(doc["kind"], doc["id"]) != rel):
            found = [f"id {doc['id']!r} of kind {doc['kind']!r} does not belong in this file"]
        faults.extend(f"{rel}: {p}" for p in found)
        if not found:
            out[doc["id"]] = doc
    for doc in out.values():
        for neighbour in doc["hints"]["not_to_be_confused_with"]:
            if neighbour["id"] not in out:
                faults.append(f"{path_of(doc['kind'], doc['id'])}: not_to_be_confused_with names "
                              f"{neighbour['id']!r}, which has no file")
    if faults:
        raise CatalogueFileError("\n".join(faults))
    return out


# -- the files against the code ---------------------------------------------------------

def drift(root: Optional[str] = None) -> List[str]:
    """Where the files and the code disagree; empty when they agree. A
    registered capability without a file, a file the code registers nothing
    for, and a file whose ``facts`` are not what the code says now (the
    differing parts are named). Raises what ``load`` raises."""
    files = load(root)
    out: List[str] = []
    registered = set()
    for kind, entry in capabilities():
        registered.add(entry["id"])
        doc = files.get(entry["id"])
        if doc is None:
            out.append(f"missing file: {path_of(kind, entry['id'])} for {entry['id']}")
            continue
        now = facts(kind, entry)
        differing = sorted(k for k in set(now) | set(doc["facts"]) if now.get(k) != doc["facts"].get(k))
        if differing:
            out.append(f"{entry['id']}: facts differ from the code in: {', '.join(differing)}")
    out.extend(f"extra file: {path_of(doc['kind'], entry_id)} names {entry_id}, which the code does "
               "not register" for entry_id, doc in sorted(files.items()) if entry_id not in registered)
    return out


def hint_gaps(root: Optional[str] = None) -> List[str]:
    """The ids of the files whose hint is not written yet: no purpose, or a
    retrieval tool without its question kinds. Sorted."""
    return sorted(entry_id for entry_id, doc in load(root).items()
                  if not doc["hints"]["purpose"].strip()
                  or (doc["kind"] == "rag" and not doc["hints"]["question_kinds"]))


def _kept_hints(path: str, kind: str) -> Dict[str, Any]:
    """The hints already written in the file at ``path``, untouched; for a
    file that does not exist yet, the empty hint of its kind. A file that is
    there and cannot be read is never overwritten."""
    blank: Dict[str, Any] = {"approved": False, "purpose": "", "not_to_be_confused_with": []}
    if kind == "rag":
        blank["question_kinds"] = []
    if not os.path.exists(path):
        return blank
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    except (OSError, ValueError) as exc:
        raise CatalogueFileError(f"{path}: unreadable, so not rewritten (fix or delete it): {exc}") from exc
    hints = existing.get("hints") if isinstance(existing, dict) else None
    return hints if isinstance(hints, dict) else blank


def write(root: Optional[str] = None, prune: bool = False) -> Dict[str, Any]:
    """Write the ``facts`` of every capability's file from the code, keeping
    each file's hints as they are; a capability without a file gets one with
    an empty hint. A file the code registers nothing for is reported under
    ``extra`` (and removed only with ``prune``). Returns what was done:
    ``written``, ``unchanged``, ``extra``, ``removed``."""
    root = root or files_dir()
    report: Dict[str, Any] = {"written": [], "unchanged": 0, "extra": [], "removed": []}
    wanted = set()
    for kind, entry in capabilities():
        rel = path_of(kind, entry["id"])
        wanted.add(rel)
        path = os.path.join(root, rel)
        doc = {"schema": SCHEMA_NAME, "id": entry["id"], "kind": kind,
               "facts": facts(kind, entry), "hints": _kept_hints(path, kind)}
        text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as f:
                if f.read().replace("\r\n", "\n") == text:
                    report["unchanged"] += 1
                    continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        report["written"].append(rel)
    for rel in _listing(root):
        if rel == SCHEMA_FILE or rel in wanted:
            continue
        if prune:
            os.remove(os.path.join(root, rel))
        report["removed" if prune else "extra"].append(rel)
    return report


# -- using a file -----------------------------------------------------------------------

def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def call_for(doc: Dict[str, Any], values: Any) -> Dict[str, Any]:
    """The tool call that runs a file's capability with ``values`` (input
    name -> value): ``{"tool": ..., "arguments": ...}``. Only names and
    presence are checked here -- a name the file does not list, a required
    input that is absent or blank, an alternative group with none of its
    inputs -- each refused by name. Units and ranges are the tool's to check
    when it runs."""
    spec = doc["facts"]
    if not isinstance(values, dict):
        raise CatalogueFileError(f"{doc['id']}: the values are an object of input name -> value")
    inputs = {item["name"]: item for item in spec["inputs"]}
    unknown = sorted(str(name) for name in values if name not in inputs)
    if unknown:
        raise CatalogueFileError(f"{doc['id']} has no input named {', '.join(unknown)}")
    absent = [name for name, item in inputs.items() if item["required"] and _blank(values.get(name))]
    if absent:
        raise CatalogueFileError(f"{doc['id']} needs {', '.join(absent)}")
    for group in spec.get("required_groups") or []:
        if all(_blank(values.get(name)) for name in group):
            raise CatalogueFileError(f"{doc['id']} needs one of {', '.join(group)}")
    call = spec["call"]
    arguments = copy.deepcopy(call["arguments"])
    if call.get("inputs_as"):
        arguments[call["inputs_as"]] = dict(values)
    else:
        arguments.update(values)
    return {"tool": call["tool"], "arguments": arguments}


def hint_text(doc: Dict[str, Any]) -> Tuple[str, str]:
    """A file's hints as the two texts of a catalogue entry: when to use it
    (the purpose, then what a retrieval tool does for the question kinds it
    serves) and when not (its neighbours, each with the difference, then the
    question kinds it is not for)."""
    hints = doc["hints"]
    kinds = hints.get("question_kinds") or []
    use = " ".join([hints["purpose"].strip()] + [q["how"] for q in kinds if q["role"] != "not_for"])
    apart = "; ".join(f"{n['id']} ({n['difference']})" for n in hints["not_to_be_confused_with"])
    avoid = " ".join(([f"Not to be confused with {apart}."] if apart else [])
                     + [q["how"] for q in kinds if q["role"] == "not_for"])
    return use.strip(), avoid


def enrich(entries: List[Dict[str, Any]], files: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The entries with their files' hints, as copies, in the same order.

    A hint reaches an entry only when its file says ``approved`` and the
    file's facts are what the code says of that entry now; it fills the
    entry's ``when_to_use`` / ``when_not`` where the registry declares none
    (a registry's own declaration is never replaced) and takes them off the
    entry's ``gaps``. Every other field, and every entry without such a
    file, is returned as it came: which tool runs is not touched."""
    out = []
    for entry in entries:
        entry = copy.deepcopy(entry)
        out.append(entry)
        doc = files.get(entry["id"])
        if doc is None or not doc["hints"]["approved"]:
            continue
        if doc["facts"] != facts(doc["kind"], entry):
            _LOG.warning("catalogue file of %s is not applied: its facts are not what the code says now",
                         entry["id"])
            continue
        use, avoid = hint_text(doc)
        entry["when_to_use"] = entry["when_to_use"] or use
        entry["when_not"] = entry["when_not"] or avoid
        entry["gaps"] = [g for g in entry["gaps"] if g not in ("when_to_use", "when_not") or not entry[g]]
    return out


def enriched(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``enrich`` with the shipped files, for the catalogue's own build. When
    the files cannot be loaded the entries are returned as the registries
    gave them and the fault is logged as an error: the catalogue is the
    registries' and stays whole; only the hints are missing."""
    try:
        files = load()
    except CatalogueFileError as exc:
        _LOG.error("catalogue files not used (the entries keep the registries' own text): %s", exc)
        return entries
    return enrich(entries, files)
