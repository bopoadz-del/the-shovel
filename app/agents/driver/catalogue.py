"""The catalogue: one JSON entry, in one format, for every formula, tool,
workflow, skill and knowledge file the platform registers -- the whole
registry, not a chosen few -- built by code from the registries themselves.

The model is given the short index (``index_lines``: one line per entry,
``id | kind | name | for``, in a fixed order) and the ``catalogue_lookup``
tool, which returns the full entries for the ids it names. Nothing here
reads a question; nothing here is typed per entry. An optional declaration
an entry's registry does not carry yet is listed in that entry's ``gaps``.

Entry fields: ``id`` (``<kind>:<registry name>``), ``kind``, ``name``,
``for`` (one line; also the index line), ``inputs`` (name, label, type,
unit, range ``[lo, hi]`` with either bound null when not declared, required,
basis; ``lower_bound_excluded: true`` only on an input whose range begins
above its lower bound), ``outputs`` (name, unit, main),
``when_to_use``, ``when_not``, ``standard``, ``examples`` (invented figures
only), ``feeds`` / ``fed_by`` (formulas: an output that is another formula's
input of the same name and unit), ``code`` (how the entry is called),
``source`` (the registry and the file the entry was generated from),
``stamps`` (``verified`` / ``tested`` / ``approved``; ``tested`` is the
registry's own certification, the other two are never set here) and
``gaps``.

``document_type`` and ``kernel`` are kinds of the format with no registry on
this build: they hold no entries until one exists.
"""
from __future__ import annotations

import copy
import inspect
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agents.driver.catalogue_knowledge import kb_entries, markdown_sections

#: Every kind an entry may have, in index order.
KINDS = ("formula", "tool", "workflow", "skill", "kernel", "knowledge", "document_type")

#: The optional declarations an entry is checked for; a missing one is a gap.
DECLARED = ("when_to_use", "when_not", "examples")

LOOKUP = "catalogue_lookup"

#: At most this many entries per lookup: the tool is never the whole
#: catalogue.
MAX_LOOKUP = 12

#: ... and at most this many characters of entries. The whole result (the
#: entries and the ``not_returned`` / ``unknown`` names on top of them) is
#: then held to the loop's tool-result limit by ``lookup`` itself, so the
#: loop never cuts it (app.agents.driver.loop._TOOL_RESULT_CHARS).
LOOKUP_CHARS = 16000

#: An index line's ``for`` text is cut to this many characters.
FOR_CHARS = 160

_LOG = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _one_line(text: Any, limit: int = FOR_CHARS) -> str:
    line = re.sub(r"\s+", " ", str(text or "")).strip()
    return line if len(line) <= limit else line[:limit - 1].rstrip() + "…"


def _source_file(obj: Any) -> str:
    try:
        path = inspect.getsourcefile(obj) or ""
    except TypeError:
        # A built-in or a C callable has no source file: the entry says so.
        _LOG.debug("catalogue: no source file for %r", obj, exc_info=True)
        return ""
    return os.path.relpath(path, _ROOT).replace(os.sep, "/") if path else ""


#: On an input whose declared range does not include its lower bound
#: ("more than 0"): present and true only then.
LOWER_BOUND_EXCLUDED = "lower_bound_excluded"


def _input(name: str, p: Any) -> Dict[str, Any]:
    unit = getattr(p, "unit", "") or ""
    lo, hi = getattr(p, "lo", None), getattr(p, "hi", None)
    entry = {"name": name, "label": getattr(p, "label", "") or name.replace("_", " "),
             "type": getattr(p, "kind", "") or "",
             "unit": "" if unit == "-" else unit,
             "range": [lo, hi] if lo is not None or hi is not None else None,
             "required": bool(getattr(p, "required", False)),
             "basis": getattr(p, "basis", "") or ""}
    if lo is not None and getattr(p, "above_lo", False):
        # The range begins ABOVE its lower bound: said in the entry, so the
        # reader of "range": [0, 5] is not told that 0 is accepted.
        entry[LOWER_BOUND_EXCLUDED] = True
    return entry


def _gaps(entry: Dict[str, Any], extra: Tuple[str, ...] = ()) -> List[str]:
    gaps = [k for k in DECLARED if not entry.get(k)]
    if not entry.get("for"):
        gaps.insert(0, "for")
    return gaps + list(extra)


def _entry(kind: str, key: str, **fields: Any) -> Dict[str, Any]:
    entry = {"id": f"{kind}:{key}", "kind": kind, "name": "", "for": "", "inputs": [],
             "outputs": [], "when_to_use": "", "when_not": "", "standard": None,
             "examples": [], "feeds": [], "fed_by": [], "code": {}, "source": {},
             "stamps": {"verified": None, "tested": None, "approved": None}, "gaps": []}
    entry.update(fields)
    return entry


def _formula_entries() -> List[Dict[str, Any]]:
    from app.lib import formula_registry as fr

    out = []
    for spec in fr.all_specs():
        params = fr.parameters(spec)
        entry = _entry(
            "formula", spec.name, name=spec.display_name,
            inputs=[_input(k, p) for k, p in params.items()],
            outputs=[{"name": k, "unit": "" if u == "-" else u, "main": k == spec.main_output}
                     for k, u in spec.outputs.items()],
            when_to_use=spec.when_to_use, when_not=spec.when_not,
            standard=spec.standard or None,
            examples=[dict(e) for e in spec.examples],
            code={"tool": "construction_calc", "calculation": spec.name},
            source={"registry": "formula", "owner": spec.owner, "file": _source_file(spec.fn)},
            stamps={"verified": None,
                    "tested": {"by": "formula_registry.certified"} if fr.certified(spec.name) else None,
                    "approved": None})
        entry["for"] = _one_line(spec.description)
        gaps = ("standard",) if not spec.standard else ()
        entry["gaps"] = _gaps(entry, gaps)
        out.append(entry)
    _link_formulas(out)
    return out


def _link_formulas(entries: List[Dict[str, Any]]) -> None:
    """``feeds`` / ``fed_by``: formula A feeds formula B when one of A's
    outputs is one of B's inputs by the same name and the same unit."""
    takes: Dict[Tuple[str, str], List[str]] = {}
    for e in entries:
        for i in e["inputs"]:
            if i["unit"]:
                takes.setdefault((i["name"], i["unit"]), []).append(e["id"])
    by_id = {e["id"]: e for e in entries}
    for e in entries:
        for o in e["outputs"]:
            for target in takes.get((o["name"], o["unit"]), []):
                if target != e["id"] and target not in e["feeds"]:
                    e["feeds"].append(target)
                    by_id[target]["fed_by"].append(e["id"])
    for e in entries:
        e["feeds"].sort()
        e["fed_by"].sort()


def _tool_entries() -> List[Dict[str, Any]]:
    from app.agents.core import tool_registry

    tool_registry.load()
    out = []
    for name, spec in sorted(tool_registry._TOOLS.items()):
        described = ((spec.schema or {}).get("function") or {}).get("description") or ""
        entry = _entry(
            "tool", name, name=spec.display_name,
            inputs=[_input(k, p) for k, p in (spec.params or {}).items()],
            outputs=[{"name": "result", "unit": "", "kind": spec.kind}] if spec.kind else [],
            when_to_use=spec.when_to_use, when_not=spec.when_not,
            examples=[dict(e) for e in spec.examples],
            code={"tool": name},
            source={"registry": "tool", "owner": spec.owner, "file": _source_file(spec.handler)})
        entry["for"] = _one_line(described)
        entry["gaps"] = _gaps(entry, ("params",) if spec.params is None else ())
        out.append(entry)
    return out


def _workflow_entries() -> List[Dict[str, Any]]:
    from app.agents.driver import routes

    declared = routes.cards()
    out = []
    for name in routes.catalogue():
        card = declared.get(name) or {}
        entry = _entry(
            "workflow", name, name=name.replace("_", " ").capitalize(),
            inputs=[{"name": k, "label": str(v), "type": "", "unit": "", "range": None,
                     "required": False, "basis": ""} for k, v in (card.get("inputs") or {}).items()],
            outputs=[{"name": "deliverable", "unit": "", "main": True}] if card.get("produces") else [],
            when_to_use=str(card.get("use_when") or "").strip(),
            code={"tool": routes.RUN_WORKFLOW, "workflow": name},
            source={"registry": "workflow", "file": "config/workflow_cards.yaml" if card else ""})
        entry["for"] = _one_line(card.get("produces"))
        entry["gaps"] = _gaps(entry, () if card else ("card",))
        if card.get("needs"):
            entry["needs"] = str(card["needs"]).strip()
        out.append(entry)
    return out


def _skill_entries() -> List[Dict[str, Any]]:
    from app.agents import catalog

    out = []
    for hat in sorted(catalog.list_hats(), key=lambda h: h.discipline.value):
        for proc in (hat.playbook.procedures or []) if hat.playbook else []:
            name = f"{hat.discipline.value}.{proc.id}"
            entry = _entry(
                "skill", name, name=proc.title,
                inputs=[{"name": "inputs", "label": proc.inputs.strip(), "type": "text", "unit": "",
                         "range": None, "required": True, "basis": ""}] if proc.inputs.strip() else [],
                outputs=[{"name": "output", "unit": "", "main": True, "label": proc.output.strip()}],
                when_to_use=proc.purpose.strip(),
                # A procedure's refusals are what it must not do: its when-not.
                when_not=proc.refusals.strip(),
                code={"hat": hat.discipline.value, "procedure": proc.id},
                source={"registry": "skill", "file": f"app/agents/manifests/{hat.id}.json",
                        "lineage": proc.lineage})
            entry["for"] = _one_line(f"{proc.title}: {proc.purpose}")
            entry["gaps"] = _gaps(entry, ("inputs:structured",))
            out.append(entry)
    return out


def knowledge_dir() -> str:
    from app.core.knowledge_seed import KNOWLEDGE_DIR

    return KNOWLEDGE_DIR


def _knowledge_summary(text: str) -> Tuple[str, str]:
    """The file's first heading, and its first line of prose."""
    title, prose = "", ""
    for stripped in (line.strip() for line in text.splitlines() if line.strip()):
        if stripped.startswith("#"):
            if not title:
                title = stripped.lstrip("#").strip()
            continue
        if stripped.startswith(("|", ">", "-", "*", "```")) and not prose:
            continue
        if not prose:
            prose = stripped
        if title and prose:
            break
    return title, prose


def _knowledge_entries() -> List[Dict[str, Any]]:
    root = knowledge_dir()
    out = []
    names = sorted(f for f in os.listdir(root) if f.endswith(".md")) if os.path.isdir(root) else []
    for filename in names:
        with open(os.path.join(root, filename), encoding="utf-8", errors="replace") as f:
            text = f.read()
        title, prose = _knowledge_summary(text)
        stem = filename[:-3]
        entry = _entry(
            "knowledge", stem, name=title or stem.replace("_", " "),
            code={"tool": "search_general_knowledge"},
            source={"registry": "knowledge", "file": f"docs/knowledge/{filename}",
                    "sections": markdown_sections(text)})
        entry["for"] = _one_line(prose)
        entry["gaps"] = _gaps(entry)
        out.append(entry)
    return out


_BUILDERS = (_formula_entries, _tool_entries, _workflow_entries, _skill_entries, _knowledge_entries,
             kb_entries)

_CACHE: Dict[Any, List[Dict[str, Any]]] = {}


def _knowledge_state() -> Tuple[Tuple[str, int, int], ...]:
    root = knowledge_dir()
    if not os.path.isdir(root):
        return ()
    state = []
    for name in sorted(f for f in os.listdir(root) if f.endswith(".md")):
        st = os.stat(os.path.join(root, name))
        state.append((name, st.st_mtime_ns, st.st_size))
    return tuple(state)


def _state_key() -> Any:
    """What every builder reads: the formula and tool registries, the
    workflows and their cards, the hats' procedures and the knowledge
    files. A change in any of them is a new key, so no kind is stale."""
    from app.agents import catalog
    from app.agents.core import tool_registry
    from app.agents.driver import routes
    from app.blocks import _knowledge
    from app.lib import formula_registry

    tool_registry.load()
    return (tuple((s.name, id(s)) for s in formula_registry.all_specs()),
            tuple((n, id(s)) for n, s in sorted(tool_registry._TOOLS.items())),
            tuple(routes.catalogue()), id(routes.cards()),
            tuple(id(h) for h in catalog.list_hats()),
            _knowledge_state(),
            # The knowledge base reloads when its file changes; the file's
            # identity (path, mtime, size, inode) is part of the state.
            _knowledge.kb_identity())


def order(entry: Dict[str, Any]) -> Tuple[int, str]:
    """The fixed order: by kind (``KINDS``), then by id."""
    return KINDS.index(entry["kind"]), entry["id"]


def entries() -> List[Dict[str, Any]]:
    """Every entry, in the fixed order. Built once per state of what the
    builders read (``_state_key``: a registered, replaced or removed
    formula, tool, workflow, card, skill or knowledge file is never
    stale); each caller gets its own copy."""
    key = _state_key()
    if key not in _CACHE:
        _CACHE.clear()
        built = [e for build in _BUILDERS for e in build()]
        _CACHE[key] = sorted(built, key=order)
    return copy.deepcopy(_CACHE[key])


def index_line(entry: Dict[str, Any]) -> str:
    return " | ".join((entry["id"], entry["kind"], _one_line(entry["name"], 80), entry["for"] or "-"))


def index_lines(items: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """One line per entry, ``id | kind | name | for``, in the fixed order."""
    return [index_line(e) for e in sorted(items if items is not None else entries(), key=order)]


def asked_ids(ids: Any) -> List[str]:
    """The ids a lookup call names, once each, in the order named: a list
    of strings, or one string. Anything else names none."""
    if isinstance(ids, str):
        ids = [ids]
    if not isinstance(ids, (list, tuple)):
        return []
    return list(dict.fromkeys(i.strip() for i in ids if isinstance(i, str) and i.strip()))


def _result_limit() -> int:
    from app.agents.driver.loop import _TOOL_RESULT_CHARS

    return _TOOL_RESULT_CHARS


def _size(result: Dict[str, Any]) -> int:
    import json

    return len(json.dumps(result, default=str))


def _cap_names(result: Dict[str, Any], keys: Any, limit: int) -> None:
    """Keep as many names of each list in ``keys`` (one key or several, in
    the order given), in order, as fit the result in ``limit`` characters;
    the rest of a list are counted under ``<key>_more``.

    Every list's count is held at its largest before any name is kept, so a
    later list's count always has room: keeping names only ever shrinks a
    count, and dropping a count that is not needed only frees room."""
    import json

    keys = (keys,) if isinstance(keys, str) else tuple(keys)
    lists = {key: list(result.get(key) or []) for key in keys}
    for key, names in lists.items():
        if names:
            result[key] = []
            result[f"{key}_more"] = len(names)  # sized at its largest; shrinks below
    for key, names in lists.items():
        if not names:
            continue
        more_key = f"{key}_more"
        size = _size(result)
        kept: List[str] = []
        for name in names:
            add = len(json.dumps(name)) + (2 if kept else 0)
            if size + add > limit:
                break
            kept.append(name)
            size += add
        result[key] = kept
        if len(kept) < len(names):
            result[more_key] = len(names) - len(kept)
        else:
            del result[more_key]


def read_ids(result: Any) -> Dict[str, Any]:
    """What a lookup call gave the model, for the turn's trace, read from the
    lookup's own result: the ids of the entries it returned (what the model
    read), how many held ids it named or counted without returning, and how
    many asked ids the catalogue does not hold. An id the catalogue does not
    hold is the model's own text, so it is only ever counted. Nothing is
    read from the catalogue again: this is the result, re-told."""
    result = result if isinstance(result, dict) else {}
    returned = [e for e in result.get("entries") or [] if isinstance(e, dict)]

    def count(key: str) -> int:
        return len(result.get(key) or []) + int(result.get(f"{key}_more") or 0)

    return {"ids": [str(e.get("id") or "") for e in returned],
            "not_returned": count("not_returned"), "unknown": count("unknown")}


def lookup(ids: Any, limit: Optional[int] = None) -> Dict[str, Any]:
    """The full entries for ``ids`` (entry ids from the index), in the order
    asked: at most ``MAX_LOOKUP`` entries and ``LOOKUP_CHARS`` characters of
    them; the rest are named under ``not_returned`` and ids the catalogue
    does not hold under ``unknown``. Never the whole catalogue.

    The whole result, as JSON, is at most ``limit`` characters (the loop's
    tool-result limit by default), so the loop never cuts it: names under
    ``not_returned`` and then ``unknown`` that do not fit are counted under
    ``not_returned_more`` / ``unknown_more`` instead of named."""
    import json

    limit = _result_limit() if limit is None else limit
    asked = asked_ids(ids)
    if not asked:
        return {"ok": False, "error": "Name the entry ids to look up, as a list (ids from the catalogue index)."}
    held = {e["id"]: e for e in entries()}
    found = [held[i] for i in asked if i in held]
    unknown = [i for i in asked if i not in held]
    returned: List[Dict[str, Any]] = []
    size = 0
    for entry in found[:MAX_LOOKUP]:
        size += len(json.dumps(entry, default=str))
        if returned and size > LOOKUP_CHARS:
            break
        returned.append(entry)
    result: Dict[str, Any] = {"ok": bool(found), "entries": returned, "unknown": []}
    rest = [e["id"] for e in found[len(returned):]]
    if rest:
        result["not_returned"] = rest
    if not found:
        result["error"] = "None of these ids is in the catalogue index."
    # The found ids still to ask for come first; then the unknown ones.
    result["unknown"] = unknown
    _cap_names(result, ("not_returned", "unknown"), limit)
    return result


LOOKUP_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": LOOKUP,
        "description": "Read the full catalogue entries for ids from the catalogue index: inputs with "
                       "units and ranges, outputs, when to use and when not, examples, what each feeds "
                       f"and how to call it. At most {MAX_LOOKUP} ids per call; found ids past that come "
                       "back under not_returned (counted under not_returned_more when too many to "
                       "list): ask for them in another call.",
        "parameters": {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "string"},
                                   "description": "Entry ids exactly as the index writes them."}},
            "required": ["ids"],
        },
    },
}


def index_block() -> str:
    """The index as the model reads it in the system message."""
    return ("Catalogue index (id | kind | name | what it is for). Read an entry in full with "
            f"{LOOKUP} before you rely on its inputs.\n" + "\n".join(index_lines()))
