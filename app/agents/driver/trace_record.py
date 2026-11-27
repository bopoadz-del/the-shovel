"""The trace record: one per answer, written by code (schema ``trace/1``).

It serves three uses with one shape: the tester's scoring of the six
behaviours, the saved example once a path is verified, and a
provider-neutral training row. Plain JSON; a tool call is stored as
``{"tool", "args", "result", "ms"}`` -- nothing in it is specific to one
model provider, and the provider and model are recorded as settings.

Four fields decide whether a record can ever become a training row, and
none of them is set by the model that produced the answer:

* ``verified`` -- ``{"by": "tester:<run>" | "user:<id>", "at": ...}``,
  stamped only through ``stamp_verified``;
* ``path_right`` -- set with the verification; a right figure reached by a
  wrong path is not a training row;
* ``figures_from_tools`` -- computed here from the answer's figure
  provenance: every figure is credited to a tool result, a read row or
  passage, or the user's own words;
* ``question_set`` -- ``teaching`` / ``frozen`` / ``held_back``, from the
  project's profile, or ``unset`` when it names none; a held-back or unset
  question is never exportable.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = "trace/1"
QUESTION_SETS = ("teaching", "frozen", "held_back")
#: The set of a turn whose project names none: recorded, never exportable.
UNSET = "unset"
ENDINGS = ("answer", "asked_input", "ceiling")
#: Who may stamp a path verified: a tester run or a user. Never the model.
VERIFIERS = ("tester", "user")

#: A provenance source that is not a source.
_UNSOURCED = (None, "", "unresolved")


def figures_from_tools(provenance: Iterable[Dict[str, Any]]) -> bool:
    """True when every figure of the answer has a source (a calculator
    result, a read row or passage, or the user's own words). An answer with
    no figure is true: it carries no figure that could have been invented."""
    return all(isinstance(entry, dict) and entry.get("source") not in _UNSOURCED
               for entry in provenance)


def build(*, turn: str, question: str, rounds: List[Dict[str, Any]], final_answer: str,
          figures: List[Dict[str, Any]], ended: str, question_set: str,
          seconds: float, provider: str, model: str, temperature: Optional[float] = None,
          seed: Optional[int] = None, build_sha: str = "", project_setting: str = "",
          index_offered: Optional[List[str]] = None,
          entries_looked_up: Optional[List[str]] = None,
          examples_offered: Optional[List[Dict[str, Any]]] = None,
          sheet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The record of one answered turn. Raises ``ValueError`` on an unknown
    ending or question set -- a record whose set is unknown could leak a
    held-back question."""
    if ended not in ENDINGS:
        raise ValueError(f"ended must be one of {ENDINGS}, not {ended!r}")
    if question_set not in QUESTION_SETS + (UNSET,):
        raise ValueError(f"question_set must be one of {QUESTION_SETS + (UNSET,)}, not {question_set!r}")
    rounds = copy.deepcopy(list(rounds or []))
    return {
        "schema": SCHEMA, "turn": str(turn), "project_setting": str(project_setting),
        "build_sha": str(build_sha), "provider": str(provider), "model": str(model),
        "temperature": temperature, "seed": seed, "question": str(question),
        "question_set": question_set,
        "index_offered": list(index_offered or []),
        "entries_looked_up": list(entries_looked_up or []),
        "examples_offered": copy.deepcopy(list(examples_offered or [])),
        "rounds": rounds, "sheet": copy.deepcopy(sheet) if sheet else None,
        "final_answer": str(final_answer), "figures": copy.deepcopy(list(figures or [])),
        "figures_from_tools": figures_from_tools(figures or []),
        "rounds_used": len(rounds), "seconds": round(float(seconds), 3), "ended": ended,
        "verified": None, "path_right": None,
    }


def stamp_verified(record: Dict[str, Any], by: str, at: str, path_right: bool) -> Dict[str, Any]:
    """A copy of ``record`` stamped verified by ``by`` (``tester:<run>`` or
    ``user:<id>``). Raises ``ValueError`` for any other verifier -- the model
    that wrote the answer cannot verify it."""
    kind, _, ident = str(by or "").partition(":")
    if kind not in VERIFIERS or not ident.strip():
        raise ValueError(f"a path is verified by one of {VERIFIERS} as '<kind>:<id>', not {by!r}")
    if not isinstance(path_right, bool):
        raise ValueError("path_right must be true or false")
    out = copy.deepcopy(record)
    out["verified"] = {"by": f"{kind}:{ident.strip()}", "at": str(at)}
    out["path_right"] = path_right
    return out


def exportable(record: Dict[str, Any]) -> bool:
    """A record may become a training row only when it is verified by a
    tester or a user, its path was right, every figure came from a tool or a
    read source, and its question is not held back."""
    verified = record.get("verified")
    return (isinstance(verified, dict) and str(verified.get("by") or "").partition(":")[0] in VERIFIERS
            and record.get("path_right") is True
            and record.get("figures_from_tools") is True
            and record.get("question_set") in ("teaching", "frozen"))


def example_from(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The saved example of a verified record: the question shape and the
    order of the entries used, never a figure. None unless the record is
    verified with a right path and its question is from the teaching set."""
    if not (exportable(record) and record.get("question_set") == "teaching"):
        return None
    steps: List[str] = []
    for rnd in record.get("rounds") or []:
        for call in rnd.get("calls") or []:
            entry = call.get("entry") or call.get("tool")
            if entry:
                steps.append(str(entry))
    return {"schema": "example/1", "from_turn": record.get("turn"),
            "verified": copy.deepcopy(record.get("verified")),
            "entries_in_order": steps, "ended": record.get("ended")}
