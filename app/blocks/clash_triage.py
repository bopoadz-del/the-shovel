"""Clash triage: the step after detection (S-K3 item 5).

The AABB pass in ``bim_extractor`` answers "which pairs overlap". A
coordination meeting needs three more things, and this module derives them
from the clash list itself plus an optional table the caller supplies:

* **Cause pattern.** Clashes that share an element form one connected group.
  One duct run through six beams is ONE routing problem, not six. Each group
  names the element that takes part in the most of its clashes. When two or
  more tie for that, none is named: the tie is reported instead of an
  arbitrary pick.
* **Discipline pairs.** It counts clashes per unordered category pair, with
  the deepest overlap in each pair.
* **Who moves.** This comes only from the caller's ``move_priority`` table
  (the project's coordination protocol). The table has rows of
  ``{"category": ..., "rank": int}`` or ``{"ifc_type": ..., "rank": int}``.
  A LOWER rank holds its position, and the element with the higher rank
  moves. An ``ifc_type`` row outranks a ``category`` row for the same
  element. With no table, an unranked element or equal ranks, ``moves`` is
  None and the reason is given. The code holds no discipline order of its own.

The AABB pass cannot tell a real clash from a bounding-box artefact (a
diagonal pipe's box is mostly air). This module does not pretend to: every
group says that a mesh check settles it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

_MATCH_KEYS = ("ifc_type", "category")


def _clash_side(clash: Dict, side: str) -> Dict[str, Any]:
    """Element id, category and IFC type of one side ('a' or 'b') of a clash."""
    return {
        "id": clash.get(f"element_{side}"),
        "category": clash.get(f"category_{side}", clash.get("category")),
        "ifc_type": clash.get(f"ifc_type_{side}"),
    }


def parse_move_priority(table: Any) -> Dict[Tuple[str, str], int]:
    """Validate a move-priority table into {(match_key, value): rank}.

    A bad row rejects the whole table with a ValueError that names the row. A
    half-read protocol would decide some moves and silently skip others.
    """
    if not isinstance(table, list) or not table:
        raise ValueError("move_priority must be a non-empty list of rows")
    ranks: Dict[Tuple[str, str], int] = {}
    for i, row in enumerate(table):
        if not isinstance(row, dict):
            raise ValueError(f"move_priority row {i} is not an object")
        keys = [k for k in _MATCH_KEYS if k in row]
        if len(keys) != 1:
            raise ValueError(
                f"move_priority row {i} must name exactly one of "
                f"{', '.join(_MATCH_KEYS)}"
            )
        value = row[keys[0]]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"move_priority row {i}: {keys[0]} must be text")
        rank = row.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError(f"move_priority row {i}: rank must be an integer")
        key = (keys[0], value.strip())
        if key in ranks and ranks[key] != rank:
            raise ValueError(
                f"move_priority row {i}: {keys[0]} {value!r} already has "
                f"rank {ranks[key]}"
            )
        ranks[key] = rank
    return ranks


def _rank_of(side: Dict[str, Any], ranks: Dict[Tuple[str, str], int]) -> Optional[int]:
    for key in _MATCH_KEYS:  # ifc_type first: the more specific row wins
        value = side.get(key)
        if isinstance(value, str) and (key, value) in ranks:
            return ranks[(key, value)]
    return None


def decide_move(clash: Dict, ranks: Optional[Dict[Tuple[str, str], int]]) -> Tuple[Optional[str], str]:
    """(element id that moves, reason). The id is None when the table cannot decide."""
    if not ranks:
        return None, "no move-priority table supplied"
    a, b = _clash_side(clash, "a"), _clash_side(clash, "b")
    ra, rb = _rank_of(a, ranks), _rank_of(b, ranks)
    unranked = [s for s, r in ((a, ra), (b, rb)) if r is None]
    if unranked:
        names = ", ".join(
            str(s.get("ifc_type") or s.get("category") or s.get("id"))
            for s in unranked
        )
        return None, f"no rank in the table for {names}"
    if ra == rb:
        return None, f"both sides have rank {ra}"
    mover, holder = (a, b) if ra > rb else (b, a)
    return mover["id"], (
        f"rank {max(ra, rb)} ({mover.get('ifc_type') or mover.get('category')}) "
        f"moves for rank {min(ra, rb)} "
        f"({holder.get('ifc_type') or holder.get('category')})"
    )


def _overlap(clash: Dict) -> Optional[float]:
    v = clash.get("overlap_mm")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def triage_clashes(clashes: List[Dict], move_priority: Any = None) -> Dict[str, Any]:
    """Cause groups, discipline pairs and (with a table) who moves.

    Each clash that has both element ids gets ``moves`` and ``moves_reason``
    added in place. The returned summary is small enough to lead a chat
    payload.
    """
    ranks = parse_move_priority(move_priority) if move_priority is not None else None

    usable = [c for c in clashes if c.get("element_a") and c.get("element_b")]
    # Union-find over element ids: clashes sharing an element are one group.
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in usable:
        ra, rb = find(c["element_a"]), find(c["element_b"])
        if ra != rb:
            parent[rb] = ra

    groups: Dict[str, List[Dict]] = {}
    for c in usable:
        groups.setdefault(find(c["element_a"]), []).append(c)

    pattern_groups: List[Dict[str, Any]] = []
    isolated = 0
    for members in groups.values():
        if len(members) < 2:
            isolated += 1
            continue
        degree: Dict[str, int] = {}
        label: Dict[str, str] = {}
        for c in members:
            for side in ("a", "b"):
                s = _clash_side(c, side)
                degree[s["id"]] = degree.get(s["id"], 0) + 1
                label[s["id"]] = str(s.get("ifc_type") or s.get("category") or "")
        top = max(degree.values())
        leaders = sorted(e for e, d in degree.items() if d == top)
        overlaps = [o for o in (_overlap(c) for c in members) if o is not None]
        pattern_groups.append({
            "clash_count": len(members),
            "element_count": len(degree),
            "common_element": leaders[0] if len(leaders) == 1 else None,
            "common_element_type": label[leaders[0]] if len(leaders) == 1 else None,
            "common_element_clashes": top,
            "tied_elements": leaders if len(leaders) > 1 else [],
            "max_overlap_mm": max(overlaps) if overlaps else None,
        })
    pattern_groups.sort(
        key=lambda g: (-g["clash_count"], -(g["max_overlap_mm"] or 0.0))
    )

    pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for c in usable:
        cats = tuple(sorted(
            str(_clash_side(c, s).get("category") or "unknown") for s in ("a", "b")
        ))
        entry = pairs.setdefault(cats, {"pair": list(cats), "clash_count": 0, "max_overlap_mm": None})
        entry["clash_count"] += 1
        o = _overlap(c)
        if o is not None and (entry["max_overlap_mm"] is None or o > entry["max_overlap_mm"]):
            entry["max_overlap_mm"] = o
    by_pair = sorted(pairs.values(), key=lambda e: (-e["clash_count"], e["pair"]))

    moves_decided = 0
    for c in usable:
        mover, reason = decide_move(c, ranks)
        c["moves"] = mover
        c["moves_reason"] = reason
        moves_decided += mover is not None

    return {
        "clashes_triaged": len(usable),
        "pattern_groups": pattern_groups,
        "isolated_clashes": isolated,
        "by_discipline_pair": by_pair,
        "moves_decided": moves_decided,
        "moves_undecided": len(usable) - moves_decided,
        "move_table_supplied": ranks is not None,
        "real_or_artefact": (
            "not decided: a bounding-box overlap cannot tell a real clash from "
            "box air around a sloped or curved element; confirm on the mesh"
        ),
    }
