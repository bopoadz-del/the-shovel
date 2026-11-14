"""The evidence-fetch loop: plan -> actions -> coverage -> re-query (item 2).

Orchestrates the multi-shape retrieval over an evidence plan. It is PURE
orchestration over callables, so it is unit-testable without a live index:
the runtime supplies the retrieval actions (semantic / lexical / structural),
each ``(plan_item_text) -> evidence_text``. This module decides which plan
items still lack evidence, which fetched blocks are handed under the loop's
budget, and bounds the rounds.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.core.rag.evidence_budget import (
    CONTEXT_TRUNCATED_NOTE,
    cap_evidence_text,
    loop_item_char_budget,
    loop_total_char_budget,
)
from app.core.rag.evidence_plan import (
    ROUND_LIMIT,
    EvidencePlan,
    is_platform_note,
    missing_plan_items,
)

#: A retrieval action: a plan item (phrase) in, evidence text out.
RetrievalAction = Callable[[str], str]


@dataclass(frozen=True)
class RoundRecord:
    """One re-query round: what was still missing going in, what remained after."""
    number: int
    queried_items: tuple[str, ...]
    missing_after: tuple[str, ...]


@dataclass
class EvidenceRun:
    """The outcome of the loop: combined evidence, remaining gaps, and rounds."""
    plan: EvidencePlan
    evidence: str
    missing: tuple[str, ...]
    rounds: list[RoundRecord] = field(default_factory=list)
    #: Why the loop stopped: ``covered`` (nothing missing), ``no_new_evidence``
    #: (no still-missing item can yield a line not already held) or
    #: ``round_limit`` (``max_rounds`` exhausted while items still yielded).
    stop_reason: str = ""
    #: Characters the loop added to the first pass, per plan item and total.
    added_chars: int = 0
    item_chars: dict[str, int] = field(default_factory=dict)
    #: Fetched blocks (one excerpt, or one note with its rows) handed,
    #: skipped as already held this turn, and withheld by the loop's budget.
    added_blocks: int = 0
    duplicate_blocks: int = 0
    withheld_blocks: int = 0
    #: True when the whole-turn backstop cut the merged text.
    backstop_cut: bool = False


_NOTE_LINES = frozenset(
    line.strip() for line in CONTEXT_TRUNCATED_NOTE.splitlines() if line.strip()
)

#: The excerpt marker every handed chunk carries (``inject.chunk_marker``):
#: it names the chunk's document and position, which is the chunk's identity.
_MARKER_RE = re.compile(r"^\[doc_id=(\S+) chunk=(-?\d+) ")


def _evidence_lines(text: str) -> set[str]:
    """The evidence lines of ``text`` (chunks/rows are newline-joined).

    The budget's truncation note is bookkeeping, not evidence, so it never
    counts as something a round fetched.
    """
    return {
        line.strip() for line in (text or "").splitlines()
        if line.strip() and line.strip() not in _NOTE_LINES
    }


def _blocks(text: str) -> list[str]:
    """``text`` split into blocks: one excerpt (its marker line and the lines
    that follow it), or one platform note with the lines that follow it (a
    row list under its note). Text with neither is one block."""
    blocks: list[list[str]] = []
    for line in (text or "").splitlines():
        if not blocks or _MARKER_RE.match(line) or is_platform_note(line):
            blocks.append([])
        blocks[-1].append(line)
    out = ("\n".join(lines).strip() for lines in blocks)
    return [block for block in out if block]


def _block_key(block: str) -> tuple:
    """An excerpt is identified by its document and chunk position, so the
    same chunk fetched by two actions, two items or two rounds -- or already
    in the first pass -- is one block; any other block by its text."""
    match = _MARKER_RE.match(block)
    if match:
        return ("chunk", match.group(1), match.group(2))
    return ("text", block)


def _interleave(lists: Sequence[Sequence[str]]) -> list[str]:
    """Round-robin over ``lists``: the first of each, then the second of each.

    Each action returns its blocks best first, so every action gets a share
    of an item's budget instead of the first action taking all of it.
    """
    out: list[str] = []
    depth = max((len(blocks) for blocks in lists), default=0)
    for i in range(depth):
        out.extend(blocks[i] for blocks in lists if i < len(blocks))
    return out


@dataclass
class _Selection:
    """What the loop has handed so far this turn, and its budget spend."""
    item_budget: int
    total_budget: int
    held_keys: set[tuple]
    held_lines: set[str]
    item_chars: dict[str, int] = field(default_factory=dict)
    added_chars: int = 0
    added_blocks: int = 0
    duplicate_blocks: int = 0
    withheld_blocks: int = 0

    def admit(self, item: str, block: str) -> bool | None:
        """Hand ``block`` for ``item`` if it is new and fits both budgets.

        Returns None when the block was not handed, else whether it carried a
        line not already held.
        """
        if block in _NOTE_LINES:
            # An action's own truncation note: the loop states sampling once.
            self.withheld_blocks += 1
            return None
        key = _block_key(block)
        if key in self.held_keys:
            self.duplicate_blocks += 1
            return None
        cost = len(block) + 1
        if (self.item_chars.get(item, 0) + cost > self.item_budget
                or self.added_chars + cost > self.total_budget):
            self.withheld_blocks += 1
            return None
        self.held_keys.add(key)
        self.item_chars[item] = self.item_chars.get(item, 0) + cost
        self.added_chars += cost
        self.added_blocks += 1
        lines = _evidence_lines(block)
        new = bool(lines - self.held_lines)
        self.held_lines |= lines
        return new


def _select_round(
    selection: _Selection,
    queued: Sequence[tuple[str, list[str]]],
) -> tuple[list[str], set[str]]:
    """Hand blocks round-robin across plan items, under the budgets.

    Every item gets its next block before any item gets a second one, so a
    later plan item is never starved by an earlier item's large fetch.
    Returns the handed blocks in order and the items that handed a new line.
    """
    handed: list[str] = []
    fresh: set[str] = set()
    queues = [(item, list(blocks)) for item, blocks in queued]
    while any(blocks for _, blocks in queues):
        for item, blocks in queues:
            while blocks:
                block = blocks.pop(0)
                new = selection.admit(item, block)
                if new is None:
                    continue
                handed.append(block)
                if new:
                    fresh.add(item)
                break
    return handed, fresh


def gather_evidence(
    plan: EvidencePlan,
    actions: Sequence[RetrievalAction],
    initial_evidence: str = "",
    max_rounds: int = ROUND_LIMIT,
) -> EvidenceRun:
    """Fetch evidence for a plan, re-querying missing items up to ``max_rounds``.

    ``initial_evidence`` is the first-pass retrieval text (round 1). For each
    subsequent round, every still-missing plan item is given to every action.
    The fetched text is not appended whole: it is split into blocks (one
    excerpt, or one note with its rows); a block already held this turn --
    in the first pass or an earlier fetch -- is skipped, and the rest are
    handed round-robin across items under a declared budget per plan item
    and in total (``evidence_budget.loop_item_char_budget`` /
    ``loop_total_char_budget``). Coverage is then re-checked on the first
    pass plus the handed blocks. The loop stops when nothing is missing,
    when a round hands nothing new, or when ``max_rounds`` is exhausted --
    round ``max_rounds + 1`` never runs.

    "Nothing new" is measured against the evidence lines already held. The
    actions are deterministic over a fixed index, so an item whose round
    handed no line not already held would hand nothing new again: it is not
    re-queried in later rounds (it stays in ``missing``). When no missing
    item is left to query, the loop ends with ``stop_reason="no_new_evidence"``.
    """
    first = initial_evidence or ""
    parts = [first] if first.strip() else []
    missing = missing_plan_items(plan, first)
    rounds: list[RoundRecord] = []
    selection = _Selection(
        item_budget=loop_item_char_budget(),
        total_budget=loop_total_char_budget(),
        held_keys={_block_key(block) for block in _blocks(first)},
        held_lines=_evidence_lines(first),
    )
    exhausted: set[str] = set()
    for number in range(1, max_rounds + 1):
        queried = tuple(item for item in missing if item not in exhausted)
        if not queried:
            break
        queued = [
            (item, _interleave([_blocks(action(item) or "") for action in actions]))
            for item in queried
        ]
        handed, fresh = _select_round(selection, queued)
        exhausted.update(item for item in queried if item not in fresh)
        parts.extend(handed)
        new_missing = missing_plan_items(plan, "\n".join(parts))
        rounds.append(RoundRecord(number=number, queried_items=queried,
                                  missing_after=new_missing))
        missing = new_missing
    if not missing:
        stop_reason = "covered"
    elif all(item in exhausted for item in missing):
        stop_reason = "no_new_evidence"
    else:
        stop_reason = "round_limit"
    combined = "\n".join(parts)
    # Whole-turn backstop: the declared loop budget above keeps the merge
    # under it; it still bounds a first pass that is itself oversized. Keep
    # the front (relevance order) and record the truncation so the answer
    # says the table was sampled.
    combined, cut = cap_evidence_text(combined)
    if cut or selection.withheld_blocks:
        combined = combined.rstrip() + CONTEXT_TRUNCATED_NOTE
    return EvidenceRun(
        plan=plan, evidence=combined, missing=missing, rounds=rounds,
        stop_reason=stop_reason, added_chars=selection.added_chars,
        item_chars=dict(selection.item_chars),
        added_blocks=selection.added_blocks,
        duplicate_blocks=selection.duplicate_blocks,
        withheld_blocks=selection.withheld_blocks, backstop_cut=cut,
    )
