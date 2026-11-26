"""Tools register themselves where they are defined, under the package that owns them.

A tool is ``@tool("name", owner="base" | "<hat>", display_name="...")`` on an
``async def handle(call: ToolCall) -> dict`` in its owner's package
(``app.agents.base`` or ``app.agents.hats.<hat>``). The agent's tool loop
looks the name up here; a hat's manifest lists exactly the tools its package
registers. Adding a hat is a new package -- no core edit. ``display_name`` is
what a user reads when an answer credits the tool; the name is internal.

A tool may declare the KIND of result it returns (``kind=``, one of
``RESULT_KINDS``). A ``"computed"`` tool returns figures it worked out
deterministically and declares ``statement=``: a function from its result to
the figures it states (``Statement``). When the model's synthesis is missing,
that statement is the answer -- by kind, never by a list of tool names. A
``"search"`` tool returns passages: its result is evidence for an answer and
is never shown as the answer itself.

A computed tool also DECLARES the reason codes its failures carry
(``reason_codes=``). Only a declared code is ever shown to a user (as
"code: X"); any other error text a tool returns is never shown at all.

A block that needs its arguments shaped before it runs registers an adapter
the same way, in its owner's ``blocks`` module: ``@block_args("name",
owner=...)`` on ``async def adapt(call, block_input, block_params)``, which
returns the new ``(block_input, block_params)`` -- or a result dict when it
answers the call itself.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

OWNERS = ("base", "commercial", "contracts", "design", "planning", "procurement",
          "qaqc", "quantities", "safety")


@dataclass
class ToolCall:
    """Everything a tool handler may read about the call it is answering."""
    agent: Any
    name: str
    args: Dict[str, Any]
    tool_call: Dict[str, Any] = field(default_factory=dict)
    api_key: Optional[str] = None
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    depth: int = 0
    call_stack: List[str] = field(default_factory=list)
    user_message: Optional[str] = None
    history: Optional[list] = None
    #: The turn's ``app.core.rag.handed_set.SearchTurn`` (search cap and the
    #: chunks already handed), when the chat loop runs one.
    search_turn: Any = None


Handler = Callable[[ToolCall], Awaitable[Dict[str, Any]]]

#: "computed": deterministic figures (needs ``statement``); "search":
#: passages, never an answer on their own; "": undeclared.
RESULT_KINDS = ("", "computed", "search")

#: The shape of a reason code a tool may declare.
_REASON_CODE_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")


def _reason_codes(owner_kind: str, name: str, codes: Any) -> Tuple[str, ...]:
    out = tuple(codes or ())
    for code in out:
        if not isinstance(code, str) or not _REASON_CODE_RE.fullmatch(code):
            raise ValueError(f"{owner_kind} {name!r}: reason code {code!r} is not a snake_case code")
    return out


@dataclass(frozen=True)
class Statement:
    """One figure a computed tool states, in the answer contract's terms:
    the figure and its unit, what was counted or summed, the source it was
    computed from, and the source rows behind it."""
    #: None: this part of the result did not return (see ``failure``).
    figure: Optional[float]
    #: What the figure is. For ``basis`` "count"/"sum" it is the measure
    #: (a data word); otherwise the tool's own name for the figure.
    what: str
    unit: str = ""
    source: str = ""
    rows: Tuple[str, ...] = ()
    #: The figure is a count of rows, not a sum of a measure.
    count: Optional[int] = None
    #: "count" or "sum" over matching rows; "" for any other figure. The
    #: answer's wording for a basis comes from the labels of the question's
    #: language (app.agents.core.computed_answer), never from the tool.
    basis: str = ""
    #: The words the rows were matched on, and the section words.
    terms: Tuple[str, ...] = ()
    within: Tuple[str, ...] = ()
    #: Why this part did not return: a reason code or the tool's error.
    failure: str = ""
    #: The tool's own working for the figure (e.g. a calculator's note),
    #: shown as data under the figure.
    working: str = ""
    #: The figure each row in ``rows`` carries, in the same order, where the
    #: tool states it (None for a row it states without one). Empty when the
    #: tool states no per-row figure.
    row_figures: Tuple[Optional[float], ...] = ()


Stater = Callable[[Dict[str, Any]], List[Statement]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    owner: str
    handler: Handler
    #: What a user reads when an answer credits this tool.
    display_name: str = ""
    #: The tool's function schema, when it declares one where it is defined.
    schema: Optional[Dict[str, Any]] = None
    #: Each parameter as declared (app.lib.formula_registry.Param: label,
    #: kind, unit, range, required). None until the tool declares them.
    params: Optional[Dict[str, Any]] = None
    #: The kind of result the tool returns (``RESULT_KINDS``).
    kind: str = ""
    #: For a ``"computed"`` tool: its result -> the figures it states.
    statement: Optional[Stater] = None
    #: The reason codes the tool's failures carry; the only error words of
    #: the tool a user may be shown.
    reason_codes: Tuple[str, ...] = ()
    #: Catalogue declarations, as on a formula (app.lib.formula_registry):
    #: optional; a missing one is listed in the catalogue entry's ``gaps``.
    when_to_use: str = ""
    when_not: str = ""
    examples: Tuple[Dict[str, Any], ...] = ()


_TOOLS: Dict[str, ToolSpec] = {}


def tool(name: str, *, owner: str, display_name: str,
         schema: Optional[Dict[str, Any]] = None,
         params: Optional[Dict[str, Any]] = None, kind: str = "",
         statement: Optional[Stater] = None,
         reason_codes: Tuple[str, ...] = (),
         when_to_use: str = "", when_not: str = "",
         examples: Tuple[Dict[str, Any], ...] = ()) -> Callable[[Handler], Handler]:
    from app.lib.formula_registry import declared_examples

    if owner not in OWNERS:
        raise ValueError(f"unknown tool owner {owner!r}; one of {OWNERS}")
    if not (display_name or "").strip():
        raise ValueError(f"tool {name!r} needs a display_name")
    if kind not in RESULT_KINDS:
        raise ValueError(f"tool {name!r}: unknown result kind {kind!r}; one of {RESULT_KINDS}")
    if (kind == "computed") != (statement is not None):
        raise ValueError(f"tool {name!r}: a computed tool, and only a computed tool, declares a statement")
    codes = _reason_codes("tool", name, reason_codes)
    shown = declared_examples("tool", name, examples)

    def register(fn: Handler) -> Handler:
        existing = _TOOLS.get(name)
        if existing is not None and existing.handler is not fn:
            raise ValueError(f"tool {name!r} is registered twice")
        _TOOLS[name] = ToolSpec(name, owner, fn, display_name.strip(), schema,
                                dict(params) if params is not None else None,
                                kind, statement, codes,
                                (when_to_use or "").strip(), (when_not or "").strip(), shown)
        return fn
    return register


Adapter = Callable[[ToolCall, Any, Any], Awaitable[Union[Tuple[Any, Any], Dict[str, Any]]]]


@dataclass(frozen=True)
class BlockArgsSpec:
    name: str
    owner: str
    adapt: Adapter


_BLOCK_ARGS: Dict[str, BlockArgsSpec] = {}


def block_args(name: str, *, owner: str) -> Callable[[Adapter], Adapter]:
    if owner not in OWNERS:
        raise ValueError(f"unknown block owner {owner!r}; one of {OWNERS}")

    def register(fn: Adapter) -> Adapter:
        existing = _BLOCK_ARGS.get(name)
        if existing is not None and existing.adapt is not fn:
            raise ValueError(f"block {name!r} has two argument adapters")
        _BLOCK_ARGS[name] = BlockArgsSpec(name, owner, fn)
        return fn
    return register


def get_block_args(name: str) -> Optional[BlockArgsSpec]:
    load()
    return _BLOCK_ARGS.get(name)


def block_args_of(owner: str) -> List[str]:
    load()
    return sorted(n for n, s in _BLOCK_ARGS.items() if s.owner == owner)


@dataclass(frozen=True)
class ResultSpec:
    """The declared result kind of a name the model calls that is not a
    registered tool (a block, or a manifest action aliased to one)."""
    name: str
    owner: str
    display_name: str
    kind: str
    statement: Optional[Stater] = None
    reason_codes: Tuple[str, ...] = ()


_RESULTS: Dict[str, ResultSpec] = {}
# While a deferred declarer runs, its declarations are collected here (per
# thread) and committed to _RESULTS only when it returns: one that raises
# part-way leaves nothing behind, so its retry cannot hit "declared twice".
_STAGE = threading.local()


def declare_result(name: str, *, owner: str, display_name: str, kind: str,
                   statement: Optional[Stater] = None,
                   reason_codes: Tuple[str, ...] = ()) -> None:
    """Declare what a block's (or an aliased action's) result is, with the
    same rules as ``tool(kind=, statement=)``. A registered tool declares
    its kind on ``tool`` instead."""
    if owner not in OWNERS:
        raise ValueError(f"unknown result owner {owner!r}; one of {OWNERS}")
    if not (display_name or "").strip():
        raise ValueError(f"result {name!r} needs a display_name")
    if kind not in RESULT_KINDS or not kind:
        raise ValueError(f"result {name!r}: unknown result kind {kind!r}; one of {RESULT_KINDS[1:]}")
    if (kind == "computed") != (statement is not None):
        raise ValueError(f"result {name!r}: a computed result, and only a computed result, declares a statement")
    spec = ResultSpec(name, owner, display_name.strip(), kind, statement,
                      _reason_codes("result", name, reason_codes))
    staged = getattr(_STAGE, "results", None)
    existing = _RESULTS.get(name)
    if existing is None and staged is not None:
        existing = staged.get(name)
    if (existing is not None and existing != spec) or name in _TOOLS:
        raise ValueError(f"result {name!r} is declared twice")
    (_RESULTS if staged is None else staged)[name] = spec


_DEFERRED: List[Callable[[], None]] = []


def declare_results_when_read(declarer: Callable[[], None]) -> None:
    """Run ``declarer`` (which calls ``declare_result``) on the first read of
    the result declarations, not at import. For a declaration that needs
    another registry -- one that imports libraries too heavy for app start --
    so ``load()`` at boot stays cheap."""
    _DEFERRED.append(declarer)


_DEFERRED_LOCK = threading.RLock()
_DEFERRED_RUNNING = False
_LOG = logging.getLogger(__name__)
#: After a declarer fails it is not run again for this many seconds: a
#: declarer that always fails (a heavy import that breaks) must not be re-run,
#: under the lock, by every tool call. Reads in between see what is declared.
DEFERRED_RETRY_INTERVAL_S = 30.0
#: The clock the back-off reads (monotonic; a test substitutes a fake one).
_clock: Callable[[], float] = time.monotonic
_RETRY_AT: Dict[Callable[[], None], float] = {}
# Set while a worker thread drains the list for a read made on an event loop.
_OFFLOOP_LOCK = threading.Lock()
_OFFLOOP_RUNNING = False


def _due() -> bool:
    """Is any pending declarer allowed to run now? Unlocked: a read while
    every pending declarer is backing off must not touch the lock."""
    now = _clock()
    return any(_RETRY_AT.get(d, 0.0) <= now for d in list(_DEFERRED))


def _on_event_loop() -> bool:
    """Is this thread running an event loop? (``_get_running_loop`` is the
    non-raising form of ``get_running_loop``, exported by asyncio.)"""
    return asyncio._get_running_loop() is not None


def _drain_off_loop() -> None:
    global _OFFLOOP_RUNNING
    try:
        _run_pending()
    finally:
        with _OFFLOOP_LOCK:
            _OFFLOOP_RUNNING = False


def _start_off_loop() -> None:
    """Hand the pending declarers to a worker thread (at most one at a time)."""
    global _OFFLOOP_RUNNING
    with _OFFLOOP_LOCK:
        if _OFFLOOP_RUNNING:
            return
        _OFFLOOP_RUNNING = True
    threading.Thread(target=_drain_off_loop, name="deferred-declarers", daemon=True).start()


def _run_pending() -> None:
    global _DEFERRED_RUNNING
    with _DEFERRED_LOCK:
        if _DEFERRED_RUNNING:
            return
        _DEFERRED_RUNNING = True
        try:
            for declarer in list(_DEFERRED):
                if _RETRY_AT.get(declarer, 0.0) > _clock():
                    continue
                _STAGE.results = staged = {}
                try:
                    declarer()
                    for name, spec in staged.items():
                        if _RESULTS.get(name, spec) != spec:
                            raise ValueError(f"result {name!r} is declared twice")
                except Exception as exc:  # retried after the back-off interval
                    _RETRY_AT[declarer] = _clock() + DEFERRED_RETRY_INTERVAL_S
                    _LOG.warning("deferred result declarer %s failed (%s); retried in %.0f s",
                                 getattr(declarer, "__qualname__", "?"), type(exc).__name__,
                                 DEFERRED_RETRY_INTERVAL_S)
                    continue
                finally:
                    _STAGE.results = None
                _RESULTS.update(staged)        # commit: all of it, only on success
                _DEFERRED.remove(declarer)
                _RETRY_AT.pop(declarer, None)
        finally:
            _DEFERRED_RUNNING = False


def declare_deferred_results() -> None:
    """Run every waiting declarer, under one lock, so a second thread's first
    read waits until the declarations exist. A declarer's declarations are
    collected and committed only when it SUCCEEDS: one that raises leaves
    nothing declared, stays in the list, is logged by its exception class, and
    is tried again no sooner than ``DEFERRED_RETRY_INTERVAL_S`` later. A read
    made from inside a declarer (same thread) does not re-enter the loop.

    The lock is taken ONLY while a pending declarer is due: every tool call
    reads the declarations, so once the list is empty (or everything left is
    backing off) a read must never queue on it.

    A read made ON an event loop never runs a declarer itself (the declarers
    import heavy libraries; seconds of import there stall every user) and
    never waits on the lock: it hands the work to a worker thread and reads
    what is declared now. The app declares before it serves (app.main's
    lifespan, off the loop), so such a read is only the fallback."""
    if not _DEFERRED or not _due():
        return
    if _on_event_loop():
        _start_off_loop()
        return
    _run_pending()


def get(name: str) -> Optional[ToolSpec]:
    load()
    return _TOOLS.get(name)


def result_spec(name: str) -> Optional[Union[ToolSpec, ResultSpec]]:
    """What ``name`` declares about its result: the tool's spec, else the
    block's result declaration, else None."""
    load()
    declare_deferred_results()
    return _TOOLS.get(name) or _RESULTS.get(name)


def tools_of_kind(kind: str) -> List[str]:
    load()
    declare_deferred_results()
    names = {n for n, s in _TOOLS.items() if s.kind == kind}
    names.update(n for n, s in _RESULTS.items() if s.kind == kind)
    return sorted(names)


def tools_of(owner: str) -> List[str]:
    load()
    return sorted(n for n, s in _TOOLS.items() if s.owner == owner)


_loaded = False


def tool_modules() -> List[str]:
    """Every module of ``app.agents.base`` and of each ``app.agents.hats.<hat>``
    (sub-packages such as ``formulas`` excepted) -- discovered, so a new hat
    is a new package and no core edit."""
    import importlib
    import pkgutil

    import app.agents.hats as hats

    packages = ["app.agents.base"] + [
        f"app.agents.hats.{info.name}"
        for info in sorted(pkgutil.iter_modules(hats.__path__), key=lambda i: i.name)
        if info.ispkg
    ]
    mods = []
    for pkg in packages:
        try:
            package = importlib.import_module(pkg)
        except ModuleNotFoundError:
            continue
        mods.extend(f"{pkg}.{info.name}"
                    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name)
                    if not info.ispkg)
    return mods


def load() -> None:
    global _loaded
    if _loaded:
        return
    import importlib

    for mod in tool_modules():
        importlib.import_module(mod)
    _loaded = True
