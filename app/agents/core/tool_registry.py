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

A block that needs its arguments shaped before it runs registers an adapter
the same way, in its owner's ``blocks`` module: ``@block_args("name",
owner=...)`` on ``async def adapt(call, block_input, block_params)``, which
returns the new ``(block_input, block_params)`` -- or a result dict when it
answers the call itself.
"""
from __future__ import annotations

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


_TOOLS: Dict[str, ToolSpec] = {}


def tool(name: str, *, owner: str, display_name: str,
         schema: Optional[Dict[str, Any]] = None,
         params: Optional[Dict[str, Any]] = None, kind: str = "",
         statement: Optional[Stater] = None) -> Callable[[Handler], Handler]:
    if owner not in OWNERS:
        raise ValueError(f"unknown tool owner {owner!r}; one of {OWNERS}")
    if not (display_name or "").strip():
        raise ValueError(f"tool {name!r} needs a display_name")
    if kind not in RESULT_KINDS:
        raise ValueError(f"tool {name!r}: unknown result kind {kind!r}; one of {RESULT_KINDS}")
    if (kind == "computed") != (statement is not None):
        raise ValueError(f"tool {name!r}: a computed tool, and only a computed tool, declares a statement")

    def register(fn: Handler) -> Handler:
        existing = _TOOLS.get(name)
        if existing is not None and existing.handler is not fn:
            raise ValueError(f"tool {name!r} is registered twice")
        _TOOLS[name] = ToolSpec(name, owner, fn, display_name.strip(), schema,
                                dict(params) if params is not None else None,
                                kind, statement)
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


def get(name: str) -> Optional[ToolSpec]:
    load()
    return _TOOLS.get(name)


def tools_of_kind(kind: str) -> List[str]:
    load()
    return sorted(n for n, s in _TOOLS.items() if s.kind == kind)


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
