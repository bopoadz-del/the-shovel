"""RestrictedPython sandbox — Reasoning Engine Plan 3.

Executes untrusted, LLM-generated Python in a jailed namespace so the
reasoning engine can run code it did not write without risking the host
process. The sandbox:

* compiles code with ``RestrictedPython.compile_restricted`` (AST rewrite),
* exposes only a whitelist of safe builtins and importable modules,
* injects session state as ordinary variables,
* captures ``print`` output and the value of a designated result variable,
* always returns a structured :class:`SandboxResult` — sandboxed code can
  fail, but it can never crash or escape the caller.

This is `app/core/sandbox.py`. It is unrelated to the legacy
`app/blocks/sandbox.py` block.
"""

from __future__ import annotations

import builtins as _py_builtins
import copy
import logging
import warnings
from types import ModuleType
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field
from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector

# --------------------------------------------------------------------------
# Policy: what sandboxed code may import and which builtins it may not call.
# --------------------------------------------------------------------------

#: Modules sandboxed code is allowed to ``import``. Everything else is denied.
#: ``numpy``/``sympy``/``pint`` are whitelisted because LLM-generated formulas
#: routinely reach for them (vectorised math, symbolic algebra, unit-aware
#: arithmetic); without the whitelist the sandbox would silently force retries.
#: ``scipy`` is whitelisted for the numeric methods closed-form formulas lack
#: (root finding, optimisation, interpolation, integration, ODE solvers), but
#: only through :data:`ALLOWED_SUBPACKAGES`.
#: Note: numpy exposes a handful of file-touching helpers (``loadtxt``,
#: ``savetxt``, ``fromfile``) — RestrictedPython's attribute guard is the
#: backstop, but treat this as a deliberate widening of the attack surface.
ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "app.lib.pm_computations",
        "app.schemas.cpm",  # CPMInput/Activity/Dependency — inputs to pm_computations
        "math",
        "statistics",
        "datetime",
        "json",
        "numpy",
        "scipy",
        "sympy",
        "pint",
    }
)

#: Whitelisted packages whose subpackages are allowed one by one. The root
#: itself imports, but a dotted path below it passes only when its first
#: component is listed. SciPy is the case in point: its numeric subpackages
#: are why it is allowed, while ``scipy.io`` reads and writes files and
#: ``scipy.datasets`` downloads over the network. A package not named here
#: keeps the root-wide rule.
ALLOWED_SUBPACKAGES: Mapping[str, frozenset[str]] = {
    "scipy": frozenset(
        {
            "constants",
            "integrate",
            "interpolate",
            "linalg",
            "optimize",
            "special",
            "stats",
        }
    ),
}

#: Builtins that are never exposed, even if RestrictedPython would allow them.
#: ``open`` / ``eval`` / ``exec`` / ``__import__`` are already absent from
#: ``safe_builtins``; listing them documents intent and guards against future
#: RestrictedPython releases that might add them back.
BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
)

#: Pure, side-effect-free builtins that ``RestrictedPython.safe_builtins``
#: conservatively omits but PM/formula code routinely needs (aggregates,
#: iteration helpers, container constructors). None of these can touch the
#: filesystem, network, or process — they are safe to expose.
_EXTRA_SAFE_BUILTINS: frozenset[str] = frozenset(
    {
        "sum", "min", "max", "all", "any", "enumerate", "map", "filter",
        "reversed", "list", "dict", "set", "frozenset", "iter", "next",
    }
)

#: Name of the variable the sandbox reads back as the structured result.
DEFAULT_RESULT_VAR = "result"


class SandboxError(Exception):
    """Raised for sandbox *setup* failures (bad config), never for errors
    inside sandboxed code — those are reported via :class:`SandboxResult`."""


class SandboxResult(BaseModel):
    """Structured outcome of one :func:`run_sandboxed` call.

    ``success`` is ``True`` only when the code compiled and ran without
    raising. On failure ``error`` holds a human-readable message and
    ``error_type`` the exception class name; ``stdout`` still contains
    whatever was printed before the failure.
    """

    success: bool
    stdout: str = ""
    #: The genuine computed value of the result variable. May be a
    #: non-JSON-serializable object (e.g. a datetime or a custom type); the
    #: caller is responsible for handling it — ``model_dump_json()`` will fail
    #: on such values unless the caller coerces ``result`` first.
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    #: Final values of injected/created variables, coerced to ``repr()``
    #: strings (best effort, for debugging). Always JSON-serializable.
    namespace: Dict[str, str] = Field(default_factory=dict)


def module_allowed(name: str) -> bool:
    """Whether sandboxed code may import or hold the module ``name``.

    An exact :data:`ALLOWED_MODULES` entry passes. Otherwise the dotted root
    must be whitelisted: numpy/sympy/pint load internal submodules
    (``numpy._core._methods``, ``sympy.core.numbers``, …) that a strict
    equality check would block. A root listed in :data:`ALLOWED_SUBPACKAGES`
    narrows that to its listed subpackages.
    """
    if name in ALLOWED_MODULES:
        return True
    parts = name.split(".")
    if parts[0] not in ALLOWED_MODULES:
        return False
    subpackages = ALLOWED_SUBPACKAGES.get(parts[0])
    if subpackages is None:
        return True
    return len(parts) > 1 and parts[1] in subpackages


def _module_reachable(name: str) -> bool:
    """Whether a module may be reached by attribute access.

    Allowed modules, plus the parent packages on the way to a dotted entry
    (``import app.lib.pm_computations`` binds ``app``, and the code walks
    ``app.lib`` to get there). Those parents are pass-through only: any other
    module under them fails :func:`module_allowed` on the next step.
    """
    if module_allowed(name):
        return True
    prefix = name + "."
    return any(entry.startswith(prefix) for entry in ALLOWED_MODULES)


def _deny_module(name: str) -> ImportError:
    return ImportError(
        f"Import of '{name}' is blocked by the sandbox. "
        f"Allowed modules: {', '.join(sorted(ALLOWED_MODULES))}."
    )


def _guarded_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """``_getattr_`` for sandboxed code: :func:`safer_getattr`, plus the
    import policy for any module the attribute yields.

    Packages load subpackages lazily on attribute access (``scipy.io``), which
    never passes through ``__import__``; without this check a whitelisted
    root would open every subpackage under it.
    """
    value = safer_getattr(obj, name, default)
    if isinstance(value, ModuleType) and not _module_reachable(value.__name__):
        raise _deny_module(value.__name__)
    return value


def _make_guarded_import() -> Any:
    """Build an ``__import__`` replacement that honours :data:`ALLOWED_MODULES`.

    A submodule import (``import app.lib.pm_computations``) is allowed only if
    its full dotted path is whitelisted; ``from os import path`` is denied
    because ``os`` is not in the whitelist.
    """

    def _guarded_import(
        name: str,
        globals: Optional[Dict[str, Any]] = None,
        locals: Optional[Dict[str, Any]] = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise ImportError("Relative imports are not permitted in the sandbox.")
        if not module_allowed(name):
            raise _deny_module(name)
        module = __import__(name, globals, locals, fromlist, level)
        # `from scipy import io` binds a subpackage without a dotted import
        # name; the names it binds pass the same policy.
        for item in fromlist or ():
            value = getattr(module, item, None)
            if isinstance(value, ModuleType) and not module_allowed(value.__name__):
                raise _deny_module(value.__name__)
        return module

    return _guarded_import


def _build_builtins() -> Dict[str, Any]:
    """Return the ``__builtins__`` mapping exposed to sandboxed code."""
    builtins: Dict[str, Any] = dict(safe_builtins)
    # Re-add pure aggregate/iteration builtins safe_builtins omits.
    for name in _EXTRA_SAFE_BUILTINS:
        builtins[name] = getattr(_py_builtins, name)
    # Strip anything explicitly blocked (defensive — most are absent already,
    # and this also guarantees nothing dangerous slipped in via the line above).
    for blocked in BLOCKED_BUILTINS:
        builtins.pop(blocked, None)
    # Provide a whitelisted importer so `import math` etc. still works.
    builtins["__import__"] = _make_guarded_import()
    return builtins


def _deepcopy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy injected state so sandboxed mutations cannot leak back to the
    caller. Copies each key individually: a key whose value cannot be
    deep-copied (e.g. a ``threading.Lock``) falls back to passing the original
    object through, and a :class:`UserWarning` is emitted so the caller knows
    isolation was downgraded for that key.
    """
    copied: Dict[str, Any] = {}
    leaked: list[str] = []
    for key, value in state.items():
        try:
            copied[key] = copy.deepcopy(value)
        except Exception:
            copied[key] = value
            leaked.append(key)
    if leaked:
        warnings.warn(
            "Sandbox isolation downgraded: could not deep-copy state key(s) "
            f"{sorted(leaked)}; the caller's original object(s) are exposed "
            "to sandboxed code and may be mutated.",
            stacklevel=2,
        )
    return copied


def _build_globals(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the exec globals: restricted builtins, RestrictedPython
    guards, and the injected session ``state``."""
    glb: Dict[str, Any] = {
        "__builtins__": _build_builtins(),
        # RestrictedPython rewrites code to call these helpers by name.
        "_print_": PrintCollector,
        "_getattr_": _guarded_getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_write_": full_write_guard,
    }
    # Inject session state as ordinary, mutable variables.
    glb.update(state)
    return glb


def run_sandboxed(
    code: str,
    state: Optional[Dict[str, Any]] = None,
    result_var: str = DEFAULT_RESULT_VAR,
) -> SandboxResult:
    """Execute ``code`` in the RestrictedPython jail and return the outcome.

    Args:
        code: Untrusted Python source to run.
        state: Variables injected into the namespace before execution
            (the session-state round-trip). Defaults to empty. Note: the
            result is read only from the names the executed code bound, so a
            state key named ``result_var`` (default ``"result"``) is injected
            but silently ignored on read-back unless the code reassigns it.
        result_var: Name of the variable read back into
            :attr:`SandboxResult.result` after a successful run. If the
            variable is never assigned, ``result`` stays ``None``.

    Returns:
        A :class:`SandboxResult`. This function never raises for errors
        *inside* the sandboxed code — compile errors, blocked imports,
        and runtime exceptions are all captured into the result.

    Raises:
        SandboxError: For a *setup* misconfiguration by the caller (an
            empty or non-string ``result_var``), never for sandboxed-code
            failures.
    """
    if not isinstance(result_var, str) or not result_var.isidentifier():
        raise SandboxError(
            f"result_var must be a valid Python identifier, got {result_var!r}."
        )
    # Deep-copy so sandboxed mutations of injected mutables (lists, dicts)
    # never leak back into the caller's session state.
    injected = _deepcopy_state(state or {})
    glb = _build_globals(injected)

    # --- compile (catches SyntaxError and RestrictedPython rejections) -----
    try:
        with warnings.catch_warnings():
            # RestrictedPython warns when `print` is used but `printed`
            # is never read — harmless here, we read the collector directly.
            warnings.simplefilter("ignore", SyntaxWarning)
            byte_code = compile_restricted(code, filename="<sandbox>", mode="exec")
    except SyntaxError as exc:
        return SandboxResult(
            success=False,
            error=f"Syntax error: {exc}",
            error_type="SyntaxError",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return SandboxResult(
            success=False,
            error=f"Code rejected by sandbox compiler: {exc}",
            error_type=type(exc).__name__,
        )

    # --- execute (catches every runtime failure) ---------------------------
    # One namespace, as for a module: with a separate locals dict, top-level
    # code behaves like a class body, and a function or lambda the snippet
    # defines cannot see the snippet's other top-level names (a residual
    # lambda calling two curve helpers raised NameError). Read-back still
    # covers only the names the code itself bound.
    preset = dict(glb)
    try:
        exec(byte_code, glb)
    except Exception as exc:
        local_ns = _bound_by_code(glb, preset)
        stdout = _extract_stdout(local_ns)
        return SandboxResult(
            success=False,
            stdout=stdout,
            error=str(exc) or type(exc).__name__,
            error_type=type(exc).__name__,
            namespace=_safe_namespace(local_ns),
        )

    local_ns = _bound_by_code(glb, preset)
    return SandboxResult(
        success=True,
        stdout=_extract_stdout(local_ns),
        result=local_ns.get(result_var),
        namespace=_safe_namespace(local_ns),
    )


def _bound_by_code(ns: Dict[str, Any], preset: Dict[str, Any]) -> Dict[str, Any]:
    """The names the executed code bound: new keys, or keys rebound to a
    different object. Injected state the code only read or mutated in place,
    builtins and guard helpers are not the code's own bindings, so they stay
    out of the read-back (an injected ``result`` is not read back unless the
    code reassigns it).
    """
    return {
        k: v for k, v in ns.items()
        if k not in preset or v is not preset[k]
    }


def _extract_stdout(local_ns: Dict[str, Any]) -> str:
    """Pull captured text out of RestrictedPython's print collector.

    RestrictedPython rewrites ``print(...)`` to append to a ``_print``
    :class:`PrintCollector` in the local namespace; calling it (or reading
    ``.txt``) yields the accumulated output.
    """
    collector = local_ns.get("_print")
    if collector is None:
        return ""
    try:
        if callable(collector):
            return collector()
        txt = getattr(collector, "txt", None)
        if isinstance(txt, list):
            return "".join(txt)
        return str(txt or "")
    except Exception:  # pragma: no cover - defensive
        logger.warning("sandbox stdout extract failed", exc_info=True)
        return ""


def _safe_namespace(local_ns: Dict[str, Any]) -> Dict[str, str]:
    """Return the user-visible namespace, dropping RestrictedPython internals
    (the ``_print`` collector, dunder names) for clean inspection.

    Values are coerced to ``repr()`` strings so the namespace is always
    JSON-serializable: it may contain functions, custom objects, etc., and
    serializing those directly would crash :meth:`SandboxResult.model_dump`.
    """
    safe: Dict[str, str] = {}
    for k, v in local_ns.items():
        if k.startswith("_"):
            continue
        try:
            safe[k] = repr(v)
        except Exception:  # pragma: no cover - defensive
            safe[k] = f"<unrepresentable {type(v).__name__}>"
    return safe
