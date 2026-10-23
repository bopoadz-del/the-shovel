"""No module under app/ may import a sibling as though it were top-level.

Audit 2026-08-12, agents subsystem.

`app/agents/catalog.py` did `from models import AgentManifest, ...` and
`app/agents/activation.py` did `from catalog import ...`. That is Python 2
implicit-relative-import syntax. Under Python 3 it only resolves if
`app/agents` itself is on `sys.path`, which meant:

    >>> import app.agents.catalog
    ModuleNotFoundError: No module named 'models'

So no application code could reach the discipline hats at all -- the feature
was unreachable from the package it lives in. Its own tests worked only
because they did `sys.path.insert(0, app/agents)` first, which ALSO meant the
tests and the application would have loaded two separate module objects for
the same file: a pydantic class from `models` fails an isinstance check
against the identical class from `app.agents.models`.

Static, so it costs nothing and cannot be defeated by import side effects.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

APP = Path("app")


def _implicit_relative_imports() -> list[tuple[str, int, str]]:
    """(file, line, module) for every import of a same-directory sibling.

    A plain `import X` / `from X import ...` is implicit-relative when a file
    named `X.py` sits next to the importer. Explicit relative imports
    (`from .X import`) have level > 0 and are skipped -- those are correct.
    Stdlib and third-party names are excluded by construction: they do not
    have a sibling file of that name.
    """
    offenders: list[tuple[str, int, str]] = []
    for path in sorted(APP.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        siblings = {p.stem for p in path.parent.glob("*.py")} - {path.stem}
        siblings |= {p.name for p in path.parent.iterdir() if p.is_dir()
                     and (p / "__init__.py").is_file()}
        if not siblings:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module and node.module.split(".")[0] in siblings:
                    offenders.append((str(path), node.lineno, node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in siblings:
                        offenders.append((str(path), node.lineno, alias.name))
    return offenders


def test_no_module_imports_a_sibling_as_top_level():
    offenders = _implicit_relative_imports()
    assert not offenders, (
        "these imports only resolve if their own directory is on sys.path, so "
        "`import app.<pkg>.<mod>` raises ModuleNotFoundError and the module is "
        "unreachable from application code:\n"
        + "\n".join(f"  {f}:{ln}  imports {m!r} -- use the full app.* path"
                    for f, ln, m in offenders)
    )


def test_every_agents_module_imports_through_the_package():
    """The concrete case, checked by actually importing.

    The static test above catches the syntax; this catches any other reason a
    module in this subsystem cannot be loaded the way the application loads it.
    """
    import importlib

    modules = sorted(p.stem for p in Path("app/agents").glob("*.py")
                     if p.stem != "__init__")
    assert modules, "no modules found under app/agents -- has the layout changed?"

    failures = {}
    for name in modules:
        full = f"app.agents.{name}"
        try:
            importlib.import_module(full)
        except Exception as exc:          # noqa: BLE001 - reporting, not handling
            failures[full] = f"{type(exc).__name__}: {exc}"
    assert not failures, (
        "these modules cannot be imported by their package path:\n"
        + "\n".join(f"  {m} -> {e}" for m, e in sorted(failures.items()))
    )


def test_the_hat_modules_and_the_app_share_one_module_object():
    """Guards against the dual-import hazard the old test setup created.

    With `app/agents` on sys.path, `models` and `app.agents.models` are two
    distinct module objects holding two distinct `AgentManifest` classes, and
    an instance built by one fails `isinstance` against the other. Nothing
    should reintroduce that, so the class the catalogue returns must be the
    class the package exports.
    """
    from app.agents.catalog import get_base
    from app.agents.models import AgentManifest

    base = get_base()
    assert isinstance(base, AgentManifest), (
        f"get_base() returned {type(base)!r} which is not "
        f"{AgentManifest!r} -- the modules have been loaded twice under "
        "different names"
    )
    if "models" in sys.modules and "app.agents.models" in sys.modules:
        assert sys.modules["models"] is sys.modules["app.agents.models"], (
            "both `models` and `app.agents.models` are loaded as separate "
            "modules; something is still inserting app/agents onto sys.path"
        )
