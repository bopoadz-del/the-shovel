"""Every declared formula is defined in its owner's package, and
scripts/move_formulas.py splits a synthetic module by declared owner."""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

from app.lib import formula_registry

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("move_formulas", ROOT / "scripts" / "move_formulas.py")
mover = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mover
spec.loader.exec_module(mover)


def _package_of(owner: str) -> str:
    return "app.agents.base.formulas" if owner == "base" else f"app.agents.hats.{owner}.formulas"


def test_every_formula_is_defined_in_its_owner_package():
    specs = formula_registry.all_specs()
    assert specs
    misplaced = [(s.name, s.owner, s.fn.__module__) for s in specs
                 if not s.fn.__module__.startswith(_package_of(s.owner) + ".")]
    assert misplaced == []


def test_formula_modules_are_discovered_from_the_packages():
    mods = formula_registry.formula_modules()
    assert mods and all(m.startswith(("app.agents.base.formulas.", "app.agents.hats.")) for m in mods)


def test_old_import_paths_still_answer():
    from app.lib import construction_formulas as cf

    assert set(cf.CALCULATORS) == {s.name for s in formula_registry.all_specs()}


SOURCE = '''"""Synthetic formulas."""
from __future__ import annotations

import math

from app.lib.formula_registry import formula

_FACTOR = 2.0
_SHARED_PI = math.pi


def _area(r):
    return _SHARED_PI * r * r


@formula(owner="design", description="d", inputs={"r": "m"}, outputs={"a": "m2"})
def circle_area(r):
    return {"a": _area(r) * _FACTOR}


@formula(owner="qaqc", description="q", inputs={"r": "m"}, outputs={"c": "m"})
def circumference(r):
    return {"c": 2 * _SHARED_PI * r}


def run_engine(name):
    return {"circle_area": circle_area, "circumference": circumference}[name]
'''


def test_mover_splits_by_declared_owner():
    sp = mover.split(SOURCE, "synthetic.formulas_mod")
    assert sp.problems == []
    assert sp.owners_of == {"_FACTOR": "design", "_SHARED_PI": mover.SHARED, "_area": "design",
                            "circle_area": "design", "circumference": "qaqc", "run_engine": ""}
    new, rest = mover.render(SOURCE, sp)
    assert set(new) == {"app.agents.hats.design.formulas.formulas_mod",
                        "app.agents.hats.qaqc.formulas.formulas_mod",
                        "app.agents.base.formulas.formulas_mod_shared"}
    design = new["app.agents.hats.design.formulas.formulas_mod"]
    assert "def circle_area" in design and "def _area" in design and "_FACTOR = 2.0" in design
    assert "from app.agents.base.formulas.formulas_mod_shared import" in design
    assert "import math" not in design  # only the imports its own code uses
    shared = new["app.agents.base.formulas.formulas_mod_shared"]
    assert "import math" in shared and "_SHARED_PI = math.pi" in shared
    # What no formula uses stays, and imports the moved names back first.
    tree = ast.parse(rest)
    assert [n.name for n in tree.body if isinstance(n, ast.FunctionDef)] == ["run_engine"]
    assert rest.index("import (") < rest.index("def run_engine")
    for text in list(new.values()) + [rest]:
        ast.parse(text)


def test_a_helper_a_formula_uses_moves_with_it_and_the_engine_imports_it_back():
    src = SOURCE + """

def describe(name):
    return f"{name}: area of radius 1 is {_area(1):.2f}"
"""
    sp = mover.split(src, "synthetic.formulas_mod")
    assert sp.problems == []
    assert sp.owners_of["_area"] == "design" and sp.owners_of["describe"] == ""
    _new, rest = mover.render(src, sp)
    assert "    _area,\n" in rest  # imported back for the engine code that stays
    assert "def describe" in rest
