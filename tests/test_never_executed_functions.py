"""Cover the functions the whole suite never executed.

A coverage sweep on 2026-08-15 (3500 tests, 76% line coverage) found exactly
15 functions in app/ whose bodies had never run once. Line coverage hides
these: a module can sit at 80% while a specific function has never executed,
because coverage reports LINES, not functions.

This file closes the ones reachable without external services. The remainder
are nested closures behind a live Google Drive, an MCP server, or a binary
.doc, and are listed at the bottom rather than quietly ignored -- an untested
function nobody has written down is indistinguishable from one nobody noticed.
"""
from __future__ import annotations

import pytest

from app.core import panels
from app.core.learning import model_registry
from tests._fixture_values import ENV_SECRET_09

# ── app/core/panels.py ────────────────────────────────────────────────────

def test_make_error_panel_is_renderable_and_flagged():
    """The UI renders a red banner off these fields; all must be present."""
    p = panels.make_error_panel("quantities", "Quantities", "extractor blew up")
    assert p["type"] == "quantities"
    assert p["title"] == "Quantities"
    assert p["error"] is True
    assert p["error_message"] == "extractor blew up"
    # data must exist and be empty -- a renderer that iterates it must not
    # trip over a missing key while showing the error.
    assert p["data"] == {}


def test_validate_panel_passes_a_well_formed_panel_through():
    """A valid panel must come back UNCHANGED, not normalised into something else."""
    panel = {
        "type": "document_info",
        "title": "Document",
        "data": {"filename": "spec.pdf", "pages": 12},
    }
    out = panels.validate_panel(dict(panel))
    assert out["type"] == "document_info"
    assert out.get("error") is not True, out


def test_validate_panel_never_raises_on_a_broken_panel():
    """The documented contract: a backend regression must not break the UI.

    validate_panel returns a typed error panel instead of raising, so the
    frontend shows an obvious banner rather than losing the whole response.
    """
    broken = {"type": "quantities", "data": {"items": "not-a-list"}}
    out = panels.validate_panel(broken)  # must not raise
    assert isinstance(out, dict)
    assert out.get("type") == "quantities", "the error panel must keep the type"


def test_validate_panel_tolerates_an_unknown_panel_type():
    out = panels.validate_panel({"type": "not_a_real_panel", "data": {}})
    assert isinstance(out, dict)


# ── app/core/learning/model_registry.py ───────────────────────────────────

def test_resolve_base_model_returns_the_concrete_id():
    assert model_registry.resolve_base_model("construction_v1") == (
        "Qwen/Qwen2.5-3B-Instruct"
    )


def test_unknown_alias_raises_instead_of_silently_falling_back():
    """The module's own rationale: a silent fallback would hide a real
    misconfiguration -- you would fine-tune against the wrong base model and
    never know."""
    with pytest.raises(ValueError) as exc:
        model_registry.resolve_base_model("does-not-exist")
    # The message must name the valid choices, or the operator cannot act on it.
    assert "construction_v1" in str(exc.value)


def test_list_aliases_returns_a_COPY_not_the_live_registry():
    """Mutating the dashboard's view must not repoint the real registry."""
    first = model_registry.list_aliases()
    first["construction_v1"] = "tampered/model"
    assert model_registry.resolve_base_model("construction_v1") != "tampered/model"
    assert model_registry.list_aliases()["construction_v1"] == (
        "Qwen/Qwen2.5-3B-Instruct"
    )


# ── app/blocks/sympy_reasoning.py — the no-sympy fallback ─────────────────
# `_eval_symbolic` has two branches. Every existing test exercised the sympy
# branch, so the pure-Python fallback -- the one that runs on a deployment
# WITHOUT sympy installed -- had never executed. Its `_lookup` helper is the
# only thing standing between that deployment and a NameError.

def _no_sympy(monkeypatch):
    import app.blocks.sympy_reasoning as sr
    monkeypatch.setattr(sr, "_symbolic", lambda: {"available": False})
    return sr


def test_fallback_reads_values_by_plain_string_key(monkeypatch):
    sr = _no_sympy(monkeypatch)
    assert sr._eval_symbolic("variance_pct", {"actual": 110, "avg": 100}) == 10.0
    assert sr._eval_symbolic("z_score",
                             {"actual": 112, "avg": 100, "std_dev": 6}) == 2.0
    assert sr._eval_symbolic("cost_impact",
                             {"actual": 110, "avg": 100, "quantity": 5}) == 50


def test_fallback_also_accepts_symbol_like_keys(monkeypatch):
    """_lookup matches on a `.name` attribute, so sympy-shaped keys still
    resolve after sympy has gone away mid-flight."""
    sr = _no_sympy(monkeypatch)

    class _FakeSymbol:
        def __init__(self, name):
            self.name = name

    subs = {_FakeSymbol("actual"): 120, _FakeSymbol("avg"): 100}
    assert sr._eval_symbolic("variance_pct", subs) == 20.0


def test_fallback_defaults_missing_inputs_instead_of_raising(monkeypatch):
    """A missing key must fall back to the default, not KeyError -- and the
    zero-denominator guards must hold rather than raise ZeroDivisionError."""
    sr = _no_sympy(monkeypatch)
    assert sr._eval_symbolic("variance_pct", {"actual": 50}) == 0.0   # avg -> 0
    assert sr._eval_symbolic("z_score", {"actual": 50, "avg": 40}) == 0.0  # std -> 0
    # quantity defaults to 1, so cost_impact is the bare difference
    assert sr._eval_symbolic("cost_impact", {"actual": 110, "avg": 100}) == 10


def test_fallback_rejects_an_unknown_formula(monkeypatch):
    sr = _no_sympy(monkeypatch)
    with pytest.raises(RuntimeError):
        sr._eval_symbolic("not_a_formula", {"actual": 1, "avg": 1})


# ── app/blocks/drawing_qto.py — the old-style POLYLINE branch ─────────────
# LWPOLYLINE is what modern CAD emits and what every fixture used, so the
# legacy POLYLINE branch -- and its `_pt_xyz` vertex reader -- had never run.
# Old drawings in a client archive are exactly where legacy entities live.

def test_legacy_polyline_entities_are_measured():
    ezdxf = pytest.importorskip("ezdxf")
    from app.blocks.drawing_qto import DrawingQTOBlock

    doc = ezdxf.new()
    msp = doc.modelspace()
    # 3-4-5 triangle legs: total run is 3 + 4 = 7 drawing units.
    msp.add_polyline3d([(0, 0, 0), (3, 0, 0), (3, 4, 0)])

    results, _bulges, _meta = DrawingQTOBlock()._extract_measurements(msp, 1.0)
    polylines = [r for r in results if str(r.get("type")).startswith("polyline")]
    assert polylines, f"legacy POLYLINE was not measured at all: {results}"
    p = polylines[0]
    assert p["type"] == "polyline_3d", p
    assert p["is_3d"] is True
    assert p["vertex_count"] == 3
    # 3 along x then 4 along y: the vertex reader must produce real coords,
    # not zeros -- a broken _pt_xyz would silently measure 0.
    assert p["length_m"] == pytest.approx(7.0, abs=1e-6), p


def test_legacy_polyline_length_scales_with_units():
    """unit_factor converts drawing units to metres; a mm drawing must not
    report millimetre magnitudes as metres."""
    ezdxf = pytest.importorskip("ezdxf")
    from app.blocks.drawing_qto import DrawingQTOBlock

    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_polyline3d([(0, 0, 0), (1000, 0, 0)])

    results, _b, _m = DrawingQTOBlock()._extract_measurements(msp, 0.001)
    p = next(r for r in results if str(r.get("type")).startswith("polyline"))
    assert p["length_m"] == pytest.approx(1.0, abs=1e-9), p


def test_legacy_2d_polyline_takes_the_non_3d_path():
    """The same helper, reached through the is_3d=False branch."""
    ezdxf = pytest.importorskip("ezdxf")
    from app.blocks.drawing_qto import DrawingQTOBlock

    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_polyline2d([(0, 0), (6, 0), (6, 8)])

    results, _b, _m = DrawingQTOBlock()._extract_measurements(msp, 1.0)
    p = next(r for r in results if str(r.get("type")).startswith("polyline"))
    assert p.get("is_3d") is not True, p
    assert p["length_m"] == pytest.approx(14.0, abs=1e-6), p


# ── app/blocks/sandbox.py — the restricted-shell execution path ───────────
# Every existing test stopped at the allowlist REJECTION branch, so the code
# that actually spawns a process -- `_spawn_and_wait` -- had never run. That
# is also where the secret-scrubbing lives, so the security property it
# carries was equally unexercised.

@pytest.mark.asyncio
async def test_allowlisted_command_actually_executes():
    from app.blocks.sandbox import SandboxBlock, SandboxPolicy
    block = SandboxBlock()
    out = await block._execute_bash("echo sandbox-ran", SandboxPolicy())
    assert not out.get("blocked"), out
    assert "sandbox-ran" in (out.get("stdout") or out.get("output") or ""), out


@pytest.mark.asyncio
async def test_non_allowlisted_command_is_blocked_without_spawning():
    from app.blocks.sandbox import SandboxBlock, SandboxPolicy
    block = SandboxBlock()
    out = await block._execute_bash("curl https://example.com", SandboxPolicy())
    assert out.get("blocked") is True, out
    assert "allowlist" in (out.get("error") or "").lower(), out


@pytest.mark.asyncio
async def test_the_sandbox_shell_cannot_echo_secrets_out(monkeypatch):
    """audit 6.1: the subprocess env is scrubbed, so a shell command must not
    be able to read SECRET_KEY / DATABASE_URL back out of the sandbox."""
    monkeypatch.setenv("SECRET_KEY", ENV_SECRET_09)
    monkeypatch.setenv("DATABASE_URL", "postgres://leak:leak@host/db")
    from app.blocks.sandbox import SandboxBlock, SandboxPolicy
    block = SandboxBlock()
    # A literal marker rides along with the secret expansion. Without it this
    # test could not tell "ran and scrubbed" from "never ran at all" -- both
    # produce empty output, and the empty one would pass for the wrong reason.
    out = await block._execute_bash(
        "echo MARKER-OK $SECRET_KEY $DATABASE_URL", SandboxPolicy())
    blob = f"{out.get('stdout', '')}{out.get('output', '')}{out.get('stderr', '')}"
    assert "MARKER-OK" in blob, f"the command never actually ran: {out}"
    assert ENV_SECRET_09 not in blob, out
    assert "postgres://leak" not in blob, out


# ── app/containers/construction/documents.py — the quantities panel ───────
# `_qty_val` normalises a quantity that may arrive as a bare number OR as a
# {"quantity": n} dict. It guards whether the quantities panel is shown at
# all, and it had never executed: the auto_pipeline tests all ran documents
# that produced no quantities.

@pytest.mark.asyncio
async def test_quantities_panel_appears_only_when_a_value_is_non_zero(monkeypatch):
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()

    async def _fake_analyse(text, hint=None):
        return {
            "status": "success",
            "doc_type": "boq",
            # deliberately MIXED shapes: a bare number and a {"quantity": n}
            # dict, which is exactly what _qty_val exists to reconcile.
            "extracted_quantities": {"concrete_m3": 945, "rebar_t": {"quantity": 12.5}},
        }

    monkeypatch.setattr(container, "_analyse_text_only", _fake_analyse, raising=False)
    out = await container.auto_pipeline({"extracted_text": "BOQ text"}, {})
    panels_out = out.get("panels") or []
    kinds = [p.get("type") for p in panels_out]
    assert "quantities" in kinds, f"quantities panel missing: {kinds}"


@pytest.mark.asyncio
async def test_all_zero_quantities_do_not_raise_a_quantities_panel(monkeypatch):
    """The other direction: an all-zero extraction is not a result worth
    showing, and must not be rendered as though quantities were found."""
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()

    async def _fake_analyse(text, hint=None):
        return {
            "status": "success",
            "doc_type": "boq",
            "extracted_quantities": {"concrete_m3": 0, "rebar_t": {"quantity": 0}},
        }

    monkeypatch.setattr(container, "_analyse_text_only", _fake_analyse, raising=False)
    out = await container.auto_pipeline({"extracted_text": "BOQ text"}, {})
    kinds = [p.get("type") for p in (out.get("panels") or [])]
    assert "quantities" not in kinds, f"a zero-value panel was rendered: {kinds}"


# ── Register: the functions still never executed, and WHY ─────────────────
#
# These are the remainder of the 15. Each needs a dependency the suite cannot
# stand up honestly, so they are written down rather than silently left out.
# If any of these becomes reachable, delete its line and add the test.
#
#   app/core/doc_index.py:_read_stream
#       Nested in the legacy .doc reader. Needs a real OLE2 compound binary;
#       olefile can read but not synthesise one, and shipping a binary .doc
#       fixture is what the repo already treats as a liability.
#
#   app/routers/admin.py:_list, _walk
#       Nested in the Drive folder scanner. Need a live Google Drive session
#       with a real access token; Drive is currently unconnected (409).
#
#   app/routers/mcp.py:_list_tools, _call_tool
#       Nested inside _build_server(), which only runs inside a live MCP SSE
#       session driven by the mcp server library's transport.
#
#   app/routers/mcp.py:mount_message_endpoint, mcp_sse_unavailable
#       Defined ONLY in the else-branch taken when mcp[server] is absent. The
#       package IS installed here, so this branch is unreachable in this
#       environment by construction -- testing it means uninstalling a
#       dependency, which would be a different (and dishonest) build.
#
# Counted honestly: of the 15 functions that had never executed, this file
# covers 8. The other 7 are environment-gated, not forgotten.
