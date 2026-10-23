#!/usr/bin/env python3
"""Fail the build on ``except Exception: pass`` (body is only Pass).

Bare ``except: pass`` is ruff S110 (baseline 0). This twin is the typed
form S110 does not cover: ``except Exception: pass`` and
``except Exception as <name>: pass``.

Walks the same tree as ``scripts/audit_stubs.py`` (skips ``tests/`` and
vendor dirs). Exit 1 with a file:line list if any remain that are not
covered by an ALLOWLIST entry with a named reason. Empty allowlist is the
goal — prefer logging or a narrower ``except`` over growing the list.

Both allowlists are keyed by the handler's ENCLOSING FUNCTION, not its line:
``"relative/path.py::qualified_function"`` (``Class.method``,
``outer.inner``; code outside any def/class is ``path::<module>``). The value
is how many silent sites that function may hold, plus the named reason. An
edit above a baselined handler moves its line, not its function, so it no
longer turns the gate red; a NEW silent site in the same function goes over
the count and still fails.
"""
from __future__ import annotations

import ast
import os
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

# Same vendor / worktree dirs ``audit_stubs.py`` refuses to walk. ``tests/``
# is skipped separately (the same deviation audit_stubs documents): a
# planted fixture in the suite is how this scanner is tested, not a
# production swallow.
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    "generated",
    ".worktrees",
    ".claude",
    "dist",
    "build",
    ".pytest_cache",
}

#: Scope name for a handler that sits outside every def / class.
MODULE_SCOPE = "<module>"


class Allowance(TypedDict):
    """How many silent sites one function may hold, and why."""

    count: int
    reason: str


_BASELINE = "baseline 2026-09-10"


def _baseline(count: int = 1, reason: str = _BASELINE) -> Allowance:
    return {"count": count, "reason": reason}


# Keys are "relative/path.py::qualified_function". Value is how many
# Exception+pass sites that function may keep, and the named reason. Same
# shape as docs/KNOWN_INCOMPLETE.md entries (path + reason, visible, greppable).
# Empty is the goal.
#
# The SSE watchdog (app/routers/chat_watchdog.py) catches Exception so
# CancelledError still cancels, but its body logs and closes the turn —
# it is not a Pass, so it does not belong here.
ALLOWLIST: dict[str, Allowance] = {}


# The RETURN twin's baseline. Keys are "relative/path.py::qualified_function";
# the value is how many empty-return handlers that function may hold and why.
# Every entry has the same reason -- it was here before the scanner was --
# and the counts may only fall. A function that holds more sites than its
# count fails the gate; a count larger than the sites still there is stale
# and fails too (lower it when a handler is fixed).
#
# Converted 2026-10-05 from the old "path:line" keys: each of the 70 line
# entries resolved to its enclosing function on that tree, so the totals per
# function are exactly the line entries that pointed into it. No new room.
#
# The twin exists to stop the next site, not to pretend the remainder are fine.
# Each closes one of two ways: log what failed and keep returning empty (the
# caller genuinely tolerates nothing), or raise a typed outcome (a decision
# path, where "nothing there" and "it broke" must not be the same answer).
#
# See the live per-function counts:  python scripts/scan_exception_pass.py --list-returns
RETURN_ALLOWLIST: dict[str, Allowance] = {
    "app/agents/formulas.py::_get_construction_calculators": _baseline(),
    "app/agents/formulas.py::_get_construction_additional_calculators": _baseline(),
    "app/agents/formulas.py::_get_construction_knowledge_functions": _baseline(),
    "app/agents/runtime.py::_project_has_non_rag_context": _baseline(),
    "app/agents/runtime.py::_file_tool_hint": _baseline(),
    "app/agents/runtime.py::_has_unread_windows": _baseline(2),
    "app/agents/runtime.py::_requested_char_offset": _baseline(2),
    "app/agents/runtime.py::_coerce_tool_arguments": _baseline(),
    "app/agents/runtime.py::_forced_specific_tool": _baseline(),
    "app/agents/runtime.py::_cg_to_number": _baseline(),
    "app/agents/runtime.py::_aca_claim_amount": _baseline(),
    "app/blocks/bim_extractor.py::BIMExtractorBlock._extract_storeys": _baseline(),
    "app/blocks/bim_extractor.py::BIMExtractorBlock._extract_spaces": _baseline(),
    "app/blocks/bim_extractor.py::BIMExtractorBlock._lookup_ifc_element": _baseline(),
    "app/blocks/boq_processor.py::_to_float": _baseline(),
    "app/blocks/boq_processor.py::_to_float_safe": _baseline(),
    "app/blocks/image.py::_placeholder_image_error": _baseline(2),
    "app/blocks/image.py::_yolo_available": _baseline(),
    "app/blocks/ocr.py::OCRBlock._prepare_images": _baseline(2),
    "app/blocks/ocr.py::OCRBlock._extract_pdf_text": _baseline(),
    "app/blocks/primavera_parser.py::_to_float": _baseline(),
    "app/blocks/primavera_parser.py::_orig_dur_days": _baseline(),
    "app/blocks/safety_world_detector.py::_load_yolo": _baseline(),
    "app/blocks/smart_orchestrator.py::SmartOrchestratorBlock._cross_domain_enrichment": _baseline(),
    "app/blocks/smart_orchestrator.py::SmartOrchestratorBlock._predict_learned": _baseline(),
    "app/blocks/validation_pipeline.py::_get_ureg": _baseline(),
    "app/blocks/voice.py::_pydub_available": _baseline(),
    "app/containers/construction/__init__.py::ConstructionContainer._download_file": _baseline(),
    "app/containers/construction/__init__.py::ConstructionContainer._calculate_duration_days": _baseline(),
    "app/containers/construction/__init__.py::ConstructionContainer._calculate_date_diff": _baseline(),
    "app/containers/construction/__init__.py::ConstructionContainer._compare_as_built_to_design": _baseline(),
    "app/containers/construction/__init__.py::ConstructionContainer._add_weeks": _baseline(),
    "app/containers/construction/__init__.py::ConstructionContainer._days_between": _baseline(),
    "app/containers/construction/__init__.py::ConstructionContainer._parse_event_date": _baseline(),
    "app/containers/construction/boq.py::ConstructionBoqMixin._calculate_order_date": _baseline(),
    "app/containers/construction/helpers.py::_parse_money_str": _baseline(),
    "app/core/doc_index.py::_document_size_mb": _baseline(),
    "app/core/doc_index.py::_ocr_pdf_page": _baseline(2),
    "app/core/doc_index.py::_pdf_tables_enabled": _baseline(),
    "app/core/doc_index.py::_pdf_tables_markdown": _baseline(),
    "app/core/doc_index.py::_extract_msg": _baseline(2),
    "app/core/doc_index.py::_extract_doc": _baseline(),
    "app/core/doc_index.py::_ifc_step_census_chunk": _baseline(),
    "app/core/rag/embeddings.py::_has_model2vec": _baseline(),
    "app/core/rag/vector_store.py::VectorStore.bm25_search_photos": _baseline(),
    "app/infra/monitoring.py::_bump_prometheus_request_counter": _baseline(),
    "app/lib/boq_excel.py::evaluate_workbook_total._eval": _baseline(),
    "app/lib/boq_pricing.py::_num": _baseline(),
    "app/lib/pm_computations.py::_xer_hours": _baseline(),
    "app/lib/wbs_duration_overrides.py::_clamp_days": _baseline(),
    "app/main.py::_drain_request_body": _baseline(),
    "app/routers/chat.py::_resolve_attached_documents": _baseline(
        reason=(
            "baseline 2026-09-10; renumbered Agent D look_ahead synthetic "
            "helper, then again by _message_vetoes_predefined + "
            "_warm_predefined_vetoes (SET5 A3)"
        )
    ),
    "app/routers/chat_watchdog.py::event_type": _baseline(),
    "app/routers/mcp.py::mcp_router_available": _baseline(),
    "app/worker/ingest_queue.py::enqueue_ingest": _baseline(),
    "scripts/build_rate_card.py::num": _baseline(),
    "scripts/extract_demolition_boq.py::crop_qty": _baseline(),
    "scripts/extract_boq_rw.py::_num": _baseline(),
    "scripts/extract_boq_section.py::_num": _baseline(),
    "scripts/extract_wetransfer_boqs.py::num": _baseline(),
    "scripts/generate_scenarios_drive_archive.py::parse_qa": _baseline(),
    "scripts/generate_scenarios_drive_archive_v2.py::parse_qa": _baseline(),
    "scripts/price_boq.py::num": _baseline(),
    "scripts/security_scan.py::scan_file": _baseline(),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _catches_exception(handler: ast.ExceptHandler) -> bool:
    """True only for ``except Exception`` (optionally ``as name``).

    Bare ``except:`` is ruff S110. Narrow types (OSError, …) are a
    deliberate cleanup. ``BaseException`` is a different, worse bug and
    is fenced by the SSE-watchdog AST test, not this twin.
    """
    t = handler.type
    if t is None:
        return False
    names = t.elts if isinstance(t, ast.Tuple) else [t]
    for n in names:
        if isinstance(n, ast.Name) and n.id == "Exception":
            return True
        if isinstance(n, ast.Attribute) and n.attr == "Exception":
            return True
    return False


def _body_is_only_pass(handler: ast.ExceptHandler) -> bool:
    body = [
        n
        for n in handler.body
        if not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
    ]
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def _is_exception_pass(handler: ast.ExceptHandler) -> bool:
    return _catches_exception(handler) and _body_is_only_pass(handler)


def _handler_sites(
    tree: ast.AST, matches: Callable[[ast.ExceptHandler], bool]
) -> list[tuple[int, str]]:
    """``(lineno, qualified_function)`` for every handler ``matches`` accepts.

    The qualified name is the chain of enclosing ``def`` / ``class`` names,
    dot-joined (``Class.method``, ``outer.inner``), or ``<module>``.
    """
    out: list[tuple[int, str]] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, scope + (child.name,))
                continue
            if isinstance(child, ast.ExceptHandler) and matches(child):
                out.append((child.lineno, ".".join(scope) or MODULE_SCOPE))
            visit(child, scope)

    visit(tree, ())
    return sorted(out)


def exception_pass_sites(tree: ast.AST) -> list[tuple[int, str]]:
    """``(lineno, qualified_function)`` of every ``except Exception: pass``."""
    return _handler_sites(tree, _is_exception_pass)


def exception_pass_lines(tree: ast.AST) -> list[int]:
    """Line numbers of ``except Exception: pass`` / ``as <name>: pass``."""
    return [lineno for lineno, _ in exception_pass_sites(tree)]


#: The empty values a handler can hand back instead of an outcome.
_EMPTY_CONSTANTS = (None, "", 0, False)


def _returns_only_an_empty_value(handler: ast.ExceptHandler) -> bool:
    """True when the handler's only statement returns None / {} / [] / '' / 0."""
    body = [
        n
        for n in handler.body
        if not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
    ]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    value = body[0].value
    if value is None:  # bare `return`
        return True
    if isinstance(value, ast.Constant) and value.value in _EMPTY_CONSTANTS:
        return True
    if isinstance(value, ast.Dict) and not value.keys:
        return True
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)) and not value.elts:
        return True
    return False


def exception_return_sites(tree: ast.AST) -> list[tuple[int, str]]:
    """``(lineno, qualified_function)`` of every empty-return handler."""
    return _handler_sites(tree, _returns_only_an_empty_value)


def exception_return_lines(tree: ast.AST) -> list[int]:
    """Line numbers of ``except <anything>:`` whose body is only an empty return.

    ANY exception type counts here, unlike the pass-twin which is scoped to
    ``Exception``: swallowing ``OSError`` into ``return None`` is the same
    invisible degradation as swallowing ``Exception``, and the caller cannot
    tell "nothing there" from "the lookup failed" in either case.

    One statement only. A handler that logs and then returns empty has made
    the degradation visible, which is the whole ask -- it is not flagged.
    """
    return [lineno for lineno, _ in exception_return_sites(tree)]


def iter_python_files(root: Path | None = None):
    """Yield (relpath, absolute path) for every scanned ``.py`` file.

    ``relpath`` uses forward slashes so Windows and POSIX report the same
    keys (the same normalisation ``audit_stubs.py`` documents).
    """
    root = Path(root) if root is not None else repo_root()
    root = root.resolve()
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(".py"):
                continue
            abs_path = Path(dirpath) / name
            rel = abs_path.relative_to(root).as_posix()
            if rel.startswith("tests/") or "/tests/" in rel:
                continue
            yield rel, abs_path


def _iter_sites(root: Path | None, sites_of: Callable[[ast.AST], list[tuple[int, str]]]):
    """Yield ``(relpath, lineno, "relpath::qualified_function")`` per site."""
    for rel, abs_path in iter_python_files(root):
        try:
            tree = ast.parse(abs_path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for lineno, qualname in sites_of(tree):
            yield rel, lineno, f"{rel}::{qualname}"


def _allowed(allowlist: dict[str, Allowance], key: str) -> int:
    """Sites ``key`` may hold. An entry with no named reason allows none."""
    entry = allowlist.get(key)
    if not entry or not str(entry.get("reason", "")).strip():
        return 0
    return int(entry.get("count", 0))


def _over_allowance(
    root: Path | None,
    sites_of: Callable[[ast.AST], list[tuple[int, str]]],
    allowlist: dict[str, Allowance],
) -> list[str]:
    """``file:line`` for every site in a function holding more than it may.

    A function with no allowance reports each site as plain ``file:line``.
    A function over its count cannot say WHICH site is the new one, so every
    site in it is listed, tagged with the key and the count it broke.
    """
    by_key: dict[str, list[str]] = {}
    for rel, lineno, key in _iter_sites(root, sites_of):
        by_key.setdefault(key, []).append(f"{rel}:{lineno}")
    findings: list[str] = []
    for key, where in by_key.items():
        allowed = _allowed(allowlist, key)
        if len(where) <= allowed:
            continue
        if allowed == 0:
            findings += where
        else:
            findings += [
                f"{w} ({key}: {len(where)} sites, {allowed} allowed)" for w in where
            ]
    return findings


def stale_entries(allowlist: dict[str, Allowance], live: dict[str, int]) -> list[str]:
    """Entries that allow more sites than their function still holds.

    A stale allowance is room: fix one of two handlers, leave the count at
    2, and the next new site in that function slips in under the old reason.
    """
    out: list[str] = []
    for key, entry in sorted(allowlist.items()):
        have = live.get(key, 0)
        if int(entry.get("count", 0)) > have:
            out.append(f"{key} (allows {entry.get('count')}, tree has {have})")
    return out


def scan(root: Path | None = None) -> list[str]:
    """Return ``file:line`` findings not covered by a named ALLOWLIST entry."""
    return _over_allowance(root, exception_pass_sites, ALLOWLIST)


def scan_returns(root: Path | None = None) -> list[str]:
    """``file:line`` for every empty-return handler beyond RETURN_ALLOWLIST."""
    return _over_allowance(root, exception_return_sites, RETURN_ALLOWLIST)


def all_pass_sites(root: Path | None = None) -> dict[str, int]:
    """Live ``except Exception: pass`` count per ``path::function``."""
    return dict(Counter(key for _, _, key in _iter_sites(root, exception_pass_sites)))


def all_return_sites(root: Path | None = None) -> dict[str, int]:
    """Live empty-return count per ``path::function``. Backs --list-returns."""
    return dict(Counter(key for _, _, key in _iter_sites(root, exception_return_sites)))


def main() -> int:
    if "--list-returns" in sys.argv:
        # Live per-function counts in RETURN_ALLOWLIST shape. A function not
        # already baselined is marked: it is a new site to fix, not to paste.
        for key, count in sorted(all_return_sites().items()):
            entry = RETURN_ALLOWLIST.get(key)
            reason = entry["reason"] if entry else "NEW -- fix it, do not baseline"
            sys.stdout.write(
                '    "%s": {"count": %d, "reason": "%s"},\n' % (key, count, reason)
            )
        return 0

    rc = 0
    findings = scan()
    if findings:
        sys.stdout.write(
            "SILENT except Exception: pass (log it or name a reason in "
            "ALLOWLIST):\n"
        )
        for item in findings:
            sys.stdout.write(f"  {item}\n")
        sys.stdout.write(f"TOTAL: {len(findings)}\n")
        rc = 1
    else:
        sys.stdout.write("NO silent except Exception: pass handlers.\n")

    returns = scan_returns()
    if returns:
        sys.stdout.write(
            "SILENT except -> empty return. Log what failed, or raise a typed "
            "outcome; do not add to RETURN_ALLOWLIST:\n"
        )
        for item in returns:
            sys.stdout.write(f"  {item}\n")
        sys.stdout.write(f"TOTAL: {len(returns)}\n")
        rc = 1
    else:
        total = sum(e["count"] for e in RETURN_ALLOWLIST.values())
        sys.stdout.write(
            f"RETURN: 0 new ({total} baselined in {len(RETURN_ALLOWLIST)} "
            "functions).\n"
        )

    stale = stale_entries(ALLOWLIST, all_pass_sites()) + stale_entries(
        RETURN_ALLOWLIST, all_return_sites()
    )
    if stale:
        sys.stdout.write(
            "STALE allowlist entries (a handler was fixed or its function "
            "renamed -- lower the count or delete the entry):\n"
        )
        for item in stale:
            sys.stdout.write(f"  {item}\n")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
