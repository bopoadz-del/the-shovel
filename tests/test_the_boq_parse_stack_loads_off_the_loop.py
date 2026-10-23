"""The BOQ parse stack is imported, decrypted and parsed off the event loop.

LANE5 #984/#992: a BOQ export's first request imported the block, then
doc_index, file_crypto, pandas and pdfplumber on the event loop (a first
import of a C extension holds the GIL) and froze every user ~0.4-0.6 s. The
imports stay lazy -- nothing heavy at boot, the boot check is unchanged --
but every one of them, and the decrypt and parse, now runs in a worker
thread. Each test records the thread that executes each import statement
(builtins.__import__ and importlib.import_module run even when the
module is already loaded), so it
holds in either CI boot mode, cold or warm.
"""
from __future__ import annotations

import ast
import asyncio
import builtins
import importlib
import os
import pathlib
import tempfile
import threading

import pytest

_WATCHED = ("pandas", "pdfplumber", "app.core.doc_index", "app.core.file_crypto", "app.blocks.boq_processor")


@pytest.fixture
def import_threads(monkeypatch):
    """{module: [thread ident, ...]} for every import statement of a watched
    module while the test runs."""
    seen: dict = {}
    real = builtins.__import__

    def recording(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in _WATCHED:
            seen.setdefault(name, []).append(threading.get_ident())
        return real(name, globals, locals, fromlist, level)

    real_module = importlib.import_module

    def recording_module(name, package=None):
        if name in _WATCHED:
            seen.setdefault(name, []).append(threading.get_ident())
        return real_module(name, package)

    monkeypatch.setattr(builtins, "__import__", recording)
    monkeypatch.setattr(importlib, "import_module", recording_module)
    return seen


def _drive(factory):
    loop_thread: list = []

    async def main():
        loop_thread.append(threading.get_ident())
        return await factory()

    out = asyncio.run(main())
    return loop_thread[0], out


@pytest.fixture
def boq_files():
    pdfium = pytest.importorskip("pypdfium2")
    made = []
    fd, csv_path = tempfile.mkstemp(suffix="_BOQ.csv")
    os.write(fd, b"Item,Description,Unit,Quantity,Rate,Amount\n1,Concrete,m3,10,100,1000\n")
    os.close(fd)
    made.append(csv_path)
    doc = pdfium.PdfDocument.new()
    doc.new_page(595, 842)
    fd, pdf_path = tempfile.mkstemp(suffix="_BOQ.pdf")
    os.close(fd)
    doc.save(pdf_path)
    made.append(pdf_path)
    yield {"csv": csv_path, "pdf": pdf_path}
    for p in made:
        try:
            os.remove(p)
        except OSError:
            pass


@pytest.mark.parametrize("kind,libs", [("csv", ("pandas",)), ("pdf", ("pdfplumber", "app.core.doc_index"))])
def test_a_boq_parse_imports_decrypts_and_parses_off_the_loop(import_threads, boq_files, monkeypatch, kind, libs):
    pytest.importorskip("pandas")
    pytest.importorskip("pdfplumber")
    from app.blocks import boq_processor as bp
    from app.core import file_crypto

    opened: list = []
    real_open = file_crypto.open_plaintext

    def probe(path):
        opened.append(threading.get_ident())
        return real_open(path)

    monkeypatch.setattr(file_crypto, "open_plaintext", probe)
    import_threads.clear()
    loop_ident, _ = _drive(lambda: bp.BOQProcessorBlock().process({"file_path": boq_files[kind]}, {}))
    assert opened and loop_ident not in opened, "the plaintext read ran on the event loop"
    from app.core import extract_isolated

    # A PDF's table parse runs in the isolated child where fork exists: its
    # libraries are then imported in the child, never in this process.
    in_child = kind == "pdf" and extract_isolated.isolation_available()
    for lib in (*libs, "app.core.file_crypto"):
        if in_child and lib not in ("app.core.file_crypto", "app.core.doc_index"):
            assert loop_ident not in import_threads.get(lib, []), f"{lib} was imported on the event loop"
            continue
        assert import_threads.get(lib), f"{lib} was never imported by the parse -- the probe saw nothing"
        assert loop_ident not in import_threads[lib], f"{lib} was imported on the event loop"


def test_the_export_handlers_load_the_block_off_the_loop(import_threads):
    from app.routers import exports

    import_threads.clear()
    loop_ident, cls = _drive(exports._boq_processor_block)
    assert cls.__name__ == "BOQProcessorBlock"
    assert import_threads.get("app.blocks.boq_processor"), "the block was never imported"
    assert loop_ident not in import_threads["app.blocks.boq_processor"]


def test_no_async_export_handler_imports_the_block_itself():
    """Both BOQ export handlers go through the off-loop loader: an inline
    import inside any async handler would put the first load back on the loop."""
    source = pathlib.Path(__file__).resolve().parents[1] / "app" / "routers" / "exports.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    inline = [
        node.name
        for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
        for sub in ast.walk(node)
        if isinstance(sub, ast.ImportFrom) and sub.module == "app.blocks.boq_processor"
    ]
    assert not inline, f"async handlers importing the BOQ block inline: {inline}"
    users = [n.name for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
             and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_boq_processor_block"
                     for c in ast.walk(n))]
    assert len(users) >= 2, users


def test_the_parse_libraries_are_warmed_after_the_port_opens(monkeypatch):
    """Live users never pay the parse stack's first import: the post-start
    warm-up imports it (the boot check keeps it out of startup itself)."""
    import importlib as _importlib

    from app import main

    names = []
    monkeypatch.setattr(_importlib, "import_module", lambda name, package=None: names.append(name))
    main._warm_parse_libraries()
    assert {"pandas", "pdfplumber"} <= set(names)
    source = pathlib.Path(main.__file__).read_text(encoding="utf-8")
    warm = source[source.index("async def _warm_models"):source.index("async def _warm_models") + 3000]
    assert "_warm_parse_libraries" in warm
