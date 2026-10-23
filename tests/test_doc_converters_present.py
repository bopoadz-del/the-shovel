"""Fences for legacy .doc extraction.

``app.core.doc_index._extract_doc`` shells out to ``antiword`` then ``catdoc``.
Its remaining fallbacks cannot apply in the Linux runtime image -- ``textract``
is not a declared dependency and ``win32com`` is Windows-only -- so if neither
binary is installed the converter list comes out EMPTY and every .doc extracts
to "" and indexes as ZERO_CHUNK.

That is exactly what production did until 2026-08-20: the runtime apt layer
installed tesseract/ffmpeg/node but no .doc converter, so 100% of .doc uploads
produced no chunks, silently -- an empty return is indistinguishable from an
empty document. These tests fence both halves: the image must ship a converter,
and a host without one must say so instead of failing quietly.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The converters _extract_doc actually invokes, in its own order of preference.
DOC_CONVERTERS = ("antiword", "catdoc")


def _runtime_apt_packages() -> str:
    """Everything the Dockerfile's RUNTIME stage apt-installs, as one blob."""
    return (REPO / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_installs_a_doc_converter():
    dockerfile = _runtime_apt_packages()
    found = [c for c in DOC_CONVERTERS if re.search(rf"^\s*{c}\s*\\?\s*$",
                                                    dockerfile, re.MULTILINE)]
    assert found, (
        "Dockerfile installs no .doc converter. _extract_doc() shells out to "
        f"{' or '.join(DOC_CONVERTERS)}; without one, shutil.which() returns "
        "None for both, textract/win32com cannot apply on Linux, and EVERY "
        ".doc silently extracts to empty text (ZERO_CHUNK)."
    )


def test_aptfile_lists_a_doc_converter():
    aptfile = (REPO / "Aptfile").read_text(encoding="utf-8").split()
    found = [c for c in DOC_CONVERTERS if c in aptfile]
    assert found, (
        f"Aptfile must list one of {DOC_CONVERTERS} so buildpack-style "
        "deploys get a .doc converter too, not just the Dockerfile path."
    )


def test_extract_doc_warns_when_no_converter_available(monkeypatch, tmp_path, caplog):
    """A host with no converter must announce it, not return "" in silence.

    The silence is what let this hide in production: the corpus filled with
    contentless .doc rows and every gate reported success.
    """
    import shutil

    from app.core import doc_index

    monkeypatch.setattr(shutil, "which", lambda _name: None)

    sample = tmp_path / "legacy.doc"
    sample.write_bytes(b"\xd0\xcf\x11\xe0dummy-ole-header")

    with caplog.at_level(logging.WARNING, logger=doc_index.logger.name):
        doc_index._extract_doc(str(sample))

    # The RETURN VALUE is deliberately not asserted: it is host-dependent.
    # On the Linux runtime image both remaining fallbacks are unavailable and
    # the function returns "". On a Windows dev box with Word + pywin32 the
    # COM fallback opens the file and hands back its raw bytes decoded as text
    # (observed: 'ﾐﾏ\x11濺ummy\r' for a stub OLE header). Pinning "" here
    # would pass in CI and fail on a developer's machine for reasons that have
    # nothing to do with the defect. The warning is the portable contract.
    assert any(
        "no .doc converter" in rec.getMessage() for rec in caplog.records
    ), (
        "no warning emitted when antiword/catdoc are both absent — the "
        "failure stays invisible, which is the bug this fences"
    )
