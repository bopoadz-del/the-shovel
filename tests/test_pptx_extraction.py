"""PowerPoint extraction must produce answerable text, not just "not crash".

`_extract_pptx` swallows every exception and returns "" on failure. That is
the right call for an indexer -- one bad deck must not abort a corpus load --
but it means a broken extractor is INDISTINGUISHABLE from an empty deck: both
index zero characters, and the document simply never answers anything. The
function was running at 6% of its body.

So these build a real .pptx and read the text back out, including the slide
numbering the docstring promises "so chunk context stays answerable".
"""
from __future__ import annotations

import os

import pytest

from app.core.doc_index import _extract_pptx


@pytest.fixture
def deck(tmp_path):
    """A real two-slide presentation on disk."""
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    blank = prs.slide_layouts[5]  # title only

    s1 = prs.slides.add_slide(blank)
    s1.shapes.title.text = "Method Statement"
    box = s1.shapes.add_textbox(
        pptx.util.Inches(1), pptx.util.Inches(2),
        pptx.util.Inches(6), pptx.util.Inches(2),
    )
    box.text_frame.text = "Concrete pour sequence for raft RB-01"

    s2 = prs.slides.add_slide(blank)
    s2.shapes.title.text = "Hold Points"
    box2 = s2.shapes.add_textbox(
        pptx.util.Inches(1), pptx.util.Inches(2),
        pptx.util.Inches(6), pptx.util.Inches(2),
    )
    box2.text_frame.text = "Pre-pour inspection witnessed by the Engineer"

    path = str(tmp_path / "method.pptx")
    prs.save(path)
    return path


def test_text_from_every_slide_is_extracted(deck):
    out = _extract_pptx(deck)
    assert "Method Statement" in out
    assert "Concrete pour sequence for raft RB-01" in out
    assert "Hold Points" in out
    assert "Pre-pour inspection witnessed by the Engineer" in out


def test_slides_are_numbered_so_a_citation_can_point_somewhere(deck):
    """The docstring's stated purpose: chunk context stays answerable. Without
    the marker a retrieved chunk cannot say WHERE in the deck it came from."""
    out = _extract_pptx(deck)
    assert "Slide 1:" in out
    assert "Slide 2:" in out
    assert out.index("Slide 1:") < out.index("Slide 2:"), "slide order lost"


def test_a_deck_with_no_text_yields_empty_without_raising(tmp_path):
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])  # fully blank
    path = str(tmp_path / "blank.pptx")
    prs.save(path)
    assert _extract_pptx(path) == ""


def test_a_corrupt_file_returns_empty_instead_of_aborting_the_load(tmp_path):
    """One bad deck must not take down a corpus ingest."""
    path = tmp_path / "not-really.pptx"
    path.write_bytes(b"this is not a presentation")
    assert _extract_pptx(str(path)) == ""


def test_a_missing_file_returns_empty(tmp_path):
    assert _extract_pptx(str(tmp_path / "absent.pptx")) == ""


def test_extraction_does_not_leave_the_file_locked(deck):
    """The reader opens through file_crypto; a leaked handle would make the
    document undeletable on Windows and strand temp files on every ingest."""
    _extract_pptx(deck)
    os.unlink(deck)  # raises PermissionError if a handle is still open
    assert not os.path.exists(deck)
