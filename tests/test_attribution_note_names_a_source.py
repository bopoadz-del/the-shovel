"""The "Unverified attribution removed" note is shown only when the removed
attribution named a source a reader could look up. Live: the model's own
"Source: the platform's calculator, on your inputs" line, a bare "Sources:"
heading and "this is general knowledge" drew the note. Scripted model."""
from __future__ import annotations

import pytest

from tests.lane5_core_support import answer_text, drive, install_driver_stream

NOTE = "Unverified attribution removed"


@pytest.mark.parametrize("answer", [
    "25% is done.\n\nSource: platform progress formula, run on your figures.",
    "25% is done.\n\n**Sources:**",
    "25% is done.\n\nSource: this is general knowledge, not a project document.",
])
def test_a_removed_line_that_names_no_source_draws_no_note(monkeypatch, answer):
    install_driver_stream(monkeypatch)
    events, _ = drive(monkeypatch, "2,000 m total, 500 m laid: how far along?", [{"content": answer}])
    assert NOTE not in answer_text(events)


def test_a_removed_line_that_names_a_source_still_draws_the_note(monkeypatch):
    install_driver_stream(monkeypatch)
    events, _ = drive(monkeypatch, "What governs the minimum slab thickness?",
                      [{"content": "The minimum thickness follows the code table.\n\n"
                                   "Source: ACI 318-19 Table 7.3.1.1."}])
    assert NOTE in answer_text(events)
