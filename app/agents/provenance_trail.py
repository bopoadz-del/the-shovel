"""Provenance trail of the answer being written in this task.

Every assistant answer carries a record of where each of its figures and
facts came from -- user input, project document (doc and page), general
knowledge (doc and page), or calculator (formula and inputs). The record is
built by ``app.agents.citation_provenance.figure_provenance`` from the turn's
evidence objects, streamed on the end event, and stored with the message.

This module only holds the source-kind vocabulary and the per-task slot the
message store reads; it decides nothing.
"""
from __future__ import annotations

import contextvars
from typing import Optional

SOURCE_USER = "user_input"
SOURCE_PROJECT = "project_document"
SOURCE_GENERAL = "general_knowledge"
SOURCE_CALCULATOR = "calculator"
#: A figure the user told in an earlier chat, read back this turn from a
#: project note that the note itself marks as told by the user. Never a
#: verified source: shown with what it is, as a document figure is shown
#: with its document.
SOURCE_USER_NOTE = "user_note"
USER_NOTE_LABEL = "Told by the user, from the project notes"

#: Provenance of the answer just finished in this task (read by the message
#: store so the record is saved with the assistant message).
LAST_PROVENANCE: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "provenance_trail_last", default=None,
)
