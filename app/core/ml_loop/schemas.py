"""Payload schemas for every ML loop operation.

One pydantic model per operation. The block validates the payload against
the model before it touches the store, so a malformed call fails with the
field that is wrong instead of writing a partial row. Every write payload
carries the same keys (:class:`RowKeys`): conversation, project, message
and build sha.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RowKeys(_Payload):
    """Keys every stored row carries."""

    project_id: str = Field(..., min_length=1, max_length=200)
    conversation_id: str | None = Field(None, max_length=200)
    message_id: str | None = Field(None, max_length=200)
    #: Overrides the settings' build sha (e.g. a replayed import).
    build_sha: str | None = Field(None, max_length=64)


class RecordTurn(RowKeys):
    """One answered turn (what ``dataset_log`` writes today)."""

    question: str = Field(..., min_length=1)
    answer: str = ""
    verdict: str = Field(..., min_length=1, max_length=64)
    route: str | None = Field(None, max_length=120)
    evidence_plan: list[str] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    cited_ids: list[str] = Field(default_factory=list)


class FeedbackKind(str, Enum):
    correct = "correct"
    wrong_figure = "wrong_figure"
    wrong_document = "wrong_document"
    missing = "missing"


class RecordFeedback(RowKeys):
    """A user's verdict on one answer (ML-2)."""

    kind: FeedbackKind
    correct_value: float | None = None
    unit: str | None = Field(None, max_length=32)
    note: str | None = Field(None, max_length=200)
    cited_ids: list[str] = Field(default_factory=list)
    user_id: str | None = Field(None, max_length=200)


class RecordCorrection(RowKeys):
    """A predicted-vs-actual numeric correction (learning_engine today)."""

    formula_id: str = Field(..., min_length=1, max_length=200)
    predicted: float
    actual: float
    source: str = Field("manual", max_length=64)


class RecordRouteCorrection(RowKeys):
    """A routing correction (``/v1/feedback/route`` today)."""

    message: str = Field(..., min_length=1)
    correct_action: str = Field(..., min_length=1, max_length=120)
    original_action: str | None = Field(None, max_length=120)
    source: str = Field("user_correction", max_length=64)


class Propose(_Payload):
    """Read-only proposal pass over a window of stored rows (ML-3)."""

    project_id: str | None = Field(None, max_length=200)
    window_days: int | None = Field(None, ge=1, le=366)


class ShadowPredict(RowKeys):
    """Log what a candidate predictor would have decided, next to the live
    decision. Never acted on (ML-4)."""

    predictor: str = Field(..., min_length=1, max_length=120)
    live_label: str = Field(..., min_length=1, max_length=120)
    shadow_label: str = Field(..., min_length=1, max_length=120)
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class ShadowReport(_Payload):
    predictor: str | None = Field(None, max_length=120)
    project_id: str | None = Field(None, max_length=200)
    window_days: int | None = Field(None, ge=1, le=366)


class Status(_Payload):
    project_id: str | None = Field(None, max_length=200)


class ExportTables(_Payload):
    project_id: str | None = Field(None, max_length=200)


class ImportTables(_Payload):
    tables: dict[str, list[dict[str, Any]]]


#: operation name -> payload model. The block's manifest is built from this.
OPERATION_SCHEMAS: dict[str, type[_Payload]] = {
    "record_turn": RecordTurn,
    "record_feedback": RecordFeedback,
    "record_correction": RecordCorrection,
    "record_route_correction": RecordRouteCorrection,
    "propose": Propose,
    "shadow_predict": ShadowPredict,
    "shadow_report": ShadowReport,
    "status": Status,
    "export_tables": ExportTables,
    "import_tables": ImportTables,
}

#: Operations that write rows (refused when the store is not configured).
WRITE_OPERATIONS = frozenset({
    "record_turn",
    "record_feedback",
    "record_correction",
    "record_route_correction",
    "shadow_predict",
    "import_tables",
})
