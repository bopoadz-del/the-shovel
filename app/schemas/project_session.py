"""Session-state schemas — Reasoning Engine Plan 2."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    ts: str = Field(default_factory=_now)


class Artifact(BaseModel):
    name: str
    path: str
    type: str       # 'excel' | 'chart' | 'pdf' | 'json' | ...


class ProjectSession(BaseModel):
    """All state for one project conversation. JSON-serialisable throughout."""
    id: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    # Owner — defaults to "system" so existing serialised sessions (Redis blobs
    # without this field) deserialise without error, and legacy-key callers
    # behave as before.
    user_id: str = "system"
    # free-form computed state: activities, cpm_results, manpower, wbs, ...
    data: Dict[str, Any] = Field(default_factory=dict)
    history: List[Message] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)
    code_cache: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def new(cls, session_id: str, user_id: str = "system") -> "ProjectSession":
        return cls(id=session_id, user_id=user_id)

    def touch(self) -> None:
        self.updated_at = _now()

    def add_message(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))
        self.touch()
