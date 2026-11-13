"""ML loop block: the learning loop as one Store-shaped block.

One interface: ``process({"operation": <name>, "payload": {...}})``. The
operations and their payload schemas are in :data:`MANIFEST`; the code
behind them lives in ``app.core.ml_loop``. Hosts (the answer path, the
routers, the learning-engine adapter) reach the loop only through these
operations.

The block names no product, project or document. It is configured
through one :class:`MLLoopSettings` (build sha, on/off, ONE database
session factory); with no session factory every write reports
``not_configured`` and nothing is stored.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from pydantic import ValidationError

from app.core.ml_loop import proposals as _proposals
from app.core.ml_loop import shadow as _shadow
from app.core.ml_loop.schemas import OPERATION_SCHEMAS, WRITE_OPERATIONS
from app.core.ml_loop.settings import MLLoopSettings
from app.core.ml_loop.store import TABLE_PREFIX, TABLES, MLLoopRepository
from app.core.universal_base import UniversalBlock

logger = logging.getLogger("ml_loop")

LOG_PREFIX = "ml_loop_"
ROUTER_PREFIX = "/v1/ml"

MANIFEST: dict[str, Any] = {
    "name": "ml_loop",
    "version": "0.1.0",
    "entry_point": "process",
    "operations": {
        name: model.model_json_schema() for name, model in OPERATION_SCHEMAS.items()
    },
    "write_operations": sorted(WRITE_OPERATIONS),
    "tables": sorted(TABLES),
    "table_prefix": TABLE_PREFIX,
    "router_prefix": ROUTER_PREFIX,
    "log_prefix": LOG_PREFIX,
}

#: Free-text fields capped at ``settings.max_text_chars`` before storage.
_CAPPED_TEXT = ("question", "answer", "message")


class MLLoopBlock(UniversalBlock):
    name = "ml_loop"
    version = MANIFEST["version"]
    description = (
        "Learning loop: records turns, answer feedback and corrections; "
        "read-only proposals; shadow comparison of candidate predictors"
    )
    layer = 3
    tags = ["ml", "learning", "feedback", "store"]
    requires: list = []
    auto_validate = False
    allow_empty_input = True
    manifest = MANIFEST

    _configured: MLLoopSettings | None = None
    _configure_lock = threading.Lock()

    @classmethod
    def configure(cls, settings: MLLoopSettings | None) -> None:
        """Install the process-wide settings (host calls this once at boot)."""
        with cls._configure_lock:
            cls._configured = settings

    @property
    def settings(self) -> MLLoopSettings:
        """Instance settings (``config["settings"]``), else the process-wide
        settings installed by :meth:`configure`, else the environment."""
        own = self.config.get("settings")
        if isinstance(own, MLLoopSettings):
            return own
        return self._configured or MLLoopSettings.from_env()

    async def run(self, operation: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.process({"operation": operation, "payload": payload or {}})

    async def process(self, input_data: Any, params: dict | None = None) -> dict[str, Any]:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}
        operation = data.get("operation") or params.get("operation") or "status"
        if "payload" in data:
            payload = data.get("payload") or {}
        else:
            payload = {k: v for k, v in data.items() if k != "operation"}
        if not isinstance(payload, dict):
            return self._error(operation, "payload must be an object")

        model = OPERATION_SCHEMAS.get(operation)
        if model is None:
            return self._error(
                operation,
                f"unknown operation; use one of: {', '.join(OPERATION_SCHEMAS)}",
            )
        try:
            request = model.model_validate(payload)
        except ValidationError as exc:
            return self._error(operation, "invalid payload", details=exc.errors(include_url=False))

        settings = self.settings
        if operation == "status":
            return self._status(settings, request.project_id)
        if not settings.store_configured:
            logger.info("%snot_configured op=%s", LOG_PREFIX, operation)
            return {"status": "not_configured", "operation": operation, "stored": False}
        if operation in WRITE_OPERATIONS and not settings.enabled:
            logger.info("%sdisabled op=%s", LOG_PREFIX, operation)
            return {"status": "disabled", "operation": operation, "stored": False}

        repo = MLLoopRepository(settings.session_factory)
        try:
            result = self._perform(operation, repo, settings, request)
        except ValueError as exc:
            return self._error(operation, str(exc))
        logger.info("%s%s ok", LOG_PREFIX, operation)
        return {"status": "success", "operation": operation, **result}

    def _perform(self, operation: str, repo, settings, request) -> dict[str, Any]:
        """Call the one handler for ``operation`` (explicit, not reflective,
        so every operation's code path is visible to readers and the gates)."""
        if operation == "record_turn":
            return self._op_record_turn(repo, settings, request)
        if operation == "record_feedback":
            return self._op_record_feedback(repo, settings, request)
        if operation == "record_correction":
            return self._op_record_correction(repo, settings, request)
        if operation == "record_route_correction":
            return self._op_record_route_correction(repo, settings, request)
        if operation == "shadow_predict":
            return self._op_shadow_predict(repo, settings, request)
        if operation == "import_tables":
            return self._op_import_tables(repo, settings, request)
        if operation == "propose":
            return self._op_propose(repo, settings, request)
        if operation == "shadow_report":
            return self._op_shadow_report(repo, settings, request)
        if operation == "export_tables":
            return self._op_export_tables(repo, settings, request)
        raise ValueError(f"operation has no handler: {operation}")

    # ── writes ──────────────────────────────────────────────────────────
    @staticmethod
    def _row(request, settings: MLLoopSettings) -> dict[str, Any]:
        """The validated payload as a row (enums as their values), stamped
        with the build sha and with free text capped by the settings."""
        row = request.model_dump(mode="json")
        row["build_sha"] = row.get("build_sha") or settings.build_sha
        for key in _CAPPED_TEXT:
            if isinstance(row.get(key), str):
                row[key] = row[key][: settings.max_text_chars]
        return row

    def _op_record_turn(self, repo, settings, request) -> dict[str, Any]:
        row_id = repo.insert("ml_turns", self._row(request, settings))
        return {"stored": True, "table": "ml_turns", "id": row_id}

    def _op_record_feedback(self, repo, settings, request) -> dict[str, Any]:
        # The note's 200-char cap is the schema's (RecordFeedback.note).
        row_id = repo.insert("ml_feedback", self._row(request, settings))
        return {"stored": True, "table": "ml_feedback", "id": row_id}

    def _op_record_correction(self, repo, settings, request) -> dict[str, Any]:
        row_id = repo.insert("ml_corrections", self._row(request, settings))
        samples = len(repo.rows("ml_corrections", where={"formula_id": request.formula_id}))
        return {"stored": True, "table": "ml_corrections", "id": row_id, "sample_count": samples}

    def _op_record_route_correction(self, repo, settings, request) -> dict[str, Any]:
        row_id = repo.insert("ml_route_corrections", self._row(request, settings))
        return {"stored": True, "table": "ml_route_corrections", "id": row_id}

    def _op_shadow_predict(self, repo, settings, request) -> dict[str, Any]:
        row_id = repo.insert("ml_shadow_predictions", self._row(request, settings))
        # Shadow means logged, never acted on: the live decision stands.
        return {
            "stored": True, "table": "ml_shadow_predictions", "id": row_id,
            "acted_on": False, "decision": request.live_label,
        }

    def _op_import_tables(self, repo, settings, request) -> dict[str, Any]:
        unknown = sorted(set(request.tables) - set(TABLES))
        if unknown:
            raise ValueError(f"unknown table(s): {unknown}")
        imported = {
            name: repo.import_rows(name, request.tables[name])
            for name in TABLES if name in request.tables
        }
        return {"stored": True, "imported": imported}

    # ── reads ───────────────────────────────────────────────────────────
    def _op_propose(self, repo, settings, request) -> dict[str, Any]:
        return _proposals.propose(
            repo, settings, project_id=request.project_id, window_days=request.window_days,
        )

    def _op_shadow_report(self, repo, settings, request) -> dict[str, Any]:
        return _shadow.shadow_report(
            repo, settings, predictor=request.predictor,
            project_id=request.project_id, window_days=request.window_days,
        )

    def _op_export_tables(self, repo, settings, request) -> dict[str, Any]:
        return {"tables": repo.export_tables(project_id=request.project_id)}

    def _status(self, settings: MLLoopSettings, project_id: str | None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": "success",
            "operation": "status",
            "enabled": settings.enabled,
            "store_configured": settings.store_configured,
            "build_sha": settings.build_sha,
            "operations": list(OPERATION_SCHEMAS),
        }
        if settings.store_configured:
            out["row_counts"] = MLLoopRepository(settings.session_factory).counts(project_id)
        return out

    @staticmethod
    def _error(operation: Any, message: str, **extra: Any) -> dict[str, Any]:
        logger.info("%serror op=%s %s", LOG_PREFIX, operation, message)
        return {"status": "error", "operation": operation, "error": message, **extra}
