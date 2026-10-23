"""Aconex / CDE block — thin wrapper over app.core.cde.

Pulls CDE documents into the project corpus and posts RFI drafts back.
Does not own an RFI / submittal register.
"""

from __future__ import annotations

from typing import Any, Dict

from app.core.universal_base import UniversalBlock


class AconexBlock(UniversalBlock):
    auto_validate = False
    name = "aconex"
    version = "1.0"
    description = (
        "Aconex / CDE connector — list mail/documents/RFIs/transmittals, "
        "cache documents into the project corpus, post an RFI draft. "
        "Requires ACONEX_ENABLED + Oracle Lobby credentials (or CDE_ADAPTER=fake)."
    )
    layer = 4
    tags = ["integration", "cde", "aconex", "construction"]
    requires: list = []
    allow_empty_input = True

    ui_schema = {
        "input": {
            "type": "text",
            "placeholder": "CDE project id, or an RFI subject to post...",
            "multiline": False,
        },
        "output": {
            "type": "json",
            "fields": [{"name": "status", "type": "text", "label": "Status"}],
        },
        "quick_actions": [
            {"icon": "📥", "label": "Sync CDE", "prompt": "Pull Aconex documents into this project"},
            {"icon": "📤", "label": "Post RFI", "prompt": "Post this RFI draft to Aconex"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        from app.core.cde import (
            CdeError,
            CdeNotConfiguredError,
            default_cde_project_id,
            get_cde_client,
            oauth_client_ready,
            post_rfi_draft,
            sync_cde_documents,
        )

        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}
        operation = (
            params.get("operation")
            or data.get("operation")
            or "status"
        )
        cde_project_id = (
            params.get("cde_project_id")
            or data.get("cde_project_id")
            or default_cde_project_id()
        )
        fork_project_id = params.get("project_id") or data.get("project_id") or ""

        try:
            client = get_cde_client()
            if operation == "status":
                return {
                    "status": "success",
                    "operation": "status",
                    "vendor": client.vendor,
                    "oauth_client_ready": oauth_client_ready(),
                    "mode": "oauth" if oauth_client_ready() else "not_configured",
                    "note": (
                        "Aconex is the system of record. The Fork caches for RAG "
                        "and posts drafts; it does not keep an RFI register."
                    ),
                }
            if operation == "list_mail":
                items = await client.list_mail(cde_project_id)
                return {
                    "status": "success",
                    "operation": operation,
                    "mail": [m.as_dict() for m in items],
                }
            if operation == "list_documents":
                items = await client.list_documents(cde_project_id)
                return {
                    "status": "success",
                    "operation": operation,
                    "documents": [d.as_dict() for d in items],
                }
            if operation == "list_rfis":
                items = await client.list_rfis(cde_project_id)
                return {
                    "status": "success",
                    "operation": operation,
                    "rfis": [m.as_dict() for m in items],
                    "source_of_truth": "cde",
                }
            if operation == "list_transmittals":
                items = await client.list_transmittals(cde_project_id)
                return {
                    "status": "success",
                    "operation": operation,
                    "transmittals": [m.as_dict() for m in items],
                    "source_of_truth": "cde",
                }
            if operation == "get_mail":
                mail_id = str(params.get("mail_id") or data.get("mail_id") or "")
                item = await client.get_mail(cde_project_id, mail_id)
                return {"status": "success", "operation": operation, "mail": item.as_dict()}
            if operation == "get_document":
                document_id = str(
                    params.get("document_id") or data.get("document_id") or ""
                )
                item = await client.get_document(cde_project_id, document_id)
                return {
                    "status": "success",
                    "operation": operation,
                    "document": item.as_dict(),
                }
            if operation == "sync":
                if not fork_project_id:
                    return {
                        "status": "error",
                        "error": "project_id is required to cache CDE documents",
                    }
                from app.core import privileges

                result = await sync_cde_documents(
                    fork_project_id, cde_project_id,
                    eager_index=privileges.caller_may_add_to_project_rag(),
                )
                return {"status": "success", "operation": "sync", **result}
            if operation in ("post_rfi", "post_mail"):
                posted = await post_rfi_draft(cde_project_id, {**data, **params})
                return {
                    "status": "success",
                    "operation": operation,
                    "post": posted.as_dict(),
                    "source_of_truth": "cde",
                }
            return {"status": "error", "error": f"Unknown operation: {operation}"}
        except CdeNotConfiguredError as exc:
            return {"status": "error", "error": str(exc), "not_configured": True}
        except CdeError as exc:
            return {"status": "error", "error": str(exc)}
