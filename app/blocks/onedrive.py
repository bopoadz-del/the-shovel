"""OneDrive Block - real MSAL + Microsoft Graph API"""

import base64
import os
from typing import Any, Dict

from app.core.universal_base import UniversalBlock

_GRAPH_API = "https://graph.microsoft.com/v1.0"
_AUTH_BASE = "https://login.microsoftonline.com"
_SCOPES = ["Files.Read", "Files.ReadWrite"]


def _auth_url() -> str:
    client_id = os.getenv("AZURE_CLIENT_ID", "")
    tenant = os.getenv("AZURE_TENANT_ID", "common")
    redirect = os.getenv("AZURE_REDIRECT_URI", "http://localhost:8000/auth/onedrive/callback")
    if not client_id:
        return ""
    scope = "%20".join(_SCOPES + ["offline_access"])
    return (
        f"{_AUTH_BASE}/{tenant}/oauth2/v2.0/authorize"
        f"?client_id={client_id}"
        f"&response_type=code"
        f"&redirect_uri={redirect}"
        f"&scope={scope}"
        f"&response_mode=query"
    )


class OneDriveBlock(UniversalBlock):
    """OneDrive: list, read, download files via Microsoft Graph + MSAL"""

    auto_validate = False
    name = "onedrive"
    version = "2.0"
    description = "OneDrive file operations — set AZURE_CLIENT_ID + ONEDRIVE_ACCESS_TOKEN"
    layer = 4
    tags = ["integration", "storage", "cloud", "microsoft"]
    requires = []

    ui_schema = {
        "input": {
            "type": "text",
            "accept": ["*/*"],
            "placeholder": "File path, ID, or search query...",
            "multiline": False,
        },
        "output": {
            "type": "list",
            "fields": [{"name": "files", "type": "array", "label": "Files"}],
        },
        "quick_actions": [
            {"icon": "☁️", "label": "Browse OneDrive", "prompt": "List files from OneDrive"},
            {"icon": "🔑", "label": "Auth", "prompt": "Authenticate with OneDrive"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        operation = params.get("operation", "list")

        query = ""
        if isinstance(input_data, str):
            query = input_data
        elif isinstance(input_data, dict):
            query = input_data.get("query") or input_data.get("text") or ""
            operation = input_data.get("operation", operation)

        # ── Auth / status ─────────────────────────────────────────────────────
        if operation in ("auth", "status"):
            has_token = bool(os.getenv("ONEDRIVE_ACCESS_TOKEN"))
            has_creds = bool(os.getenv("AZURE_CLIENT_ID"))
            url = _auth_url()
            return {
                "status": "success",
                "operation": "auth",
                "authenticated": has_token,
                "credentials_configured": has_creds,
                "auth_url": url or None,
                "instructions": (
                    "Visit auth_url in a browser, approve, then set ONEDRIVE_ACCESS_TOKEN env var."
                    if url and not has_token else
                    "Set AZURE_CLIENT_ID + AZURE_TENANT_ID as env vars to enable OAuth."
                    if not has_creds else
                    "Access token set. Use operation=list to browse files."
                ),
            }

        # ── List files ────────────────────────────────────────────────────────
        if operation == "list":
            access_token = params.get("access_token") or os.getenv("ONEDRIVE_ACCESS_TOKEN", "")
            if not access_token:
                return {
                    "status": "error",
                    "error": "Not authenticated. Run with operation=auth to get the auth URL.",
                    "auth_url": _auth_url() or None,
                }
            try:
                import httpx
                endpoint = (
                    f"{_GRAPH_API}/me/drive/search(q='{query}')"
                    if query else
                    f"{_GRAPH_API}/me/drive/root/children"
                )
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(
                        endpoint,
                        headers={"Authorization": f"Bearer {access_token}"},
                        params={
                            "$select": "id,name,size,lastModifiedDateTime,webUrl,file,folder",
                            "$top": params.get("limit", 20),
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()

                items = [
                    {
                        "id": i.get("id"),
                        "name": i.get("name"),
                        "type": "folder" if "folder" in i else i.get("file", {}).get("mimeType", "file").split("/")[-1],
                        "size_bytes": i.get("size", 0),
                        "modified": (i.get("lastModifiedDateTime") or "")[:10],
                        "url": i.get("webUrl", ""),
                    }
                    for i in data.get("value", [])
                ]
                return {"status": "success", "operation": "list", "files": items, "total": len(items)}
            except Exception as e:
                return {"status": "error", "error": str(e), "operation": "list"}

        # ── Download file ─────────────────────────────────────────────────────
        if operation == "download":
            item_id = query or params.get("file_id", "")
            access_token = params.get("access_token") or os.getenv("ONEDRIVE_ACCESS_TOKEN", "")
            if not item_id:
                return {"status": "error", "error": "file_id required for download"}
            if not access_token:
                return {"status": "error", "error": "Not authenticated"}
            try:
                import httpx
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                    resp = await client.get(
                        f"{_GRAPH_API}/me/drive/items/{item_id}/content",
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                    resp.raise_for_status()
                return {
                    "status": "success",
                    "operation": "download",
                    "file_id": item_id,
                    "size_bytes": len(resp.content),
                    "content_base64": base64.b64encode(resp.content).decode(),
                }
            except Exception as e:
                return {"status": "error", "error": str(e), "operation": "download"}

        return {"status": "error", "error": f"Unknown operation: {operation}. Use: auth, list, download"}
