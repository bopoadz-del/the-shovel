"""Live Google Drive acceptance test — skipped unless GOOGLE_CLIENT_ID is set
AND a token has been obtained by completing the OAuth flow once in a browser.
Run manually after connecting Drive."""
import os
import pytest
from app.core import drive_auth

pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_CLIENT_ID") or drive_auth.load_token("system") is None,
    reason="Needs GOOGLE_CLIENT_ID and a connected Drive (run /v1/drive/connect first)",
)


@pytest.mark.asyncio
async def test_live_get_access_token():
    token = await drive_auth.get_access_token("system")
    assert isinstance(token, str) and len(token) > 10


@pytest.mark.asyncio
async def test_live_list_files():
    from app.blocks.google_drive import GoogleDriveBlock
    token = await drive_auth.get_access_token("system")
    result = await GoogleDriveBlock().process(
        "", {"operation": "list", "access_token": token, "limit": 5})
    assert result["status"] == "success"
    assert isinstance(result["files"], list)
