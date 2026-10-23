"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests.conftest import listable_block_count
from tests._dev_key import DEV_KEY

client = TestClient(app, headers={"Authorization": "Bearer " + DEV_KEY})

class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint serves platform frontend HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "<!DOCTYPE html>" in response.text or "<html" in response.text
    
    def test_list_blocks(self):
        """Test listing all blocks."""
        response = client.get("/blocks")
        assert response.status_code == 200
        data = response.json()
        assert "blocks" in data
        assert "total" in data
        # count tracks boot mode (virgin generic vs kit / extended platform)
        assert data["total"] >= listable_block_count()
        
        # Check that a known generic block is included
        block_names = [b["name"] for b in data["blocks"]]
        assert "search" in block_names

    def test_get_block_info(self):
        """Test getting block info."""
        response = client.get("/blocks/search")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "search"
        assert "config" in data
    
    def test_get_nonexistent_block(self):
        """Test getting non-existent block."""
        response = client.get("/blocks/nonexistent")
        assert response.status_code == 404
    
    def test_health_endpoint(self):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "blocks" in data


class TestExecuteEndpoint:
    """Tests for execute endpoint."""
    
    def test_execute_chat_mock(self):
        """Test executing chat block with mock."""
        response = client.post("/execute", json={
            "block": "chat",
            "input": "Hello",
            "params": {"provider": "mock"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["block"] == "chat"
        assert "result" in data
    
    def test_execute_web_block(self):
        """`web` fetches any URL from inside the network: admin-only. the dev key
        is deliberately NOT an admin (test_hardening), so it is refused. What
        the block does is covered in tests/blocks/test_web.py."""
        response = client.post("/execute", json={
            "block": "web",
            "input": "<html><body>Test</body></html>",
            "params": {"operation": "html_parse"}
        })

        assert response.status_code == 403
    
    def test_execute_nonexistent_block(self):
        """Test executing non-existent block."""
        response = client.post("/execute", json={
            "block": "nonexistent",
            "input": "test",
            "params": {}
        })
        
        assert response.status_code == 404
    
    def test_execute_translate_mock(self):
        """Test executing translate block with mock."""
        response = client.post("/execute", json={
            "block": "translate",
            "input": "Hello",
            "params": {"provider": "mock", "target": "es"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["block"] == "translate"


class TestChainEndpoint:
    """Tests for chain endpoint — auth + privilege must not rot."""

    def test_chain_rejects_missing_auth(self):
        from fastapi.testclient import TestClient
        from app.main import app

        bare = TestClient(app)
        response = bare.post("/v1/chain", json={
            "steps": [{"block": "translate", "params": {"target": "en"}}],
            "initial_input": "Hello",
        })
        assert response.status_code in (401, 403)

    def test_chain_rejects_privileged_block_for_non_admin(self):
        response = client.post("/v1/chain", json={
            "steps": [{"block": "code", "params": {"operation": "execute"}}],
            "initial_input": "print(1)",
        })
        assert response.status_code == 403

    def test_chain_empty_steps(self):
        """Empty chain is auth-gated then handled (200 error envelope or 422)."""
        response = client.post("/v1/chain", json={
            "steps": [],
            "initial_input": "test"
        })
        assert response.status_code in [200, 422]

    def test_chain_non_privileged_block(self):
        response = client.post("/v1/chain", json={
            "steps": [
                {"block": "translate", "params": {"provider": "mock", "target": "es"}}
            ],
            "initial_input": "Hello World"
        })
        assert response.status_code == 200
        data = response.json()
        assert "steps_executed" in data
        assert "final_output" in data


@pytest.mark.extended_boot
class TestDriveEndpoints:
    """Tests for drive-specific endpoints (extended platform boot only)."""

    @pytest.fixture(autouse=True)
    def _require_extended_boot(self):
        from tests.conftest import is_extended_boot

        if not is_extended_boot():
            pytest.skip("requires CEREBRUM_VIRGIN=false")

    def test_local_drive_list(self):
        """`local_drive` reads and writes the data disk, which holds every
        user's uploads: admin-only, so the non-admin the dev key is refused.
        What the block does is covered in tests/blocks/test_local_drive.py."""
        response = client.post("/execute", json={
            "block": "local_drive",
            "input": "/",
            "params": {"operation": "list"}
        })

        assert response.status_code == 403
    
    def test_google_drive_mock(self):
        """`google_drive` falls back to a server-wide token when none is passed:
        admin-only, so the non-admin the dev key is refused. Users reach their
        own drive through /v1/drive/*."""
        response = client.post("/execute", json={
            "block": "google_drive",
            "input": {},
            "params": {"operation": "list"}
        })

        assert response.status_code == 403
    
    def test_onedrive_mock(self):
        """`onedrive` falls back to a server-wide token when none is passed:
        admin-only, so the non-admin the dev key is refused. Users reach their
        own drive through /v1/drive/*."""
        response = client.post("/execute", json={
            "block": "onedrive",
            "input": {},
            "params": {"operation": "list"}
        })

        assert response.status_code == 403
    
    def test_android_drive_paths(self):
        """Test Android Drive get_paths."""
        response = client.post("/execute", json={
            "block": "android_drive",
            "input": {},
            "params": {"operation": "get_paths"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["block"] == "android_drive"
        assert "paths" in data["result"]

