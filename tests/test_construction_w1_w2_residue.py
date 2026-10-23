"""STEP 3 residue — W1 (orphaned methods) + W2 (digital_twin_sync name honesty).

W1: two redundant/dead orphaned methods were deleted (qa_inspection delegated to
the reachable qa_qc_inspection; progress_tracking only returned an honest-error
redirect to progress_tracker). The real reachable actions must still exist.

W2: digital_twin_sync PREPARES platform sync payloads but never pushes to a live
platform — its response must say so (sync_status) rather than imply a completed
sync.
"""

from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer


@pytest.fixture
def container():
    return ConstructionContainer()


class TestW1OrphansRemoved:
    def test_redundant_orphans_deleted(self, container):
        # The two redundant/dead orphaned methods are gone...
        assert not hasattr(container, "qa_inspection")
        assert not hasattr(container, "progress_tracking")

    def test_real_reachable_twins_remain(self, container):
        # ...while their real, reachable counterparts remain.
        assert hasattr(container, "qa_qc_inspection")
        assert hasattr(container, "progress_tracker")


class TestW2DigitalTwinSyncHonesty:
    @pytest.mark.asyncio
    async def test_sync_status_says_prepared_not_pushed(self, container):
        r = await container.digital_twin_sync(
            {"data": {"elements": [{"id": "w1", "type": "wall"}]}},
            {"platform": "bim360", "mode": "initial_sync"})
        assert r["status"] == "success"
        # The response must not imply a completed live sync.
        assert r["sync_status"] == "prepared_not_pushed"
        assert "note" in r and "no live push" in r["note"].lower()
