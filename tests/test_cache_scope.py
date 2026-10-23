"""Cache keys must include project/contract id AND source class.

#443-via-cache: an unscoped key lets AB-2022's answer satisfy a DD-2023
question. This file proves that shape FAILS on a deliberately unscoped
key, then PASSES on the scoped implementation.
"""

from __future__ import annotations

import pytest

from app.blocks.cache_manager import CacheManagerBlock, compose_scoped_key


def _unscoped_key(question: str) -> str:
    """The key shape that would have shipped #443 through the cache."""
    return question


@pytest.mark.asyncio
async def test_unscoped_key_allows_a_cross_contract_hit():
    """The defect: same question, two contracts, one key → HIT."""
    block = CacheManagerBlock()
    question = "What is the Time for Completion?"
    key = _unscoped_key(question)
    await block.set(
        {},
        {"key": key, "value": {"contract": "AB-2022-202", "days": 365}, "ttl": 60},
    )
    hit = await block.get({}, {"key": _unscoped_key(question)})
    assert hit["found"] is True
    assert hit["value"]["contract"] == "AB-2022-202"
    # A later turn on a different contract would have received 365 days.
    assert hit["key"] == key
    assert "AB-2023" not in hit["key"]
    assert "class=" not in hit["key"]


@pytest.mark.asyncio
async def test_scoped_key_makes_a_cross_contract_hit_impossible():
    block = CacheManagerBlock()
    question = "What is the Time for Completion?"
    await block.set(
        {
            "key": question,
            "project_id": "proj-a",
            "contract_id": "AB-2022-202",
            "source_class": "project_corpus",
            "value": {"contract": "AB-2022-202", "days": 365},
            "ttl": 60,
        },
        {},
    )
    other = await block.get(
        {
            "key": question,
            "project_id": "proj-a",
            "contract_id": "AB-2023-101",
            "source_class": "project_corpus",
        },
        {},
    )
    assert other["found"] is False
    assert "AB-2023-101" in other["key"]
    assert "class=project_corpus" in other["key"]
    assert "p=proj-a" in other["key"]


@pytest.mark.asyncio
async def test_scoped_key_includes_project_contract_and_source_class():
    key = compose_scoped_key(
        "delay-damages",
        project_id="proj-a",
        contract_id="AB-2023-101",
        source_class="template",
    )
    assert "p=proj-a" in key
    assert "c=AB-2023-101" in key
    assert "class=template" in key
    other_class = compose_scoped_key(
        "delay-damages",
        project_id="proj-a",
        contract_id="AB-2023-101",
        source_class="project_corpus",
    )
    assert key != other_class


@pytest.mark.asyncio
async def test_template_class_cannot_cache_hit_this_contract():
    """G1-shaped: a template answer must not satisfy a project_corpus get."""
    block = CacheManagerBlock()
    await block.set(
        {},
        {
            "key": "schedule-10",
            "project_id": "proj-a",
            "contract_id": "AB-2023-101",
            "source_class": "template",
            "value": "sets out any applicable Works Guarantees",
            "ttl": 60,
        },
    )
    hit = await block.get(
        {},
        {
            "key": "schedule-10",
            "project_id": "proj-a",
            "contract_id": "AB-2023-101",
            "source_class": "project_corpus",
        },
    )
    assert hit["found"] is False
