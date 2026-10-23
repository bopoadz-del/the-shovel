"""Tests for the dynamic block catalogue shim — Reasoning Engine."""

from app import blocks as blocks_module
from app.core.catalogue import (
    build_tool_catalogue,
    format_catalogue_for_prompt,
    get_block,
    list_blocks,
)
from app.prompts.reasoner_system import build_reasoner_prompt
from app.core.session_store import InMemorySessionStore

import pytest


def test_list_blocks_delegates_to_registry():
    names = list_blocks()
    assert isinstance(names, list)
    assert "pdf" in names  # generic block present in the default virgin boot


def test_get_block_delegates_to_registry():
    assert get_block("pdf") is blocks_module.get_block("pdf")
    assert get_block("definitely_not_a_real_block") is None


def test_build_tool_catalogue_includes_registered_blocks():
    cat = build_tool_catalogue()
    names = [t["name"] for t in cat]
    assert "pdf" in names
    assert "ocr" in names
    for entry in cat:
        assert "name" in entry
        assert "description" in entry
        assert "inputSchema" in entry


def test_build_tool_catalogue_excludes_mcp_adapter():
    names = [t["name"] for t in build_tool_catalogue()]
    assert "mcp_adapter" not in names


def test_format_catalogue_for_prompt_renders_blocks():
    cat = build_tool_catalogue()
    prompt = format_catalogue_for_prompt(cat)
    assert "AVAILABLE BLOCKS" in prompt
    assert "pdf" in prompt


def test_format_catalogue_for_prompt_empty():
    prompt = format_catalogue_for_prompt([])
    assert "none registered" in prompt.lower()


def test_format_catalogue_defaults_to_runtime_catalogue():
    prompt = format_catalogue_for_prompt()
    assert "AVAILABLE BLOCKS" in prompt
    # The runtime catalogue should contain at least the generic blocks.
    assert "pdf" in prompt


@pytest.mark.asyncio
async def test_reasoner_prompt_includes_dynamic_catalogue():
    session = await InMemorySessionStore().get_or_create("catalogue-test")
    prompt = build_reasoner_prompt(session, "what tools do I have?")
    assert "AVAILABLE BLOCKS" in prompt
    assert "pdf" in prompt
