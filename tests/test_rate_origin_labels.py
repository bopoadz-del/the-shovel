"""STREAM D3: every route that can give a rate says where the rate comes from.

Chat can reach a rate through an encoded table (the benchmark block's own
table; the knowledge copy generated from the rate-card data file) or through
a document that speaks for itself. When the values differ, the answer states
each one with its origin, so each result must carry it. Inputs here are
constructed by the tests; nothing is copied from a product output.
"""
from __future__ import annotations

import asyncio
import os
import re

import pytest

from app.core import doc_typing

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.parametrize("text,source", [
    ("# A table copy\n\nGenerated from `app/lib/data/one_table.json` by a script.\n\n| a | b |\n",
     "app/lib/data/one_table.json"),
    ("Title\nGenerated from `config/other/rates.yaml`; edit the source.\nbody",
     "config/other/rates.yaml"),
])
def test_a_generated_copy_declares_its_encoded_table(text, source):
    assert doc_typing.declared_origin(text) == {"kind": "encoded_table", "source": source}


@pytest.mark.parametrize("text", [
    "# Construction rates\n\nIndicative rates. Source: a published survey, 2025.\n",
    # The phrase inside running text, not on a line of its own, is not a
    # declaration; nor is a path without the backtick quoting.
    "The values were generated from `somewhere` in prose.\nGenerated from app/x.json\n",
    None,
])
def test_a_document_without_the_declaration_speaks_for_itself(text):
    assert doc_typing.declared_origin(text) == {"kind": "document", "source": None}


def _knowledge_docs() -> list[str]:
    root = os.path.join(_REPO, "docs", "knowledge")
    return sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".md"))


def test_every_declared_source_of_a_knowledge_copy_exists():
    # General guard: a copy that names its encoded table must name a file
    # that is really in the repository (a renamed table breaks the label).
    declared = []
    for path in _knowledge_docs():
        with open(path, encoding="utf-8") as fh:
            origin = doc_typing.declared_origin(fh.read(4000))
        if origin["kind"] == "encoded_table":
            declared.append(origin["source"])
            assert os.path.isfile(os.path.join(_REPO, origin["source"])), origin
    assert declared, "no knowledge copy declares its encoded table"


@pytest.mark.parametrize("item,unit", [("concrete", "m3"), ("structural steel", "t")])
def test_a_benchmark_rate_carries_its_encoded_table_origin(item, unit):
    from app.blocks.historical_benchmark import HistoricalBenchmarkBlock

    block = HistoricalBenchmarkBlock()
    out = asyncio.run(block.process({}, {"item": item, "unit": unit, "location": "Riyadh"}))
    assert out["status"] == "success"
    origin = out["origin"]
    assert origin["kind"] == "encoded_table"
    # The label is the block's own declared description, a nameable phrase.
    assert origin["label"] == block.description and re.search(r"[A-Za-z]{2,}", origin["label"])
    assert os.path.isfile(os.path.join(_REPO, origin["source"]))
    assert origin["source"].startswith("app/") and "\\" not in origin["source"]


@pytest.mark.parametrize("item", ["unobtainium cladding", "zz no such item"])
def test_an_unpriced_item_states_no_origin(item):
    from app.blocks.historical_benchmark import HistoricalBenchmarkBlock

    out = asyncio.run(HistoricalBenchmarkBlock().process({}, {"item": item}))
    assert out["status"] == "error"
    assert "origin" not in out


@pytest.mark.parametrize("input_data,params", [
    ({"action": "record", "item": "concrete"}, {}),
    ({}, {"action": "record", "item": "structural steel"}),
])
def test_a_recorded_sample_is_not_a_priced_rate(input_data, params):
    # Recording a sample prices nothing, so it states no rate and no origin,
    # even when the payload names an item the table can price.
    from app.blocks.historical_benchmark import HistoricalBenchmarkBlock

    out = asyncio.run(HistoricalBenchmarkBlock().process(input_data, params))
    assert out["status"] == "success"
    assert out["action"] == "record"
    assert "rates" not in out and "origin" not in out
