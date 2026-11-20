from __future__ import annotations

from unittest import mock

import pytest

from app.core.contract_data_chunks import (
    filled_particulars_rows,
    parse_contract_data_rows,
    per_search_parse_cache,
)

CHUNK_A = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage\n"
    "Time for Completion: 730 calendar days\n"
    "Delay damages: 0.05% of the Contract Price per day\n"
)

CHUNK_B = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage\n"
    "Accepted Contract Amount: AED 12,345,678.90\n"
    "Performance Security: 10% of the Accepted Contract Amount\n"
)

CHUNKS = [CHUNK_A, CHUNK_B]

BLANK_CHUNKS = ["", "\n", "\n\n", "   ", "\t"]


@pytest.mark.parametrize("chunk", BLANK_CHUNKS)
def test_filled_particulars_rows_blank_chunk_is_empty(chunk: str) -> None:
    """A chunk with no rows parses to nothing; the empty guard holds inside
    and outside a parse-cache scope."""
    assert filled_particulars_rows(chunk) == []
    with per_search_parse_cache():
        assert filled_particulars_rows(chunk) == []


@pytest.mark.parametrize("chunk", CHUNKS)
def test_filled_particulars_rows_identical_inside_and_outside_cache(chunk: str) -> None:
    """The memoised parse must return the exact same rows as the raw parse."""
    baseline = filled_particulars_rows(chunk)
    with per_search_parse_cache():
        assert filled_particulars_rows(chunk) == baseline
    # After the scope closes the cache is gone, so the raw path runs again.
    assert filled_particulars_rows(chunk) == baseline


@pytest.mark.parametrize("chunk", CHUNKS)
def test_cache_parses_each_chunk_once_per_search(chunk: str) -> None:
    """Inside one scope the underlying table parse runs once per chunk."""
    with mock.patch(
        "app.core.contract_data_chunks.parse_contract_data_rows",
        wraps=parse_contract_data_rows,
    ) as parse:
        with per_search_parse_cache():
            first = filled_particulars_rows(chunk)
            second = filled_particulars_rows(chunk)
        assert first == second
        assert parse.call_count == 1, (
            f"expected one parse for one chunk, got {parse.call_count}"
        )


@pytest.mark.parametrize("chunk", CHUNKS)
def test_cache_does_not_persist_across_searches(chunk: str) -> None:
    """A fresh scope re-parses; nothing is kept across searches."""
    with mock.patch(
        "app.core.contract_data_chunks.parse_contract_data_rows",
        wraps=parse_contract_data_rows,
    ) as parse:
        with per_search_parse_cache():
            filled_particulars_rows(chunk)
        with per_search_parse_cache():
            filled_particulars_rows(chunk)
        assert parse.call_count == 2, (
            f"expected one parse per search, got {parse.call_count}"
        )
