"""The bill row count and sum reads the derived sheet as the store keeps it.

``bill_derived`` writes the derived workbook through ``byte_store.write_bytes``,
which encrypts it when ``DATA_ENCRYPTION_KEY`` is set (as it is in production).
The aggregate must read those stored bytes, not hand the ciphertext to the
workbook reader; and a stored sheet that still is not a workbook must come back
as that bill's error, never as an exception that ends the chat turn.

Every input here is built by the test: invented rows, a key generated per test.
"""

from __future__ import annotations

import json
import time

import pytest
from cryptography.fernet import Fernet

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g
from app.core import file_crypto

#: Wall-clock budget for one count/sum tool call over a stored (encrypted)
#: derived sheet of ``_BIG_ROWS`` rows, decrypt and read included. The tool is
#: deterministic and makes no model call; this is the bound the test holds it to.
TOOL_BUDGET_S = 3.0
_BIG_ROWS = 4000

_PATH = ("Drainage", "Chambers")


def _row(page, ref, desc, unit, qty, amount):
    return g.BillRow(section=_PATH[-1], source_page=page,
                     fields={"description": desc, "unit": unit, "quantity": qty, "total": amount},
                     extra={"Item": ref}, path=_PATH)


ROWS = [
    _row(1, "C1", "Inspection chamber depth not exceeding 1.5 m", "nr", "4", "800"),
    _row(1, "C2", "Inspection chamber depth 1.5 to 2 m", "nr", "3", "900"),
    _row(2, "C3", "Pipe 300 mm bore in trench", "m", "120.5", "2410"),
    _row(2, "C4", "Pipe 450 mm bore in trench", "m", "60", "1800"),
]


def _workbook_bytes(tmp_path, rows) -> bytes:
    plain = tmp_path / "plain.xlsx"
    g.write_bill_workbook(rows, str(plain))
    return plain.read_bytes()


def _store(tmp_path, name, data: bytes) -> str:
    """Write ``data`` the way the store writes a derived sheet."""
    path = tmp_path / name
    file_crypto.write_document(str(path), data)
    return str(path)


@pytest.fixture(params=["encrypted", "plaintext"])
def key_mode(request, monkeypatch):
    if request.param == "encrypted":
        monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    else:
        monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    return request.param


@pytest.mark.parametrize("terms, count, total", [
    ("chamber", 2, 7.0),
    ("pipe trench", 2, 180.5),
])
def test_reads_the_stored_sheet_with_or_without_encryption(tmp_path, key_mode, terms, count, total):
    path = _store(tmp_path, "d.bill.xlsx", _workbook_bytes(tmp_path, ROWS))
    with open(path, "rb") as fh:
        on_disk = fh.read()
    assert file_crypto.looks_encrypted(on_disk) is (key_mode == "encrypted")
    out = agg.aggregate(agg.read_sheet_rows(path), terms=terms)
    assert (out["count"], out["sum"]) == (count, total)


def _project(monkeypatch, docs):
    from app.core import projects

    monkeypatch.setattr(projects, "list_documents", lambda pid, **_kw: docs)
    monkeypatch.setattr(projects, "materialize_document_path", lambda d: d["path"])


def _derived(doc_id, path):
    from app.core import bill_derived

    return {"id": doc_id, "original_name": f"{doc_id}.bill.xlsx", "path": path,
            "metadata": {"derivation": bill_derived.DERIVATION, "derived_from": f"src-{doc_id}"}}


def _not_a_workbook(tmp_path, monkeypatch):
    return _store(tmp_path, "bad.bill.xlsx", b"these bytes are not a workbook")


def _truncated_workbook(tmp_path, monkeypatch):
    return _store(tmp_path, "cut.bill.xlsx", _workbook_bytes(tmp_path, ROWS)[:300])


def _other_key(tmp_path, monkeypatch):
    """Stored under one key, read under another (a rotated key)."""
    data = _workbook_bytes(tmp_path, ROWS)
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    path = _store(tmp_path, "old-key.bill.xlsx", data)
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    return path


@pytest.mark.parametrize("make_bad", [_not_a_workbook, _truncated_workbook, _other_key],
                         ids=["not-a-workbook", "truncated", "other-key"])
def test_an_unreadable_sheet_is_that_bills_error_and_the_others_still_count(
        tmp_path, monkeypatch, make_bad):
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    bad = make_bad(tmp_path, monkeypatch)
    good = _store(tmp_path, "good.bill.xlsx", _workbook_bytes(tmp_path, ROWS))
    _project(monkeypatch, [_derived("bad", bad), _derived("good", good)])

    out = agg.aggregate_project_bills("p", terms="chamber")

    assert out["status"] == "success"
    by_id = {b["document_id"]: b for b in out["bills"]}
    assert by_id["bad"]["status"] == "error"
    assert by_id["bad"]["reason"] == agg.UNREADABLE_SHEET
    assert (by_id["good"]["status"], by_id["good"]["count"], by_id["good"]["sum"]) == ("success", 2, 7.0)


def _tool_call(args):
    return {"function": {"name": "bill_rows_aggregate", "arguments": json.dumps(args)}}


@pytest.fixture
def runtime():
    from app.agents.runtime import Agent
    return Agent(name="test-agent", description="t", system_prompt="t", allowed_blocks=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_first", [True, False], ids=["bad-first", "bad-last"])
async def test_the_tool_answers_through_the_agent_when_a_sheet_is_unreadable(
        tmp_path, monkeypatch, runtime, bad_first):
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    bad = _derived("bad", _not_a_workbook(tmp_path, monkeypatch))
    good = _derived("good", _store(tmp_path, "good.bill.xlsx", _workbook_bytes(tmp_path, ROWS)))
    _project(monkeypatch, [bad, good] if bad_first else [good, bad])

    out = await runtime._run_tool_call(_tool_call({"terms": "pipe", "measure": "amount"}),
                                       project_id="p")

    assert out["ok"] is True
    sums = {b["document_id"]: b.get("sum") for b in out["result"]["bills"]}
    assert sums == {"bad": None, "good": 4210.0}


def _big_rows(n):
    return [_row(1 + i // 40, f"R{i}", f"Chamber {'deep' if i % 5 == 0 else 'shallow'} ring {i}", "nr", "2", "100")
            for i in range(n)]


@pytest.mark.serial
@pytest.mark.timing
@pytest.mark.asyncio
@pytest.mark.parametrize("terms, count", [("chamber", _BIG_ROWS), ("deep chamber", _BIG_ROWS // 5)])
async def test_count_and_sum_over_a_stored_sheet_answers_inside_the_budget(
        tmp_path, monkeypatch, runtime, terms, count):
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    path = _store(tmp_path, "big.bill.xlsx", _workbook_bytes(tmp_path, _big_rows(_BIG_ROWS)))
    _project(monkeypatch, [_derived("big", path)])

    started = time.perf_counter()
    out = await runtime._run_tool_call(_tool_call({"terms": terms}), project_id="p")
    elapsed = time.perf_counter() - started

    print(f"bill count/sum over {_BIG_ROWS} rows: {elapsed:.3f}s (budget {TOOL_BUDGET_S}s)")
    bill = out["result"]["bills"][0]
    assert (bill["status"], bill["count"], bill["sum"]) == ("success", count, 2.0 * count)
    assert elapsed < TOOL_BUDGET_S, f"count/sum took {elapsed:.2f}s, budget {TOOL_BUDGET_S}s"
