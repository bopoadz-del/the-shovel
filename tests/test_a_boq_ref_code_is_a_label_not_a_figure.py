"""A BOQ item reference is an identifier. Retrieve it, display it, cite it --
never ground a cost claim against it.

``D529.2`` is the item number of a bill row. The grounding gate read it as the
number 529.2 because it is digits in a retrieved chunk, and ``A.1.2.3`` as 1.2
and 3. Label numbers then sat in the set a cost figure may be traced to.

The taxonomy is deliberately narrow: material grades and bar sizes look similar
and their numbers ARE real (T12 is a 12 mm bar, C30/37 a strength class), as
are plain decimals. Synthetic rows and codes throughout.
"""
import json

import pytest

from app.agents import runtime as rt
from app.core.rag import retriever
from app.core.rag.vector_store import Chunk
from app.lib import boq_ref_codes as rc

IDENTIFIERS = ["D529.2", "D.589.1", "A.1.2.3", "E101.4", "D.999.9", "BQ.12.4", "1.2.3"]
FIGURES = ["T12", "C30", "C30/37", "M20", "B500B", "4.2", "18.75", "142.00", "0.75", "34,844"]


# ── taxonomy ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", IDENTIFIERS)
def test_an_item_reference_is_an_identifier(code):
    assert rc.is_ref_code(code)
    assert rc.strip_ref_codes(f"item {code} follows") == "item   follows"


@pytest.mark.parametrize("token", FIGURES)
def test_a_grade_size_or_plain_decimal_stays_a_figure(token):
    assert not rc.is_ref_code(token)
    assert rc.strip_ref_codes(f"value {token} here") == f"value {token} here"


def test_a_priced_row_keeps_every_real_figure():
    row = "D529.2 Breaking out existing wall 34,844 m 142.00 4,947,848.00"
    assert rc.find_ref_codes(row) == ["D529.2"]
    stripped = rc.strip_ref_codes(row)
    for figure in ("34,844", "142.00", "4,947,848.00"):
        assert figure in stripped
    assert "529.2" not in stripped


# ── the gate never sees a ref code as a figure ─────────────────────────────

def _grounded_from_row(row):
    rag = f"[doc_id=b1 chunk=0] Priced bill, rate and amount per item: {row}"
    return rt._cg_grounded_numbers(rag, [])


def test_the_gate_does_not_ground_a_ref_code():
    grounded = _grounded_from_row("D.589.1 Excavate in rock 120 m3 88.00 10,560.00")
    assert not rt._cg_is_grounded(589.1, grounded), "589.1 is an item number"
    assert rt._cg_is_grounded(10560.0, grounded), "the row's amount still grounds"
    assert rt._cg_is_grounded(88.0, grounded), "the row's rate still grounds"


def test_a_ref_code_cannot_ground_an_invented_figure_by_arithmetic():
    # 529.2 x 142.00 = 75,146.40 -- a product of a LABEL and a rate, which is
    # not a figure this bill contains.
    grounded = _grounded_from_row("D529.2 Breaking out existing wall 34,844 m 142.00 4,947,848.00")
    assert not rt._cg_is_grounded(75146.40, grounded)
    assert rt._cg_is_grounded(4947848.0, grounded)


def test_a_ref_code_in_a_tool_result_is_not_a_figure_either():
    tool = json.dumps({"status": "success", "line_items": [
        {"item": "D529.2", "quantity": 34844, "unit_cost": 142.0, "total_cost": 4947848.0}]})
    grounded = rt._cg_grounded_numbers("", [{"role": "tool", "content": tool}])
    assert not rt._cg_is_grounded(529.2, grounded)
    assert rt._cg_is_grounded(4947848.0, grounded)


def test_a_grade_in_a_chunk_still_grounds():
    grounded = _grounded_from_row("Blinding C30/37 with T12 bars, rate 4.2 per m2")
    assert rt._cg_is_grounded(12.0, grounded), "T12's 12 mm is a real figure"
    assert rt._cg_is_grounded(30.0, grounded)


# ── turn 1 still answers: the code is retrieved, labelled and displayed ────

def test_retrieval_labels_the_codes_without_touching_the_text(monkeypatch):
    row = "D529.2 Breaking out existing concrete wall/barrier 34,844 m 142.00 4,947,848.00"
    plain = "Concrete C30/37 with T12 bars at 4.2 m centres"
    seeded = [
        Chunk(chunk_id="c1", project_id="p1", doc_id="d1", chunk_index=0, text=row, score=0.91),
        Chunk(chunk_id="c2", project_id="p1", doc_id="d1", chunk_index=1, text=plain, score=0.80),
    ]
    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search",
                        lambda self, project_id, qvec, k, query_text=None:
                        [c for c in seeded if c.project_id == project_id][:k])
    monkeypatch.setattr(retriever, "_doc_name_for_id", lambda did: "synthetic bill.pdf", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")

    chunks, _noise = retriever.retrieve_with_filter("quantity rate and amount for D529.2", "p1", k=5)
    by_text = {c.text: c for c in chunks}
    assert row in by_text, "the priced row is retrieved"
    assert by_text[row].ref_codes == ("D529.2",), "retrieval labels the identifier"
    assert by_text[row].text == row, "text untouched -- display and citation unchanged"
    if plain in by_text:
        assert by_text[plain].ref_codes == (), "a grade is not an identifier"


def test_an_item_lookup_answer_is_not_refused():
    # Turn 1 of the live pair: quantity, rate and amount for the item, with the
    # code quoted in the answer. The gate must pass it.
    rag = {"role": "system", "content": (
        "[doc_id=b1 chunk=0] Priced bill of quantities, rate and amount columns: "
        "D529.2 Breaking out existing concrete wall/barrier 34,844 m 142.00 4,947,848.00")}
    answer = ("**D529.2 -- Breaking out existing concrete wall/barrier**\n\n"
              "| Quantity | 34,844 m |\n| Rate | SAR 142.00 / m |\n"
              "| Amount | SAR 4,947,848.00 |")
    messages = [{"role": "user", "content": "What is the quantity, rate and amount for D529.2?"}]
    assert rt._cost_grounding_gate(answer, rag, messages) == answer
