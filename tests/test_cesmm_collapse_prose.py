"""Regression: the CESMM OCR collapse must not rewrite prose.

normalize_cesmm_item_codes REPLACES text at index time (chunk rows
feed the tsvector/FTS5 lexical columns), so any prose it touches
loses its real tokens. The first any-digit cut rewrote
'a 30 day notice' -> 'a30 day notice' in every reindexed prose doc.
Only the dotted CESMM shape (D 549.2) may collapse.
"""

from app.core.rag.vector_store import normalize_cesmm_item_codes as n


def test_prose_and_quantities_survive():
      assert n("a 30 day notice period") == "a 30 day notice period"
      assert n("A 30 DAY NOTICE PERIOD") == "A 30 DAY NOTICE PERIOD"
      assert n("Section A 1 of the works") == "Section A 1 of the works"
      assert n("Type B 2 storey block") == "Type B 2 storey block"
      assert n("a 100 mm dia pipe") == "a 100 mm dia pipe"
      assert n("I 100 percent agree") == "I 100 percent agree"


def test_dotted_cesmm_codes_still_collapse():
      assert n("D 549.2") == "D549.2"
      assert n("d 599.5") == "d599.5"
      assert n("I  112.3") == "I112.3"
      assert n("the rate for D 549.2 is 80.00") == "the rate for D549.2 is 80.00"
      assert n("D 549.2 and E 425.1 items") == "D549.2 and E425.1 items"


def test_reference_codes_unchanged():
      assert n("drawing QZ-ENG-320-0000-EXC-DWG") == "drawing QZ-ENG-320-0000-EXC-DWG"
      assert n("see AB-2023-101 Vol 1") == "see AB-2023-101 Vol 1"
  
