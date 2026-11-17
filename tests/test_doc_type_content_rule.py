"""Document typing reads the document's own text, not only its name.

Ingest types a row before its text exists, so a name with no type word left
it ``document`` whatever the text said. The backfill now samples the
document's opening chunks; content alone decides only with enough distinct
evidence for one kind and no tie between kinds. Every input is invented here.
"""
from __future__ import annotations

import json

import pytest

from app.core import doc_types
from app.core import doc_typing_backfill as bf
from tests.test_ingest_reconcile import _reload_projects


@pytest.fixture(autouse=True)
def _no_folder_roles(monkeypatch):
    monkeypatch.setenv("DOC_ROLE_FOLDERS", json.dumps({}))


# ── the content-only bar ─────────────────────────────────────────────────────

@pytest.mark.parametrize("content, expected", [
    # two distinct programme phrases, nothing else: content decides
    ("Annex 4. The critical path runs through milestone M3; early start 02 Mar.",
     "schedule"),
    # one programme phrase only: too weak to decide alone
    ("Annex 4. Payment falls due at milestone M3.", "unrecognised"),
])
def test_content_alone_needs_two_distinct_hits(content, expected):
    assert doc_types.classify("annex_04.pdf", content)["name"] == expected


@pytest.mark.parametrize("content, expected", [
    # two kinds score the same on content alone: no kind is preferred
    ("The critical path and milestone dates. Dear Sir, yours faithfully.",
     "unrecognised"),
    # one kind ahead: it wins
    ("The critical path, milestone and early start dates. Dear Sir, yours faithfully.",
     "schedule"),
])
def test_a_content_tie_between_kinds_is_not_broken_by_table_order(content, expected):
    assert doc_types.classify("item_0091.pdf", content)["name"] == expected


@pytest.mark.parametrize("name, expected", [
    # a name match keeps today's rule: one content word may add to it
    ("Weekly_Report_07.pdf", "report"),
    ("Item_07.pdf", "unrecognised"),
])
def test_a_name_match_is_not_held_to_the_content_bar(name, expected):
    assert doc_types.classify(name, "manpower")["name"] == expected


# ── multi-word name keywords across separators ───────────────────────────────

@pytest.mark.parametrize("name, expected", [
    # "inspection and test plan" across underscores beats the one-word match
    ("Inspection_and_Test_Plan_L3.pdf", "itp"),
    ("Inspection_Request_L3.pdf", "inspection_request"),
])
def test_multi_word_keywords_match_across_separators(name, expected):
    assert doc_types.classify(name)["name"] == expected


@pytest.mark.parametrize("name, expected", [
    ("Method_Statement_Blockwork_L2.docx", "method_statement"),
    ("Blockwork_L2_Statement.docx", "unrecognised"),
])
def test_method_statement_is_a_kind_in_the_table(name, expected):
    assert doc_types.classify(name)["name"] == expected


# ── the backfill reads the document's own chunks ─────────────────────────────

def _project_with_doc(monkeypatch, tmp_path, name, texts):
    import numpy as np
    from sqlalchemy import update

    from app.core.db import SessionLocal, engine
    from app.core.models import Document, RagChunk

    pm = _reload_projects(monkeypatch, tmp_path)
    RagChunk.__table__.create(bind=engine, checkfirst=True)
    pm.create_project("cproj", user_id="system", project_id="cproj")
    did = pm.add_document("cproj", name, size=1)["id"]
    with SessionLocal() as s:
        s.execute(update(Document).where(Document.id == did)
                  .values(doc_type="document", doc_role="other"))
        for i, text in enumerate(texts):
            s.add(RagChunk(chunk_id=f"{did}-{i}", project_id="cproj", doc_id=did,
                           chunk_index=i, text=text,
                           embedding=np.zeros(8, dtype=np.float32),
                           created_at="2026-01-01T00:00:00Z"))
        s.commit()
    return did


@pytest.mark.parametrize("texts, counts", [
    # the name says nothing; the opening chunks are a method statement
    (["1. Scope. This method statement covers blockwork to level 2.",
      "2. Sequence of work: set out, lay, cure. 3. Risk assessment attached."],
     {"method_statement": 1}),
    # the same name, text that names no kind: nothing is filled
    (["1. Scope. Blockwork to level 2.", "2. Lay and cure."], {}),
])
def test_backfill_types_a_document_by_its_own_text(monkeypatch, tmp_path, texts, counts):
    did = _project_with_doc(monkeypatch, tmp_path, "DOC-0417-r1.pdf", texts)
    plan = bf.doc_typing_backfill(project_id="cproj")["plan"]
    assert plan["counts"]["doc_type"] == counts
    assert plan["documents"] == ([did] if counts else [])


@pytest.mark.parametrize("make_table", [False, True])
def test_no_chunks_means_no_sample(monkeypatch, tmp_path, make_table):
    # no chunks table at all (a fresh database), and a table with no chunks
    from app.core.db import SessionLocal, engine
    from app.core.models import RagChunk

    pm = _reload_projects(monkeypatch, tmp_path)
    if make_table:
        RagChunk.__table__.create(bind=engine, checkfirst=True)
    pm.create_project("cproj", user_id="system", project_id="cproj")
    with SessionLocal() as s:
        assert bf.content_samples(s, "cproj") == {}


@pytest.mark.parametrize("name, content, expected", [
    # nothing in the name, extension or text names a kind
    ("", "", "unrecognised"),
    # the extension alone names a kind
    ("plan.xer", "", "schedule"),
])
def test_a_document_matching_nothing_is_unrecognised(name, content, expected):
    assert doc_types.classify(name, content)["name"] == expected


@pytest.mark.parametrize("env, expected", [
    # a zero bar is clamped to one: a document matching nothing stays unrecognised
    ("0", 1),
    ("3", 3),
])
def test_the_content_bar_is_never_below_one(monkeypatch, env, expected):
    import importlib

    monkeypatch.setenv("DOC_TYPE_CONTENT_MIN_HITS", env)
    try:
        assert importlib.reload(doc_types).CONTENT_ONLY_MIN_HITS == expected
    finally:
        monkeypatch.delenv("DOC_TYPE_CONTENT_MIN_HITS")
        importlib.reload(doc_types)


@pytest.mark.parametrize("limit, expected", [
    (10_000, "first chunk\nsecond chunk"),
    (8, "first ch"),
])
def test_content_sample_is_opening_chunks_in_order_and_capped(monkeypatch, tmp_path, limit, expected):
    from app.core.db import SessionLocal

    did = _project_with_doc(monkeypatch, tmp_path, "x.pdf", ["first chunk", "second chunk"])
    with SessionLocal() as s:
        assert bf.content_samples(s, "cproj", limit=limit) == {did: expected}


# ── a tie on the NAME path is unrecognised too (no order among kinds) ────────

@pytest.mark.parametrize("name, expected", [
    # one word for each of two kinds: a tie, so no kind is chosen
    ("Site_photo_model_04.pdf", "unrecognised"),
    # the same name with one kind's word only
    ("Site_photo_04.pdf", "photo"),
])
def test_a_name_tie_between_kinds_is_unrecognised(name, expected):
    assert doc_types.classify(name)["name"] == expected


@pytest.mark.parametrize("content, expected", [
    # the name ties two kinds; the text adds nothing: still a tie
    ("", "unrecognised"),
    # the text adds one word for one of them: it is ahead, so it wins
    ("Attendees: the site team.", "minutes"),
])
def test_content_can_break_a_name_tie_by_score(content, expected):
    assert doc_types.classify("Daily_meeting_12.pdf", content)["name"] == expected


# ── Arabic: every kind in the table has Arabic words ─────────────────────────

@pytest.mark.parametrize("name, expected", [
    ("مخطط الدور الأرضي.pdf", "drawing"),
    ("البرنامج الزمني المحدث.pdf", "schedule"),
    ("جدول الكميات - المرحلة الأولى.xlsx", "boq"),
    ("عقد المقاولة الرئيسي.pdf", "contract"),
    ("تقرير عدم مطابقة 012.pdf", "ncr"),
    ("طلب فحص أعمال الخرسانة.pdf", "inspection_request"),
    ("خطة الفحص والاختبار للخرسانة.pdf", "itp"),
    ("طلب اعتماد مواد العزل.pdf", "material_submittal"),
    ("طريقة العمل لأعمال الحفر.docx", "method_statement"),
    ("المواصفات الفنية للخرسانة.pdf", "specification"),
    ("صور الموقع - الأسبوع 12.pdf", "photo"),
    ("نمذجة معلومات البناء - المبنى أ.pdf", "bim"),
    ("تقرير يومي 15.pdf", "report"),
    ("محضر اجتماع الموقع 7.pdf", "minutes"),
    ("خطاب إلى الاستشاري.pdf", "correspondence"),
    ("قالب فارغ.docx", "template"),
    # an Arabic name with no type word is unrecognised, never mistyped
    ("مستند 7.pdf", "unrecognised"),
])
def test_arabic_names_type_by_the_table(name, expected):
    assert doc_types.classify(name)["name"] == expected


@pytest.mark.parametrize("content, expected", [
    ("يمر المسار الحرج عبر المعالم الرئيسية للمشروع.", "schedule"),
    ("تسلسل الأعمال وتقييم المخاطر قبل الحفر.", "method_statement"),
    ("يلتزم صاحب العمل وتطبق غرامات التأخير.", "contract"),
    ("الحضور: فريق الموقع. الاجتماع القادم يوم الأحد.", "minutes"),
    # one Arabic phrase only: below the content bar
    ("يمر المسار الحرج عبر المنطقة.", "unrecognised"),
])
def test_arabic_content_alone_types_with_two_hits(content, expected):
    assert doc_types.classify("annex_04.pdf", content)["name"] == expected


@pytest.mark.parametrize("name, content", [
    # written with hamza, ta marbuta and short vowels
    ("إتفاقية الخدمات.pdf", "وفقاً للمخططات، يجب أنْ يكونَ الخرسانة"),
    # the same words written without them
    ("اتفاقيه الخدمات.pdf", "وفقا للمخططات، يجب ان يكون الخرسانة"),
])
def test_arabic_spelling_variants_fold_to_one_form(name, content):
    assert doc_types.classify(name)["name"] == "contract"
    assert doc_types.classify("annex_04.pdf", content)["name"] == "specification"


# ── the change as seen by ingest (newly uploaded names) ──────────────────────

@pytest.mark.parametrize("name, expected", [
    # was inspection_request by table order; the full phrase now scores higher
    ("Inspection and Test Plan.pdf", "itp"),
    ("Inspection Request.pdf", "inspection_request"),
    # was the first kind listed; a tie on the name is now 'document'
    ("Site photo model.pdf", "document"),
])
def test_ingest_types_new_names_by_the_same_rule(name, expected):
    from app.core.projects import classify_doc_type
    assert classify_doc_type(name) == expected


# ── Arabic keywords match WHOLE words only (#398 grader, 17 Nov) ─────────────
# A keyword glued inside a longer word is a different word: it must not type
# the document. The pairs below are built from the table itself, so every
# kind with Arabic words is covered and a new word is covered when it lands.

_GLUE_BEFORE, _GLUE_AFTER = "م", "ت"   # letters that are not a listed prefix


def _is_arabic_kw(kw: str) -> bool:
    return any("ء" <= ch <= "ي" for ch in kw)


def _glued(phrase: str) -> str:
    """Every word of ``phrase`` inside a longer word."""
    return " ".join(_GLUE_BEFORE + w + _GLUE_AFTER for w in phrase.split())


def _arabic_words(field: str, minimum: int):
    out = []
    for t in doc_types._load_builtins():
        kws = [k for k in ((t.get("match") or {}).get(field) or []) if _is_arabic_kw(k)]
        if len(kws) >= minimum:
            out.append((t["name"], kws))
    return out


_NAME_KINDS = _arabic_words("filename", 1)
_CONTENT_KINDS = _arabic_words("content", 2)


@pytest.mark.parametrize("kinds, every_kind", [
    (_NAME_KINDS, True),      # every kind has an Arabic name word
    (_CONTENT_KINDS, False),  # some kind has two Arabic content words
], ids=["filename", "content"])
def test_every_kind_has_arabic_words_to_test(kinds, every_kind):
    assert kinds
    if every_kind:
        assert len(kinds) == len(doc_types._load_builtins())


@pytest.mark.parametrize("table, expected", [
    # no types table on disk: no prefix, nothing raised
    (None, []),
    # the table's prefixes, folded, longest first
    ("arabic_prefixes: ['ال', 'وال', ' ']\n", ["وال", "ال"]),
], ids=["no_table", "table"])
def test_arabic_prefixes_come_from_the_types_table(tmp_path, monkeypatch,
                                                   table, expected):
    path = tmp_path / "document_types.yaml"
    if table is not None:
        path.write_text(table, encoding="utf-8")
    monkeypatch.setattr(doc_types, "_builtin_file", lambda: str(path))
    assert doc_types._ar_prefixes() == expected


@pytest.mark.parametrize("keyword, expected", [
    # an empty keyword (a table entry that folds to nothing) matches nothing,
    # not every gap between two separators
    ("", False),
    ("عقد", True),
], ids=["empty_keyword", "whole_word"])
def test_an_empty_keyword_never_matches(keyword, expected):
    assert doc_types._has_word(keyword, "a, عقد", []) is expected


@pytest.mark.parametrize("kind, kws", _NAME_KINDS, ids=[k for k, _ in _NAME_KINDS])
@pytest.mark.parametrize("glue", [False, True], ids=["whole", "inside_longer_word"])
def test_an_arabic_name_word_types_only_as_a_whole_word(kind, kws, glue):
    for kw in kws:
        name = (_glued(kw) if glue else kw) + " 07.pdf"
        got = doc_types.classify(name)["name"]
        if glue:
            assert got == "unrecognised", (kw, got)
        else:
            # the bare word types its kind, or ties a kind that shares it
            assert got in (kind, "unrecognised"), (kw, got)
    if not glue:
        assert doc_types.classify(kws[0] + " 07.pdf")["name"] == kind


@pytest.mark.parametrize("kind, kws", _CONTENT_KINDS, ids=[k for k, _ in _CONTENT_KINDS])
@pytest.mark.parametrize("glue", [False, True], ids=["whole", "inside_longer_word"])
def test_arabic_content_words_count_only_as_whole_words(kind, kws, glue):
    two = kws[:2]
    text = " . ".join(_glued(k) if glue else k for k in two)
    expected = "unrecognised" if glue else kind
    assert doc_types.classify("annex_04.pdf", text)["name"] == expected


@pytest.mark.parametrize("name, expected", [
    # "concept" contains the letters of "photos": not a photo
    ("تصور المشروع.pdf", "unrecognised"),
    ("صور المشروع.pdf", "photo"),
    # "complex" contains the letters of "contract": not a contract
    ("مشروع معقد.pdf", "unrecognised"),
    ("عقد المشروع.pdf", "contract"),
])
def test_a_keyword_inside_a_longer_arabic_word_does_not_type(name, expected):
    assert doc_types.classify(name)["name"] == expected


@pytest.mark.parametrize("name, expected", [
    # one listed prefix glued in front: "and a contract", "and the contract"
    ("وعقد المقاول.pdf", "contract"),
    ("والعقد المعدل.pdf", "contract"),
    # two separate clitics stacked ("and with"): not a listed prefix
    ("وبعقد المقاول.pdf", "unrecognised"),
])
def test_one_leading_arabic_prefix_is_allowed(name, expected):
    assert doc_types.classify(name)["name"] == expected


# ── common names that now tie two kinds and become untyped ───────────────────

@pytest.mark.parametrize("name, expected", [
    # a word of two kinds each, nothing else: a tie, so 'document'
    ("Inspection Report.pdf", "document"),
    ("Meeting Report.pdf", "document"),
    ("Drawing Schedule.xlsx", "document"),
    ("Method Statement Template.docx", "document"),
    ("Contract Programme.pdf", "document"),
    ("BOQ Contract.xlsx", "document"),
    # one more word for one kind breaks the tie by score
    ("Daily Inspection Report.pdf", "report"),
    # each name with one kind's word only types that kind
    ("Inspection 14.pdf", "inspection_request"),
    ("Meeting 14.pdf", "minutes"),
])
def test_common_names_that_tie_two_kinds_are_untyped(name, expected):
    from app.core.projects import classify_doc_type
    assert classify_doc_type(name) == expected


@pytest.mark.parametrize("content, expected", [
    # a Latin content phrase keeps its substring match ("milestone" in
    # "milestones"): the whole-word rule is for Arabic words
    ("Critical path and milestones.", "schedule"),
    ("Critical path only.", "unrecognised"),
])
def test_latin_content_phrases_still_match_inside_words(content, expected):
    assert doc_types.classify("annex_04.pdf", content)["name"] == expected


# ── the types file is read once per version, not once per classify call ──────

@pytest.mark.parametrize("edit, reads, expected", [
    # unchanged file: the second call reads nothing
    (None, 1, ["ال"]),
    # same size, new modification time: read again, the new list is seen
    ("mtime", 2, ["لل"]),
    # new size, modification time forced back to the old one: read again
    ("size", 2, ["وال"]),
], ids=["unchanged", "new_mtime", "new_size"])
def test_the_types_file_is_cached_per_version(tmp_path, monkeypatch,
                                              edit, reads, expected):
    import os
    path = tmp_path / "document_types.yaml"
    path.write_text("arabic_prefixes: ['ال']\ndocument_types: []\n", encoding="utf-8")
    st = os.stat(path)
    monkeypatch.setattr(doc_types, "_builtin_file", lambda: str(path))
    doc_types._read_table.cache_clear()
    doc_types._prefixes_for.cache_clear()
    real, calls = doc_types.yaml.safe_load, []
    monkeypatch.setattr(doc_types.yaml, "safe_load",
                        lambda f: calls.append(1) or real(f))

    assert doc_types._ar_prefixes() == ["ال"]
    assert doc_types._load_builtins() == []
    if edit == "mtime":
        path.write_text("arabic_prefixes: ['لل']\ndocument_types: []\n", encoding="utf-8")
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    elif edit == "size":
        path.write_text("arabic_prefixes: ['وال']\ndocument_types: []\n", encoding="utf-8")
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert doc_types._ar_prefixes() == expected
    assert doc_types._load_builtins() == []
    assert len(calls) == reads


# ── mixed Latin / Arabic names ───────────────────────────────────────────────

@pytest.mark.parametrize("name, expected", [
    # a Latin word, a space, an Arabic keyword: the Arabic word types it
    ("Tower عقد.pdf", "contract"),
    # the Latin letters glued to the Arabic keyword: one longer word, untyped
    ("Towerعقد.pdf", "unrecognised"),
    # a Latin keyword, a space, an Arabic word: the Latin word types it
    ("Inspection برج.pdf", "inspection_request"),
    # Arabic letters glued to the Latin keyword: one longer word, untyped
    ("Inspectionبرج.pdf", "unrecognised"),
], ids=["ar_kw_spaced", "ar_kw_glued", "latin_kw_spaced", "latin_kw_glued"])
def test_a_mixed_latin_arabic_name_types_by_whole_words(name, expected):
    assert doc_types.classify(name)["name"] == expected


# ── digits glued to a keyword ────────────────────────────────────────────────
# A digit is a word character (unchanged for Latin names since before the
# Arabic rule): a keyword with a digit glued on is a different token and does
# not type the name; a separator between them does. Arabic-Indic digits
# behave exactly like Latin digits.

@pytest.mark.parametrize("name, expected", [
    ("Inspection-07.pdf", "inspection_request"),
    ("Inspection07.pdf", "unrecognised"),
    ("07Inspection.pdf", "unrecognised"),
    ("عقد ٠٧.pdf", "contract"),
    ("عقد٠٧.pdf", "unrecognised"),
    ("٠٧عقد.pdf", "unrecognised"),
], ids=["latin_sep", "latin_glued_after", "latin_glued_before",
        "arabic_indic_sep", "arabic_indic_glued_after", "arabic_indic_glued_before"])
def test_digits_glued_to_a_keyword(name, expected):
    assert doc_types.classify(name)["name"] == expected
