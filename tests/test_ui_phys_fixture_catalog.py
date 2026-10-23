"""Sanitized UI-PHYS catalog stays complete and free of live client figures.

The confidential Drive pack must never land in git. This file is the CI
gate that the Playwright nightly can trust: same case IDs / assertion
shapes, fixture-only numbers, every S1–S6 source present on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ui_phys"
CATALOG = FIXTURE_DIR / "questions.json"

REQUIRED_IDS = (
    [f"A{i}" for i in range(1, 10)]
    + [f"B{i}" for i in range(1, 7)]
    + [f"C{i}" for i in range(1, 5)]
    + [f"D{i}" for i in range(1, 4)]
    + [f"E{i}" for i in range(1, 5)]
    + [f"F{i}" for i in range(1, 4)]
    + [f"G{i}" for i in range(1, 7)]
    + ["H1"]
)
NIGHTLY_IDS = ["A1", "B1", "E4", "G1", "G5", "H1"]


@pytest.fixture(scope="module")
def catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def test_catalog_declares_sanitized_variant(catalog):
    assert catalog["variant"] == "sanitized"
    assert catalog["do_not_commit_live_client_pack"] is True
    assert catalog["project_name"] == "UI-PHYS Fixture"


def test_catalog_has_the_real_set_case_ids(catalog):
    cases = catalog["cases"]
    missing = [cid for cid in REQUIRED_IDS if cid not in cases]
    assert missing == [], f"sanitized catalog is missing case ids: {missing}"


def test_nightly_subset_is_defined(catalog):
    assert catalog["nightly_ids"] == NIGHTLY_IDS
    for cid in NIGHTLY_IDS:
        assert cid in catalog["cases"]
        if cid != "H1":
            assert catalog["cases"][cid]["ask"]


def test_source_files_exist(catalog):
    for key, name in catalog["sources"].items():
        path = FIXTURE_DIR / name
        assert path.is_file(), f"{key} fixture missing: {path}"
        assert path.stat().st_size > 40


def _forbidden_terms(catalog):
    """Canary terms come from the UI_PHYS_FORBIDDEN env secret, never the repo.

    The canary list is itself confidential: committing live client strings
    to name what must not leak IS the leak (the #330 class). CI injects the
    secret; locally the check skips LOUDLY so an unconfigured run is never
    mistaken for a clean one.
    """
    import os

    env_terms = [
        t.strip()
        for t in os.getenv("UI_PHYS_FORBIDDEN", "").splitlines()
        if t.strip()
    ]
    return env_terms + list(catalog.get("forbidden_substrings") or [])


def test_no_live_client_figures_in_the_fixture_tree(catalog):
    import pytest

    terms = _forbidden_terms(catalog)
    if not terms:
        pytest.skip(
            "UI_PHYS_FORBIDDEN not configured — client-string leak check NOT RUN. "
            "Set the CI secret (newline-separated terms) to enforce."
        )
    blob = ""
    for path in FIXTURE_DIR.glob("*.md"):
        blob += path.read_text(encoding="utf-8")
    for case in catalog["cases"].values():
        blob += " " + str(case.get("ask") or "")
        blob += " " + " ".join(case.get("must") or [])
        blob += " " + " ".join(case.get("must_any") or [])
        blob += " " + " ".join(case.get("cite_any") or [])
    hits = [term for term in terms if term in blob]
    assert hits == [], f"live client strings leaked into the sanitized fixture: {hits}"


def test_llm_stub_covers_nightly_asks(catalog):
    """The Playwright stub must recognize every asked nightly prompt."""
    from tests.browser._llm_stub import _answer_for

    for cid in ("A1", "B1", "E4", "G1", "G5"):
        ask = catalog["cases"][cid]["ask"]
        out = _answer_for(ask)
        assert out, f"stub has no answer for {cid}"
        for token in catalog["cases"][cid].get("must") or []:
            assert token in out, f"{cid} stub missing {token!r}"
        if catalog["cases"][cid].get("must_any"):
            assert any(
                tok.lower() in out.lower()
                for tok in catalog["cases"][cid]["must_any"]
            ), f"{cid} stub missed must_any"


def test_nightly_cases_name_an_agent(catalog):
    """Per-role assertions: each asked nightly case pins a hat."""
    for cid in ("A1", "B1", "E4", "G1", "G5"):
        agent = catalog["cases"][cid].get("agent")
        assert agent, f"{cid} is missing an agent"
    assert catalog["cases"]["A1"]["agent"] == "project-assistant"
    assert catalog["cases"]["B1"]["agent"] == "quantity-surveyor"
    assert catalog["cases"]["E4"]["agent"] == "construction-pm"


# ── the two sources of record (owner's ruling 3, 2026-09-01) ──────────────
#
# GROUND_TRUTH_REVISIONS_2026-09-01.md is the standard: ground-truth changes
# ONLY via a dated revision log, question strings never altered, old wording
# preserved. The xlsx stays the question source; the log is the expectation
# source. README.md in the fixture directory is the long form.

README = FIXTURE_DIR / "README.md"

#: SHA-256 over every sorted "<id>\t<ask>" line in the catalog.
#:
#: A failure here means a QUESTION MOVED. Do not update this constant to make
#: it pass -- that is the exact thing it exists to prevent. Restore the
#: wording. A genuinely new question gets a NEW id, and only then does this
#: digest change, in the same commit that adds it and with the id named in
#: the message.
#:
#: Frozen because F-PHRASE-1 measured the routing to be phrasing-sensitive
#: and the sheet's phrasing to be the one that fails: E4 and C2 both PASS on
#: a conversational paraphrase and FAIL on the instrument's own wording. A
#: paraphrase is not evidence about the battery question.
QUESTION_WORDING_DIGEST = (
    "d5ad82ce90f107cf322fc19d7c451cd3c426ade24071e5e454801a319d31fe4a"
)


def _wording_digest(cases: dict) -> str:
    import hashlib

    pairs = sorted((cid, case.get("ask", "")) for cid, case in cases.items())
    blob = "\n".join(f"{cid}\t{ask}" for cid, ask in pairs)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_no_question_string_has_been_reworded(catalog):
    """Mutation killed: editing any ``ask`` in place.

    Including the well-meant edits -- fixing a typo, adding a question mark,
    expanding an abbreviation. Each one silently changes what the battery
    measures, and the verdict column cannot show it.
    """
    got = _wording_digest(catalog["cases"])
    assert got == QUESTION_WORDING_DIGEST, (
        "a battery question's wording changed (digest %s != %s). Restore the "
        "wording; a new question needs a NEW id." % (got, QUESTION_WORDING_DIGEST)
    )


def test_every_case_carries_a_question_or_names_an_action(catalog):
    """The digest alone would be satisfied by emptying every ask and pinning
    the new hash. Mutation killed: exactly that.

    H1 is the honest exception and is asserted as one rather than excused: an
    export case is driven by ``action`` (export_docx), not by a question. A
    case with NEITHER is a case that measures nothing.
    """
    blank = [
        cid for cid, case in catalog["cases"].items()
        if not case.get("ask", "").strip() and not case.get("action", "").strip()
    ]
    assert blank == [], f"cases that ask nothing and do nothing: {blank}"

    asks = {cid for cid, c in catalog["cases"].items() if c.get("ask", "").strip()}
    actions = {cid for cid, c in catalog["cases"].items() if c.get("action", "").strip()}
    assert actions == {"H1"}, f"action-driven cases changed: {sorted(actions)}"
    assert len(asks) == len(catalog["cases"]) - 1


def test_the_catalog_names_both_sources_of_record(catalog):
    """Mutation killed: dropping the pointers, which is how the two sources
    get confused again six weeks from now."""
    assert "UI-PHYS_QP2_results.xlsx" in catalog["question_source"]
    assert "ask exactly" in catalog["question_source"]
    assert catalog["expectation_source"].startswith("FLEET_OPS/artifacts/")
    assert "GROUND_TRUTH_REVISIONS_" in catalog["expectation_source"]
    assert "frozen" in catalog["wording_policy"].lower()


def test_the_readme_states_the_rule_rather_than_implying_it():
    """A convention nobody wrote down is a convention that gets broken. This
    asserts the load-bearing sentences exist, not the prose around them."""
    text = README.read_text(encoding="utf-8")
    assert "Question (ask exactly)" in text
    assert "GROUND_TRUTH_REVISIONS_2026-09-01.md" in text
    # The two rules, each in its own words.
    assert "Never reworded" in text
    assert "only by appending" in text
    # And the reason, so a reader can judge the rule rather than obey it.
    assert "F-PHRASE-1" in text
    assert "is not evidence about the" in text


def test_the_revision_log_named_by_the_catalog_is_the_one_the_readme_names(catalog):
    """Two pointers to the expectation source must not drift apart."""
    named = catalog["expectation_source"].rsplit("/", 1)[-1]
    assert named in README.read_text(encoding="utf-8")
