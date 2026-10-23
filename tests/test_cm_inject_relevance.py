"""The cross-domain prompt inject must be relevant or absent.

`_cm_prompt_fragment_for_turn` puts this text into the system prompt of
`project-assistant` and `heavy-reasoning` — the two construction agents — as
assertive domain guidance. It is not a suggestion the model can weigh against
evidence; it reads as established fact about the project.

That makes a wrong linkage worse than no linkage, which is the whole point of
these tests. Before the relevance fix, `get_cross_domain_context` took
`get_rules_for_target(domain)[:3]` — the first three rules DECLARED for each
implicated domain. Declaration order is an artefact of the order somebody typed
DEFAULT_RULES, so a variation-order question was told, in its system prompt,
that "Critical safety incident triggers stop-work order in affected zone" was a
relevant construction management linkage.
"""
from __future__ import annotations

import pytest

from app.core.cross_domain_reasoner import (
    TEMPLATE_MATCH_FLOOR,
    CrossDomainIntentDetector,
    CrossDomainReasoner,
)
from app.core.dependency_graph import Domain


@pytest.fixture
def detector():
    return CrossDomainIntentDetector()


# ── the contamination this fix exists to stop ────────────────────────────

def test_a_variation_question_is_not_told_about_safety_stop_work(detector):
    """The exact regression. R003 (safety incident -> stop work) targets
    SCHEDULE, and SCHEDULE is implicated by a variation, so declaration-order
    selection surfaced it first. Nothing about a facade variation implies a
    stop-work order, and saying so in the system prompt invites the model to
    answer as though one were in play."""
    ctx = detector.get_cross_domain_context(
        "Client issued a variation for the podium facade - what is the impact?"
    )
    assert ctx, "a variation should still produce cross-domain context"
    assert "stop-work" not in ctx.lower()
    assert "safety incident" not in ctx.lower()


def test_a_variation_question_leads_with_the_variation_rule(detector):
    """R005 is literally 'Variation order updates budget'. It should not merely
    be present, it should be first — the inject is capped at five lines and at
    800 chars downstream, so ordering decides what survives."""
    ctx = detector.get_cross_domain_context(
        "Client issued a variation for the podium facade - what is the impact?"
    )
    lines = [ln.strip() for ln in ctx.splitlines() if ln.strip().startswith("1.")]
    assert lines, ctx
    assert "variation order" in lines[0].lower()


def test_every_selected_rule_connects_to_the_message(detector):
    """The general invariant behind the two cases above: a rule earns its place
    by firing from a domain the message is about or implicates. A rule that
    merely shares a target domain is not a linkage, it is a coincidence."""
    message = "Client issued a variation for the podium facade - what is the impact?"
    sources, targets = detector.detect_domains(message)
    detected = sources | targets
    for rule in detector._rank_rules(sources, targets):
        assert rule.source_domain in detected, (
            f"{rule.rule_id} was selected but fires from "
            f"{rule.source_domain.value}, which this message never raised"
        )


def test_silence_when_no_rule_is_genuinely_relevant(detector):
    """Cross-domain intent can be detected while no dependency rule actually
    applies. The old code returned a bare `f""` on that branch — a dead
    expression that made the intent look deliberate. Now it is deliberate:
    emitting nothing is correct, because the alternative is filler presented as
    project fact."""
    assert detector.get_cross_domain_context("hello how are you") == ""


def test_no_context_is_ever_emitted_without_linkages(detector):
    """Whatever the message, the header must never appear on its own — a bare
    'Relevant construction management linkages:' with nothing under it would
    read as 'we checked and found the linkages', not 'we found none'."""
    probes = [
        "hello",
        "what is the weather",
        "we are delayed",
        "lost time injury on level 3",
        "snagging list from the handover walk",
        "clash detected between duct and beam",
    ]
    for p in probes:
        ctx = detector.get_cross_domain_context(p)
        if "linkages" in ctx:
            body = ctx.split("linkages:", 1)[1].strip()
            assert body, f"header with no linkages for: {p!r}"


# ── the gate lives in the funnel ─────────────────────────────────────────

def test_analyze_turn_withholds_a_template_below_the_floor():
    """`matched_template` used to be whatever ranked first, with no threshold,
    while the documented floor lived in `best_match`'s default argument. The
    one consumer re-checked the score itself; a second consumer that trusted
    the field would have inherited the bug."""
    reasoner = CrossDomainReasoner()
    # "dashboard" is a monthly_progress trigger but ordinary business English,
    # so it scores below the floor on its own.
    result = reasoner.analyze_turn("dashboard")

    assert 0 < result["matched_template_score"] < TEMPLATE_MATCH_FLOOR, result
    assert result["matched_template"] is None, result
    # The score is still reported, so callers can log or tune against it.
    assert result["matched_template_score"] > 0


def test_analyze_turn_reports_a_template_above_the_floor():
    reasoner = CrossDomainReasoner()
    result = reasoner.analyze_turn("we need a delay claim and extension of time")
    assert result["matched_template"] == "delay_to_claim"
    assert result["matched_template_score"] >= TEMPLATE_MATCH_FLOOR


def test_the_score_never_reports_a_template_it_withheld():
    """Whatever the message, the two fields must agree: a named template
    implies a score at or above the floor."""
    reasoner = CrossDomainReasoner()
    for msg in ["extra work", "snagging list", "lost time injury",
                "hello", "variation order impact", "float on the critical path"]:
        r = reasoner.analyze_turn(msg)
        if r["matched_template"] is not None:
            assert r["matched_template_score"] >= TEMPLATE_MATCH_FLOOR, (msg, r)


# ── scoring must not punish vocabulary ───────────────────────────────────

def test_adding_synonyms_does_not_dilute_a_match():
    """Scoring was `hits / len(triggers)`, so every synonym added to a template
    lowered the score of every phrase already in it. That put the vocabulary
    work directly at odds with the floor: enriching a template could push its
    own real matches below TEMPLATE_MATCH_FLOOR.

    Pinned as a property — one specific phrase clears the floor regardless of
    how many alternatives the template happens to list."""
    from app.core.cross_domain_reasoner import _TEMPLATE_TRIGGERS, TemplateMatcher

    matcher = TemplateMatcher()
    for template_id, triggers in _TEMPLATE_TRIGGERS.items():
        multiword = [t for t in triggers if " " in t]
        if not multiword:
            continue
        phrase = multiword[0]
        scores = dict(matcher.match(phrase))
        assert scores.get(template_id, 0.0) >= TEMPLATE_MATCH_FLOOR, (
            f"{template_id!r} does not clear the floor on its own trigger "
            f"{phrase!r} — scoring is diluted by list length"
        )


# ── site language ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message,expected",
    [
        ("We had a lost-time injury on level 3 this morning.",
         "safety_incident_response"),
        ("The main contractor is claiming 45 days for the late foundation release.",
         "delay_to_claim"),
        ("Snagging list from the handover walk has 200 open items.",
         "qa_defect_closeout"),
        ("Client issued a variation for the podium facade.",
         "change_order_impact"),
    ],
)
def test_real_site_phrasing_reaches_the_right_template(message, expected):
    """Measured 2026-08-24: three of these matched NO template and the fourth
    matched the wrong one ("snagging ... handover" went to
    commissioning_to_handover because "handover" appeared in the sentence).

    These misses are silent — the turn simply gets no cross-domain help — so
    only a test makes them visible."""
    reasoner = CrossDomainReasoner()
    result = reasoner.analyze_turn(message)
    assert result["matched_template"] == expected, result


def test_a_procurement_question_suggests_the_procurement_tool():
    """Regression guard on the source/target split. Tools were drawn from the
    IMPLICATED domains only, and once procurement keywords carried
    PROCUREMENT as their source rather than their own target, a procurement
    question stopped suggesting the procurement tool."""
    reasoner = CrossDomainReasoner()
    result = reasoner.analyze_turn("we have a material shortage on rebar")
    assert "procurement_list_generator" in result["suggested_tools"], result


def test_source_domains_are_declared_for_every_keyword():
    """The derived back-compat view cannot drift from the routes table, but a
    keyword with a nonsense source can still be typed. Pin the shape."""
    from app.core.cross_domain_reasoner import _CROSS_DOMAIN_KEYWORDS, _KEYWORD_ROUTES

    assert set(_KEYWORD_ROUTES) == set(_CROSS_DOMAIN_KEYWORDS)
    for kw, (source, targets) in _KEYWORD_ROUTES.items():
        assert isinstance(source, Domain), kw
        assert targets, f"{kw} implicates nothing"
        assert all(isinstance(d, Domain) for d in targets), kw
        assert source not in targets, (
            f"{kw} lists its own source domain as implicated — that was the "
            f"workaround for having no source field"
        )


class TestWordBoundaryMatching:
    """Single-token keywords/triggers match words (with inflections), never
    substrings. Regression net for: 'ncr' in 'concrete', 'eot' in
    'geotechnical', 'late' in 'calculated', 'vo' in 'invoice', 'lti' in
    'multi' — the last two introduced by the site-language pass itself."""

    def test_invoice_does_not_publish_variation_template(self):
        from app.core.cross_domain_reasoner import CrossDomainReasoner
        r = CrossDomainReasoner()
        assert r.analyze_turn("please process this invoice for the consultant")[
            "matched_template"
        ] is None

    def test_multi_does_not_publish_safety_template(self):
        from app.core.cross_domain_reasoner import CrossDomainReasoner
        r = CrossDomainReasoner()
        assert r.analyze_turn("planning the multi-storey car park works")[
            "matched_template"
        ] is None

    def test_concrete_geotechnical_calculated_do_not_route_domains(self):
        from app.core.cross_domain_reasoner import CrossDomainIntentDetector
        from app.core.dependency_graph import Domain
        d = CrossDomainIntentDetector()
        assert Domain.QUALITY not in d.detect_domains("pour the level 3 concrete")[0]
        assert d.detect_domains("geotechnical report for the basement")[0] == set()
        assert Domain.SCHEDULE not in d.detect_domains("the calculated bearing capacity")[0]

    def test_inflections_and_real_terms_still_fire(self):
        from app.core.cross_domain_reasoner import (
            CrossDomainIntentDetector,
            CrossDomainReasoner,
        )
        from app.core.dependency_graph import Domain
        d = CrossDomainIntentDetector()
        r = CrossDomainReasoner()
        assert Domain.SCHEDULE in d.detect_domains("we are delayed by 2 weeks")[0]
        assert Domain.QUALITY in d.detect_domains("we received an ncr on the slab")[0]
        assert r.analyze_turn("raise a VO for the facade change")[
            "matched_template"
        ] == "change_order_impact"

    def test_inflected_forms_of_site_verbs_are_reachable(self):
        """Tokens are stored as STEMS so every inflection reaches them. The
        gap this closes: "de-snag" and "snag" matched only their exact
        spelling, because the suffix group did not cover the doubled
        consonant — so "de-snagging" and "snagged", the spellings people
        actually type, matched nothing."""
        from app.core.cross_domain_reasoner import _word_pattern

        for stem, forms in [
            ("snag", ["snag", "snags", "snagged", "snagging"]),
            ("de-snag", ["de-snag", "de-snagged", "de-snagging"]),
            ("slip", ["slip", "slips", "slipped", "slipping"]),
            ("delay", ["delay", "delays", "delayed", "delaying"]),
            ("claim", ["claim", "claims", "claimed", "claiming"]),
        ]:
            pattern = _word_pattern(stem)
            for form in forms:
                assert pattern.search(form), f"{stem!r} does not reach {form!r}"

    def test_doubling_never_reopens_the_substring_hole(self):
        """The doubled-consonant suffixes must not weaken word boundaries."""
        from app.core.cross_domain_reasoner import _word_pattern

        for token, containing in [
            ("vo", "invoice"), ("lti", "multi-storey"), ("ncr", "concrete"),
            ("eot", "geotechnical"), ("late", "calculated"),
            ("float", "floatation"), ("slip", "slipstream"),
        ]:
            assert not _word_pattern(token).search(containing), (
                f"{token!r} still matches inside {containing!r}"
            )

    def test_no_token_is_covered_by_another_tokens_inflections(self):
        """A word that matches two tokens contributes two units of evidence
        from one occurrence, which inflates the score toward saturation.

        "snagging" used to match both `snag` and `snagging`; "variations" both
        `variation` and `variations`. Storing stems and letting inflection do
        the work keeps one word worth one word.
        """
        from app.core.cross_domain_reasoner import (
            _KEYWORD_ROUTES,
            _TEMPLATE_TRIGGERS,
            _word_pattern,
        )

        tokens = sorted(
            {k for k in _KEYWORD_ROUTES if " " not in k}
            | {t for v in _TEMPLATE_TRIGGERS.values() for t in v if " " not in t}
        )
        overlaps = [
            (shorter, longer)
            for shorter in tokens
            for longer in tokens
            if shorter != longer
            and len(longer) > len(shorter)
            and _word_pattern(shorter).search(longer)
        ]
        assert not overlaps, (
            "these tokens are already reached by another token's inflections, "
            f"so they double-count: {overlaps}"
        )


# ── Miss telemetry (issue #419) ───────────────────────────────────────────
# The counter exists to answer one question: how often does a construction
# turn get no cross-domain help? These tests pin the parts that make that
# number trustworthy rather than merely present.


def _stats_after(messages):
    from app.core import cross_domain_reasoner as cdr

    cdr.reset_template_match_stats()
    reasoner = cdr.CrossDomainReasoner()
    for m in messages:
        reasoner.analyze_turn(m)
    return cdr.get_template_match_stats()


def test_conversational_turns_are_not_counted_as_misses():
    """A miss on "hello" is the layer declining correctly, not a failure.

    Counting it would bury the real signal: chat is mostly conversation, so
    every genuine vocabulary hole would sit under a pile of greetings and the
    miss rate would asymptote to "almost everything", telling nobody anything.
    """
    stats = _stats_after(["hello", "thanks", "how are you today", "ok"])
    assert stats["total"] == 0, stats
    assert stats["miss_rate"] == 0.0


def test_a_matched_construction_turn_counts_as_matched():
    stats = _stats_after(["we had a lost-time injury on level 3"])
    assert stats["matched"] == 1, stats
    assert stats["total"] == 1
    assert stats["miss_rate"] == 0.0


def test_construction_intent_with_no_template_is_counted_as_a_miss():
    """Intent detected, no template — precisely the silent case #419 is about."""
    from app.core import cross_domain_reasoner as cdr

    cdr.reset_template_match_stats()
    reasoner = cdr.CrossDomainReasoner()
    result = reasoner.analyze_turn("the procurement lead time is a worry")
    stats = cdr.get_template_match_stats()

    assert result["additional_domains"], "precondition: intent must be detected"
    assert result["matched_template"] is None, "precondition: no template above floor"
    assert stats["below_floor"] + stats["no_candidate"] == 1, stats
    assert stats["miss_rate"] == 1.0


def test_below_floor_and_no_candidate_are_distinguished():
    """A near-miss and a total blank need different fixes.

    below_floor says the vocabulary is close and the floor may be wrong;
    no_candidate says the phrasing is absent from the tables entirely. Folding
    them into one number would hide which of the two is actually happening.
    """
    from app.core import cross_domain_reasoner as cdr

    assert set(cdr._MATCH_OUTCOMES) == {"matched", "below_floor", "no_candidate"}
    stats = _stats_after(["hello"])
    assert "below_floor" in stats and "no_candidate" in stats


def test_miss_rate_is_the_share_of_construction_turns_left_unhelped():
    stats = _stats_after([
        "we had a lost-time injury on level 3",   # matches
        "snagging list has 200 open items",       # matches
        "the procurement lead time is a worry",   # miss
    ])
    assert stats["total"] == 3, stats
    assert stats["matched"] == 2, stats
    assert stats["miss_rate"] == round(1 / 3, 4), stats


def test_raw_user_messages_are_not_logged_by_default(monkeypatch, caplog):
    """Misses are customer text. Opt in before it reaches the default sink."""
    import logging

    from app.core import cross_domain_reasoner as cdr

    monkeypatch.delenv("CM_LOG_TEMPLATE_MISSES", raising=False)
    cdr.reset_template_match_stats()
    secret = "the procurement lead time on the Al Maktoum job is a worry"
    with caplog.at_level(logging.INFO, logger=cdr.__name__):
        cdr.CrossDomainReasoner().analyze_turn(secret)
    assert secret not in caplog.text

    caplog.clear()
    monkeypatch.setenv("CM_LOG_TEMPLATE_MISSES", "1")
    with caplog.at_level(logging.INFO, logger=cdr.__name__):
        cdr.CrossDomainReasoner().analyze_turn(secret)
    assert secret in caplog.text


def test_telemetry_failure_cannot_break_a_turn(monkeypatch):
    """Instrumentation must never be able to take down the thing it measures."""
    from app.core import cross_domain_reasoner as cdr

    monkeypatch.setattr(cdr, "_MATCH_STATS", None)  # force an internal error
    result = cdr.CrossDomainReasoner().analyze_turn("we had a lost-time injury on level 3")
    assert result["matched_template"] == "safety_incident_response"
