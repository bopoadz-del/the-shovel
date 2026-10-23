"""Contract tests for tests/feature_matrix_manifest.yaml and its runner.

The manifest is the authoritative TASK G test plan; these tests defend the
invariants the sweep runner relies on so a hand-edit cannot silently break
the sweep: unique actions, every named structure check implemented, the
must-cover feature set present with two phrasings each, and coverage rows
with at least one prompt. load_manifest() itself raises on the structural
defects (missing prompts, missing expect_route, unknown check names), so a
successful parse here already proves those.
"""
from __future__ import annotations

import pytest

from scripts.feature_matrix_sweep import (
    DEFAULT_MANIFEST,
    STRUCTURE_CHECKS,
    ManifestError,
    build_plan,
    load_manifest,
)

# The 13 features the pre-pilot sweep MUST exercise with two phrasings each
# (canonical + colloquial). document_metadata is also class:must in the
# manifest but is a supplementary row, not part of this contract.
MUST_COVER = {
    "process_document",
    "boq_process",
    "estimate_costs",
    "spec_analyze",
    "generate_wbs",
    "resource_histogram",
    "parse_primavera_schedule",
    "cash_flow_forecast",
    "procurement_list_generator",
    "rfp_management",
    "construction_advisor",
    "drawing_qto",
    "commissioning_checklist",
}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return load_manifest(DEFAULT_MANIFEST)


def test_manifest_parses(manifest):
    assert manifest["features"], "manifest has no features"
    assert manifest["defaults"].get("project"), "defaults.project missing"
    assert manifest["defaults"].get("agent"), "defaults.agent missing"


def test_actions_unique(manifest):
    # load_manifest raises on duplicates; assert directly anyway so the
    # invariant survives a refactor of the loader's validation.
    actions = [f["action"] for f in manifest["features"]]
    assert len(actions) == len(set(actions))


def test_every_structure_check_named_in_manifest_is_implemented(manifest):
    named = {c for f in manifest["features"] for c in f["structure"]}
    missing = named - set(STRUCTURE_CHECKS)
    assert not missing, f"structure checks without implementation: {missing}"


def test_must_cover_set_present_with_two_prompts_each(manifest):
    by_action = {f["action"]: f for f in manifest["features"]}
    missing = MUST_COVER - set(by_action)
    assert not missing, f"must-cover features missing from manifest: {missing}"
    for action in sorted(MUST_COVER):
        feat = by_action[action]
        assert feat["class"] == "must", f"{action} is not class: must"
        assert len(feat["prompts"]) >= 2, f"{action} needs >= 2 prompts"


def test_coverage_features_have_at_least_one_prompt(manifest):
    for feat in manifest["features"]:
        if feat["class"] == "coverage":
            assert feat["prompts"], f"{feat['action']} has no prompts"


def test_every_prompt_has_expected_route(manifest):
    for feat in manifest["features"]:
        for prompt in feat["prompts"]:
            assert prompt["expect_route"], \
                f"{feat['action']}: prompt without expect_route"
            assert prompt["text"].strip()


def test_defaults_are_applied_with_per_feature_overrides(manifest):
    by_action = {f["action"]: f for f in manifest["features"]}
    # boq_process overrides the default project — to the CANONICAL fixture
    # name, which the sweep resolves to the live seeded id at run time
    # (hardcoded ids died on every fixture reseed).
    assert by_action["boq_process"]["project"] == "FIXTURE — BOQ"
    # estimate_costs is fixture-bound too (BOQ workbook)
    assert by_action["estimate_costs"]["project"] == "FIXTURE — BOQ"
    # generate_wbs raises min_chars above the default
    assert by_action["generate_wbs"]["min_chars"] == 1500
    assert by_action["estimate_costs"]["min_chars"] == manifest["defaults"]["min_chars"]


def test_plan_indices_are_stable_and_contiguous(manifest):
    # --start resume semantics depend on a deterministic, gap-free plan.
    plan = build_plan(manifest, action=None, klass=None, runs_per_prompt=1)
    assert [p["plan_index"] for p in plan] == list(range(len(plan)))
    n_prompts = sum(len(f["prompts"]) for f in manifest["features"])
    assert len(plan) == n_prompts


def test_unknown_structure_check_is_rejected_at_load(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "defaults: {project: p, min_chars: 10, agent: a}\n"
        "features:\n"
        "  - action: x\n"
        "    prompts:\n"
        "      - {text: hi, expect_route: [x]}\n"
        "    structure: [definitely_not_a_check]\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="definitely_not_a_check"):
        load_manifest(bad)
