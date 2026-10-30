"""K-T5 repair: every guard branch G6 listed (H1-H5) has a killing test.
Each test has two inputs; the inputs differ only in the branch's trigger."""
from __future__ import annotations

import ast
import importlib.util

import pytest

import app.lib.construction_formulas_commercial as c
import app.lib.construction_formulas_quantities as q

ACA = "of the Accepted Contract Amount per day"


# H1 -- :128 the Contract-Price exclusion, and :110 the parenthesised form.
@pytest.mark.parametrize("ctx,expected", [
    (f"Delay damages are 0.015% {ACA}.", True),
    (f"Delay damages are 0.015% {ACA}, capped by the Contract Price.", False),
])
def test_pct_of_aca_is_a_lookalike_unless_the_contract_price_is_named(ctx, expected):
    assert c.delay_damages_rate_is_coc_lookalike(0.015, ctx) is expected


@pytest.mark.parametrize("ctx", [
    f"Delay damages are 0.015% {ACA}.",
    f"Delay damages are 0.015% (zero point zero one five percent) {ACA}.",
])
def test_pct_of_aca_with_or_without_words_in_brackets(ctx):
    assert c.delay_damages_rate_is_coc_lookalike(0.015, ctx) is True


# H2 -- _sentence_of_amount: sentence end (:196), blank-line end (:178), no-match fallback (:197).
@pytest.mark.parametrize("sep", [". ", "\n\n"])
def test_a_cue_after_the_figures_sentence_does_not_stain_it(sep):
    ctx = f"The Accepted Contract Amount is SAR 12,345,678.00{sep}For example, SAR 1,000.00 per day"
    assert c._sentence_of_amount(12345678.00, ctx).strip() == "The Accepted Contract Amount is SAR 12,345,678.00"
    assert c.aca_amount_is_toy_example(12345678.00, ctx) is False


# :178 a '.' or ';' ends a sentence only before whitespace: a clause number
# ("14.1") or abbreviation ("e.g.x") inside the figure's sentence is no boundary.
@pytest.mark.parametrize("lead", ["Per Sub-Clause 14.1 the", "Per item e.g.x the"])
def test_a_dot_inside_a_word_is_not_a_sentence_end(lead):
    ctx = f"{lead} Accepted Contract Amount is SAR 12,345,678.00; For example, SAR 1,000.00 per day"
    assert c._sentence_of_amount(12345678.00, ctx) == f"{lead} Accepted Contract Amount is SAR 12,345,678.00"


@pytest.mark.parametrize("ctx,expected", [
    ("For example, the Accepted Contract Amount might be large.", True),
    ("The Accepted Contract Amount is stated in the Letter of Acceptance.", False),
])
def test_an_amount_not_in_the_text_is_judged_on_the_whole_text(ctx, expected):
    assert c._sentence_of_amount(5_000_000.00, ctx) == ctx
    assert c.aca_amount_is_toy_example(5_000_000.00, ctx) is expected


# H3 -- :99 the clause number is any n.n.n, not only 1.1.1.
@pytest.mark.parametrize("clause", ["1.1.1", "4.1.2"])
def test_any_numbered_aca_definition_row_is_the_clause_row(clause):
    text = f"{clause}: | | Accepted Contract Amount: SAR 12,345,678.00 |"
    assert c._figure_is_clause_111(text, 12345678.00) is True


@pytest.mark.parametrize("text,expected", [
    ("7.3: | | Accepted Contract Amount: SAR 12,345,678.00 |", True),
    ("Accepted Contract Amount: SAR 12,345,678.00 |", False),
])
def test_an_unnumbered_row_is_not_the_clause_row(text, expected):
    assert c._figure_is_clause_111(text, 12345678.00) is expected


# H4 -- quantities.py:339 documented waste asked, nothing stated, requested empty.
@pytest.mark.parametrize("text,has_key", [
    ("Concrete volume for a raft 30x20x1.5 m including your documented waste factor.", False),
    ("Concrete volume for a raft 30x20x1.5 m.", True),
])
def test_documented_waste_with_no_figure_is_a_missing_input(text, has_key):
    _calc, out = q.resolve_concrete_volume_calc(
        "concrete_volume", {"length_m": 30, "width_m": 20, "depth_m": 1.5, "waste_factor": 0}, text,
    )
    assert ("waste_factor" in out) is has_key


# H5 -- scan_hardwiring.py :217 *_RATE_PERCENT name and :232 the bool exclusion.
def _scan():
    spec = importlib.util.spec_from_file_location("scan_hw_kt5", "scripts/scan_hardwiring.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("src,flagged", [
    ("_DELAY_RATE_PERCENT = 0.015\n", ["_DELAY_RATE_PERCENT"]),
    ("_DELAY_RATE_NOTE = 0.015\n", []),
])
def test_a_rate_percent_constant_is_a_project_literal(src, flagged):
    found = _scan()._formula_literal_findings("m.py", ast.parse(src))
    assert [f.rsplit("::", 1)[1] for f in found] == flagged


@pytest.mark.parametrize("src,flagged", [
    ("DOCUMENTED_WASTE = 0.05\n", ["DOCUMENTED_WASTE"]),
    ("DOCUMENTED_WASTE = True\n", []),
])
def test_a_bool_flag_is_not_a_project_literal(src, flagged):
    found = _scan()._formula_literal_findings("m.py", ast.parse(src))
    assert [f.rsplit("::", 1)[1] for f in found] == flagged


# CI-M survivors on this PR: documented_concrete_waste_factor constant_return,
# drop_whole_of_works_delay_claims drop_rule.
@pytest.mark.parametrize("text,expected", [
    ("Concrete including 7% waste.", 0.07),
    ("Concrete including the documented waste.", None),
])
def test_documented_waste_factor_reads_the_stated_figure(text, expected):
    got = q.documented_concrete_waste_factor(text)
    assert got == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize("text", [
    "Rate is 0.1% per day.\nCap is 10%.",
    "Damages accrue daily.\n\nSee the Contract Data.",
])
def test_text_without_a_whole_works_claim_is_returned_untouched(text):
    assert c.drop_whole_of_works_delay_claims(text) == text


@pytest.mark.parametrize("ctx", [
    "Contract Data, Sub-Clause 8.7: 0.015% of the Contract Price per day",
    "Particulars 8.7 delay damages: 0.1% of the Contract Price",
])
def test_a_contract_data_row_on_the_contract_price_is_not_a_lookalike(ctx):
    assert c.delay_damages_rate_is_coc_lookalike(0.015, ctx) is False


@pytest.mark.parametrize("text", [None, ""])
def test_no_text_has_no_accepted_contract_amount(text):
    assert c.parse_accepted_contract_amount(text) is None


@pytest.mark.parametrize("amount", ["SAR 500.00", "USD 999.99"])
def test_a_sub_thousand_clause_111_row_is_not_the_base(amount):
    assert c.text_states_clause_111_aca(f"1.1.1: | | Accepted Contract Amount: {amount} |") is False
