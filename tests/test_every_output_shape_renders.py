"""Every calculator output renders, whatever its shape; a default is disclosed.

Order 0435Z / LANE2 root causes 1-2 (CORE electrical stages dropped, tender
ranking missing): the renderer printed plain values only, so a mapping
output (stages -> days) and a list of records (ranked tenderers) produced
nothing. One renderer for every formula: a mapping as labelled rows, a list
of records as one table with the same columns on every row, and any optional
input the run took from its default named as assumed.
"""
from app.agents.driver import render
from app.lib import construction_formulas as cf


def _env(result, calculation="electrical_installation_sequence"):
    return {"status": "success", "calculation": calculation, "result": result}


def test_a_mapping_output_renders_as_labelled_rows():
    text = render.render_calculation(_env({"stages": {"first_fix": 40, "testing": 20}, "total_days": 60}))
    assert "- First fix: 40" in text and "- Testing: 20" in text


def test_a_list_of_records_renders_as_one_table_with_the_same_columns():
    rows = [{"bidder": "Bidder A", "rank": 2, "score": 36.9}, {"bidder": "Bidder B", "rank": 1}]
    text = render.render_calculation(_env({"ranked": rows}, calculation="evaluate_tender"))
    table = [line for line in text.splitlines() if line.startswith("|")]
    assert table[0] == "| Bidder | Rank | Score |"
    assert all(line.count("|") == table[0].count("|") for line in table)
    assert "| Bidder B | 1 |  |" in table


def test_an_input_taken_from_its_default_is_disclosed():
    env = cf.run_calculation("electrical_installation_sequence", {"floor_area_m2": 2000})
    rows = env["result"]["defaults_used"]
    assert rows and all(set(row) >= {"parameter", "value", "label"} for row in rows)
    assert "Assumed, not given:" in render.render_calculation(env, {"floor_area_m2": 2000})


def test_an_input_is_called_a_default_only_when_it_was_not_given():
    """A pair, so neither half can pass by the field simply being absent."""
    missing = cf.run_calculation("electrical_installation_sequence", {"floor_area_m2": 2000})
    given = cf.run_calculation("electrical_installation_sequence", {"floor_area_m2": 2000, "num_floors": 3})
    assert "num_floors" in {row["parameter"] for row in missing["result"]["defaults_used"]}
    assert "num_floors" not in {row["parameter"] for row in given["result"].get("defaults_used", [])}


def test_a_capped_list_says_how_many_rows_it_left_out():
    rows = [{"bidder": f"Bidder {i}", "rank": i} for i in range(1, 13)]
    text = render.render_calculation(_env({"ranked": rows}, calculation="evaluate_tender"))
    assert "- and 4 more" in text


def test_only_a_declared_input_is_disclosed_as_assumed():
    env = _env({"total_days": 60, "defaults_used": [
        {"parameter": "num_floors", "value": 1, "label": "number of floors"},
        {"parameter": "_internal_switch", "value": 2, "label": "internal switch"}]})
    text = render.render_calculation(env)
    assert "number of floors 1" in text and "internal switch" not in text


def test_an_answer_carries_a_mapping_result_only_with_its_figures():
    env = _env({"stages": {"first_fix": 40, "testing": 20}})
    assert render.carried(env, {}, "First fix takes 40 days, testing 20 days.")
    assert not render.carried(env, {}, "First fix takes 40 days.")


def test_a_mapping_result_is_carried_only_with_its_row_labels():
    """LANE2 (#986 repair 2): an answer with the right figures under invented
    row names -- the electrical stage table with 5 fake stages -- does not
    carry the result."""
    env = _env({"stages": {"first_fix": 40, "testing": 20}})
    assert render.carried(env, {}, "First fix: 40 days. Testing: 20 days.")
    assert not render.carried(env, {}, "Stage A: 40 days. Stage B: 20 days.")


def test_a_record_dropped_from_the_answer_fails_carried():
    rows = [{"bidder": "Bidder A", "score": 81.5}, {"bidder": "Bidder B", "score": 77.0},
            {"bidder": "Bidder C", "score": 70.25}]
    env = _env({"ranked": rows}, calculation="evaluate_tender")
    every = "Bidder A 81.5; Bidder B 77.0; Bidder C 70.25."
    assert render.carried(env, {}, every)
    assert not render.carried(env, {}, "Bidder A 81.5; Bidder B 77.0.")


def test_carried_reads_the_same_capped_rows_the_renderer_shows():
    """A mapping past the cap: the answer must carry the rows the renderer
    shows (and the renderer says how many it left out), not the hidden ones."""
    stages = {f"stage_{i}": 10 + i for i in range(1, 13)}
    env = _env({"stages": stages})
    text = render.render_calculation(env)
    assert "- and 4 more" in text
    assert render.carried(env, {}, text)
    shown_only = "; ".join(f"Stage {i}: {10 + i}" for i in range(1, 9))
    assert render.carried(env, {}, shown_only)
    assert not render.carried(env, {}, "; ".join(f"Stage {i}: {10 + i}" for i in range(1, 8)))
