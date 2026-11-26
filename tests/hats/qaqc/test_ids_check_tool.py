"""The qaqc ``ids_check`` tool: a requirement table applied to an IFC model's
extracted elements (S-K3 item 1 part 2).

Every test builds its own extractor result and its own specification; no
element, property set or value comes from a project.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents.core.tool_registry import ToolCall
from app.agents.hats.qaqc import tools as qaqc_tools
from app.agents.hats.qaqc.ids_check import IdsSpecError
from app.agents.hats.qaqc.tools import handle_ids_check, ids_result_from_extraction


def _element(eid, rating):
    props = {} if rating is None else {"Pset_Test.Rating": rating}
    return {"id": eid, "ifc_type": "IfcDoor", "name": eid, "properties": props}


def _spec(prop="Pset_Test.Rating", value=None):
    req = {"property": prop}
    if value is not None:
        req["value"] = value
    return [{
        "name": "rating stated",
        "applicability": [{"entity": "IfcDoor"}],
        "requirements": [req],
    }]


def _extracted(elements, *, total=None, failed=()):
    return {
        "status": "success",
        "element_count": len(elements) if total is None else total,
        "building_elements": list(elements),
        "psets_failed": list(failed),
    }


# ── ids_result_from_extraction ──────────────────────────────────────────

@pytest.mark.parametrize("rating, status", [("60", "pass"), (None, "fail")])
def test_each_element_is_checked_against_the_table(rating, status):
    out = ids_result_from_extraction(_extracted([_element("D1", rating)]), _spec())
    assert out["specifications"][0]["status"] == status


@pytest.mark.parametrize("failed, status, not_checked", [
    ([{"global_id": "D1", "ifc_type": "IfcDoor", "error": "x"}], "incomplete", ["D1"]),
    ([{"global_id": "D9", "ifc_type": "IfcDoor", "error": "x"}], "fail", []),
    (["D1"], "fail", []),
    ([{"ifc_type": "IfcDoor"}], "fail", []),
])
def test_an_element_whose_psets_failed_is_not_checked(failed, status, not_checked):
    out = ids_result_from_extraction(
        _extracted([_element("D1", None)], failed=failed), _spec(),
    )
    spec = out["specifications"][0]
    assert spec["status"] == status
    assert spec["not_checked"] == not_checked


@pytest.mark.parametrize("total, status, not_read", [
    (1, "pass", None),
    (2, "incomplete", 1),
    (5, "incomplete", 4),
    ("7", "pass", None),
    (None, "pass", None),
])
def test_a_capped_extraction_never_reports_a_pass(total, status, not_read):
    extracted = _extracted([_element("D1", "60")])
    extracted["element_count"] = total
    out = ids_result_from_extraction(extracted, _spec())
    assert out["specifications"][0]["status"] == status
    assert out.get("elements_not_read") == not_read
    assert out["summary"]["passing"] == (1 if status == "pass" else 0)
    assert out["summary"]["incomplete"] == (1 if status == "incomplete" else 0)
    if not_read:
        assert out["notes"] == [
            f"{not_read} of {total} model elements were beyond the extraction "
            "cap and were not checked; no specification is reported as passing."
        ]
    else:
        assert "notes" not in out


@pytest.mark.parametrize("total, failing", [(1, 1), (3, 1)])
def test_a_capped_extraction_keeps_a_found_fault(total, failing):
    out = ids_result_from_extraction(
        _extracted([_element("D1", None)], total=total), _spec(),
    )
    assert out["specifications"][0]["status"] == "fail"
    assert out["summary"]["failing"] == failing


@pytest.mark.parametrize("specs", [[], None])
def test_an_empty_table_is_refused_not_filled(specs):
    with pytest.raises(IdsSpecError):
        ids_result_from_extraction(_extracted([_element("D1", "60")]), specs)


@pytest.mark.parametrize("extracted, status", [("not a dict", "pass"), (None, "pass")])
def test_an_unreadable_extraction_checks_no_element(extracted, status):
    out = ids_result_from_extraction(extracted, [dict(_spec()[0], cardinality="optional")])
    assert out["specifications"][0]["status"] == status
    assert out["summary"]["elements"] == 0


# ── the tool handler ────────────────────────────────────────────────────

class _Extractor:
    def __init__(self, result=None, raises=None):
        self.result = result
        self.raises = raises
        self.seen = None

    async def process(self, input_data, params):
        self.seen = (input_data, params)
        if self.raises:
            raise self.raises
        return self.result


def _call(args, blocks=("bim_extractor",)):
    return ToolCall(
        agent=SimpleNamespace(allowed_blocks=list(blocks)),
        name="ids_check", args=args, project_id="p-test",
    )


@pytest.fixture
def wire(monkeypatch):
    def _wire(extractor):
        monkeypatch.setattr("app.dependencies.get_block_instance", lambda _n: extractor)
        monkeypatch.setattr(
            "app.agents.runtime._resolve_file_path",
            lambda _pid, raw: f"/stored/{raw}",
        )
        return extractor
    return _wire


@pytest.mark.asyncio
@pytest.mark.parametrize("args, ok", [
    ({"ifc_file": "Model_A.ifc"}, True),
    ({"file_path": "model_b.IFC"}, True),
    ({"ifc_file": "Model_A.pdf"}, False),
    ({}, False),
])
async def test_only_an_ifc_model_is_read(wire, args, ok):
    extractor = wire(_Extractor(_extracted([_element("D1", "60")])))
    out = await handle_ids_check(_call(dict(args, specifications=_spec())))
    assert out["ok"] is ok
    if ok:
        name = args.get("ifc_file") or args.get("file_path")
        assert extractor.seen == ({"file_path": f"/stored/{name}"}, {"run_clash_detection": False})
        assert out["result"]["file"] == name
    else:
        assert extractor.seen is None
        assert out["result"]["error"] == "name the project's .ifc model to check"


@pytest.mark.asyncio
@pytest.mark.parametrize("blocks, ok", [(("bim_extractor",), True), (("construction",), False)])
async def test_the_agent_must_hold_the_extractor(wire, blocks, ok):
    wire(_Extractor(_extracted([_element("D1", "60")])))
    out = await handle_ids_check(_call({"ifc_file": "m.ifc", "specifications": _spec()}, blocks))
    assert out["ok"] is ok
    if not ok:
        assert out["result"]["error"] == "bim_extractor not in agent's allowed_blocks"


@pytest.mark.asyncio
@pytest.mark.parametrize("extractor, error", [
    (_Extractor(raises=RuntimeError("boom")), "bim_extractor failed: boom"),
    (_Extractor({"status": "error", "error": "IFC open error: bad"}), "IFC open error: bad"),
    (_Extractor({"status": "error"}), "bim_extractor failed"),
    (_Extractor("garbage"), "bim_extractor failed"),
])
async def test_an_extractor_failure_is_reported_not_checked(wire, extractor, error):
    wire(extractor)
    out = await handle_ids_check(_call({"ifc_file": "m.ifc", "specifications": _spec()}))
    assert out["ok"] is False
    assert out["result"] == {"status": "error", "error": error}


@pytest.mark.asyncio
@pytest.mark.parametrize("specs, ok", [(_spec(), True), ("not a table", False)])
async def test_an_invalid_table_is_an_error_payload(wire, specs, ok):
    wire(_Extractor(_extracted([_element("D1", "60")])))
    out = await handle_ids_check(_call({"ifc_file": "m.ifc", "specifications": specs}))
    assert out["ok"] is ok
    if ok:
        assert out["result"]["specifications"][0]["status"] == "pass"
    else:
        assert out["result"]["status"] == "error"
        assert out["result"]["error"]


# ── the tool is offered only where it can run ───────────────────────────

@pytest.mark.parametrize("project_id, blocks, offered", [
    ("p-test", ["bim_extractor"], True),
    (None, ["bim_extractor"], False),
    ("p-test", ["construction"], False),
])
def test_the_tool_is_offered_where_it_can_run(project_id, blocks, offered):
    from app.agents.core.tools import tool_definitions

    agent = SimpleNamespace(allowed_blocks=[], can_delegate=False)
    agent.allowed_blocks = []  # no block schemas: only the synthetic tools
    names_without = {t["function"]["name"] for t in tool_definitions(agent, project_id)}
    agent.allowed_blocks = blocks
    names = {t["function"]["name"] for t in tool_definitions(agent, project_id)}
    assert ("ids_check" in names) is offered
    assert "ids_check" not in names_without
    assert qaqc_tools.IDS_CHECK_SCHEMA["function"]["name"] == "ids_check"
