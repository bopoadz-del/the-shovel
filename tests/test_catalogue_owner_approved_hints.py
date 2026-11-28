"""The owner's first approval of formula hints: the files of ten formulas say
``approved: true`` and nothing else in them changed.

* exactly those ten files are approved; every other shipped file is not;
* each approved hint reaches its formula's entry (``when_to_use`` /
  ``when_not``) through the catalogue the model reads (``catalogue.entries``
  and ``catalogue.lookup``), and is no longer a gap there;
* an unapproved file's written purpose reaches nothing;
* an approved file whose facts no longer agree with the code is not applied;
* the hint text of the ten is the text the owner read: each file's hint
  (everything but ``approved``) hashes to the value computed from main before
  the approval, so a changed character makes the file unapproved work again.

Every broken file here is a private copy built by the test.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil

import pytest

from app.agents.driver import catalogue
from app.agents.driver import catalogue_files as cf

#: id -> sha256 of the file's hint without ``approved`` (json, sorted keys),
#: computed from the files on main at c826083, the text the owner approved.
APPROVED_HINT_SHA256 = {
    "formula:rebar_weight": "347ed3e0891157f33f72ffefd4accc57ebf476d94ebd201d66149272ecd5fffc",
    "formula:rebar_by_area": "c4ad75bf74bfddc6908d2f3a05b2a544a11696105797e5c0435214997430cbeb",
    "formula:concrete_volume": "9b78eea37c1842d7b8f7e53bbe023204a3fae6556968f64d08cb319b39c5d5bf",
    "formula:excavation_volume": "a7b1313e280bb860090a334096e27fed547524635069b6f9293d1daa2ac3def7",
    "formula:backfill_volume": "8fbf52d91cb8644a9f25232ca98f99d419957e7da6d23c7f71a111a6cdc5774f",
    "formula:calculate_payment": "6de52fc3003fe5f2ea0878d7484cdeb8c535a838127477f69ae512b1a5c28396",
    "formula:calculate_interim_payment": "c62b38e52466eace75fe577c5871653197d7892e5edb37083e56fc8997287e38",
    "formula:interim_payment_to_date": "f4acadb2a33c14df428d79b547f2966cba82f2a8dcce19953c5db91aa80468f7",
    "formula:cpm_network": "6c11e2e07af0fee94d6f1ec9a8ae237dd6da4aa27b77d5229db8dfc6eafde159",
    "formula:critical_path_float": "0393ef16fe0bbfab28015bc20a1f4ae8e05e350b348ec239edb6563a5e1c750c",
}
APPROVED = sorted(APPROVED_HINT_SHA256)


def _hint_sha256(doc):
    hint = {k: v for k, v in doc["hints"].items() if k != "approved"}
    return hashlib.sha256(json.dumps(hint, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def shipped():
    return cf.load()


@pytest.fixture(scope="module")
def entries():
    return {e["id"]: e for e in catalogue.entries()}


@pytest.mark.parametrize("flag", [True, False])
def test_exactly_the_ten_files_are_approved(shipped, flag):
    having = sorted(i for i, doc in shipped.items() if doc["hints"]["approved"] is flag)
    wanted = APPROVED if flag else sorted(i for i in shipped if i not in APPROVED_HINT_SHA256)
    assert having == wanted
    assert len(having) == (10 if flag else len(shipped) - 10) and having


@pytest.mark.parametrize("unapprove", [None, "formula:concrete_volume"])
def test_the_files_agree_with_the_code(tmp_path, unapprove):
    # An approved file is applied only while its facts are the code's; the
    # approval flag is a hint, so taking it back moves no fact.
    root = tmp_path / "catalogue"
    shutil.copytree(cf.files_dir(), root)
    if unapprove:
        path = root / cf.path_of("formula", unapprove)
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["hints"]["approved"] = False
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert cf.load(str(root))[unapprove]["hints"]["approved"] is False
    assert cf.drift(str(root)) == []


@pytest.mark.parametrize("entry_id", APPROVED)
def test_the_hint_text_is_the_text_the_owner_read(shipped, entry_id):
    assert _hint_sha256(shipped[entry_id]) == APPROVED_HINT_SHA256[entry_id]


@pytest.mark.parametrize("entry_id, field", [
    ("formula:excavation_volume", "purpose"),
    ("formula:interim_payment_to_date", "difference"),
])
def test_one_changed_character_breaks_the_pin(shipped, entry_id, field):
    doc = copy.deepcopy(shipped[entry_id])
    if field == "purpose":
        doc["hints"]["purpose"] = doc["hints"]["purpose"][:-1] + "!"
    else:
        doc["hints"]["not_to_be_confused_with"][0]["difference"] += " "
    assert _hint_sha256(doc) != APPROVED_HINT_SHA256[entry_id]
    # The approval flag alone is outside the pin.
    doc = copy.deepcopy(shipped[entry_id])
    doc["hints"]["approved"] = not doc["hints"]["approved"]
    assert _hint_sha256(doc) == APPROVED_HINT_SHA256[entry_id]


@pytest.mark.parametrize("entry_id", APPROVED)
def test_each_approved_hint_reaches_its_entry(shipped, entries, entry_id):
    use, avoid = cf.hint_text(shipped[entry_id])
    entry = entries[entry_id]
    assert use and use.startswith(shipped[entry_id]["hints"]["purpose"].strip())
    assert entry["when_to_use"] == use
    # Each neighbour the owner read is named in when_not with its difference.
    for neighbour in shipped[entry_id]["hints"]["not_to_be_confused_with"]:
        assert f"{neighbour['id']} ({neighbour['difference']})" in entry["when_not"]
    assert entry["when_not"] == avoid
    assert "when_to_use" not in entry["gaps"] and "when_not" not in entry["gaps"]


@pytest.mark.parametrize("ids", [
    ["formula:excavation_volume", "formula:backfill_volume"],
    ["formula:calculate_interim_payment", "formula:interim_payment_to_date"],
])
def test_a_lookup_hands_the_model_the_approved_hints(shipped, ids):
    found = {e["id"]: e for e in catalogue.lookup(ids)["entries"]}
    assert sorted(found) == sorted(ids)
    for entry_id in ids:
        use, avoid = cf.hint_text(shipped[entry_id])
        assert found[entry_id]["when_to_use"] == use
        assert found[entry_id]["when_not"] == avoid


def _unapproved_with_a_purpose(shipped, count):
    picked = sorted(i for i, d in shipped.items()
                    if not d["hints"]["approved"] and d["hints"]["purpose"].strip())
    assert len(picked) >= count
    return picked[:count]


def test_an_unapproved_purpose_reaches_nothing(shipped, entries):
    for entry_id in _unapproved_with_a_purpose(shipped, 2):
        entry = entries[entry_id]
        purpose = shipped[entry_id]["hints"]["purpose"].strip()
        assert purpose not in (entry["when_to_use"] or "")
        assert purpose not in (entry["when_not"] or "")


@pytest.mark.parametrize("entry_id", ["formula:rebar_weight", "formula:cpm_network"])
def test_an_approved_file_whose_facts_changed_is_not_applied(shipped, entries, entry_id):
    files = copy.deepcopy(shipped)
    files[entry_id]["facts"]["name"] += " (changed in code)"
    bare = copy.deepcopy(entries[entry_id])
    bare["when_to_use"], bare["when_not"] = "", ""
    bare["gaps"] = sorted(set(bare["gaps"]) | {"when_to_use", "when_not"})
    (out,) = cf.enrich([bare], files)
    assert out["when_to_use"] == "" and out["when_not"] == ""
    assert {"when_to_use", "when_not"} <= set(out["gaps"])
    # The same entry with the file as shipped takes the hint.
    (out,) = cf.enrich([bare], shipped)
    assert out["when_to_use"] == cf.hint_text(shipped[entry_id])[0]
