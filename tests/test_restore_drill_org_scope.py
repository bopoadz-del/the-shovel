"""Restore drill: locating the live endpoint when the key's account belongs to
an organisation.

Neon refuses an unscoped ``GET /projects`` (HTTP 400) for a key whose account
is an organisation member; the projects are listed per organisation instead.
Every fake API below is built here; ids are invented.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import restore_drill as rd


class OrgNeon:
    """A Neon API that lists projects only per organisation when ``orgs`` is
    given (the unscoped list is refused with 400), or unscoped when not."""

    def __init__(self, projects_by_org, *, org_scoped, unscoped_status=400):
        self.projects_by_org = projects_by_org
        self.org_scoped = org_scoped
        self.unscoped_status = unscoped_status
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path))
        route = urlsplit(path)
        q = parse_qs(route.query)
        if route.path == "/projects":
            if "org_id" in q:
                return {"projects": [{"id": p} for p in self.projects_by_org.get(q["org_id"][0], {})]}
            if self.org_scoped:
                raise rd.Refused(f"neon_http_{self.unscoped_status}:GET:/projects")
            return {"projects": [{"id": p} for o in self.projects_by_org.values() for p in o]}
        if route.path == "/users/me/organizations":
            orgs = list(self.projects_by_org) if self.org_scoped else []
            return {"organizations": [{"id": o} for o in orgs]}
        if route.path.startswith("/projects/") and route.path.endswith("/endpoints"):
            pid = route.path.split("/")[2]
            for o in self.projects_by_org.values():
                if pid in o:
                    return {"endpoints": o[pid]}
            return {"endpoints": []}
        raise AssertionError(f"unexpected call {method} {path}")


_LAYOUTS = [
    {"org-a": {"p-1": [{"id": "ep-aa-1", "branch_id": "br-1"}]}},
    {"org-a": {"p-1": []},
     "org-b": {"p-2": [{"id": "ep-other-9", "branch_id": "br-9"}],
               "p-3": [{"id": "ep-bb-3", "branch_id": "br-3"}]}},
]
_EXPECT = [("ep-aa-1", ("p-1", "br-1")), ("ep-bb-3", ("p-3", "br-3"))]


@pytest.mark.parametrize("layout,want", list(zip(_LAYOUTS, _EXPECT)))
def test_an_org_member_key_lists_projects_per_organisation(layout, want):
    endpoint, located = want
    neon = OrgNeon(layout, org_scoped=True)
    assert rd._locate(neon, endpoint) == located
    assert ("GET", "/users/me/organizations") in neon.calls
    assert all(m == "GET" for m, _ in neon.calls)


@pytest.mark.parametrize("layout,want", list(zip(_LAYOUTS, _EXPECT)))
def test_an_unscoped_key_never_asks_for_organisations(layout, want):
    endpoint, located = want
    neon = OrgNeon(layout, org_scoped=False)
    assert rd._locate(neon, endpoint) == located
    assert ("GET", "/users/me/organizations") not in neon.calls


@pytest.mark.parametrize("status", [401, 500])
def test_other_refusals_of_the_project_list_still_refuse(status):
    neon = OrgNeon(_LAYOUTS[0], org_scoped=True, unscoped_status=status)
    with pytest.raises(rd.Refused) as e:
        rd._locate(neon, "ep-aa-1")
    assert e.value.code == f"neon_http_{status}:GET:/projects"
    assert ("GET", "/users/me/organizations") not in neon.calls


@pytest.mark.parametrize("layout", [{}, {"org-a": {"p-1": [{"id": "ep-x", "branch_id": "br-x"}]}}])
def test_a_400_with_no_organisation_holding_the_endpoint_still_refuses(layout):
    neon = OrgNeon(layout, org_scoped=True)
    with pytest.raises(rd.Refused) as e:
        rd._locate(neon, "ep-aa-1")
    # no organisation at all: the original refusal stands; organisations but
    # no match: the endpoint is not found
    assert e.value.code == ("neon_http_400:GET:/projects" if not layout else "live_endpoint_not_found")
