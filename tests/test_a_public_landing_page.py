"""A signed-out visitor sees a public front page (0735Z item 3g).

Read from the frontend sources, as other UI contract tests here do: the
routing rules are what matter, and the frontend's own build and lint run in
CI. A link from an email (?verified=, ?reset=) still lands on the sign-in page
with its query string; any other signed-out visit goes to /welcome.
"""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / "frontend" / "src"


def test_signed_out_visitors_go_to_the_front_page_and_email_links_to_sign_in():
    guard = (SRC / "auth" / "ProtectedRoute.tsx").read_text(encoding="utf-8")
    assert "<Navigate to=\"/welcome\" replace />" in guard
    assert "params.has('verified') || params.has('reset')" in guard
    assert "pathname: '/login', search: window.location.search" in guard


def test_the_front_page_is_routed_and_public():
    app = (SRC / "App.tsx").read_text(encoding="utf-8")
    route = re.search(r'<Route path="/welcome" element=\{<Landing />\} />', app)
    assert route, "the /welcome route must render Landing without a ProtectedRoute"


def test_the_front_page_claims_no_figures_or_names():
    page = (SRC / "pages" / "Landing.tsx").read_text(encoding="utf-8")
    copy = " ".join(re.findall(r"(?:title|body): '([^']+)'", page))
    assert copy, "capability copy not found"
    assert not re.search(r"\d", copy), "the front page must not claim figures"


def test_create_an_account_opens_the_register_form():
    login = (SRC / "pages" / "Login.tsx").read_text(encoding="utf-8")
    assert "get('mode') === 'register'" in login
    assert "/login?mode=register" in (SRC / "pages" / "Landing.tsx").read_text(encoding="utf-8")
