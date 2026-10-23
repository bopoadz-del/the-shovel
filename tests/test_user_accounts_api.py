"""End-to-end tests for register/login/me — Stream A."""
import os
import importlib
import pytest
from fastapi.testclient import TestClient
from tests._dev_key import DEV_KEY
from tests._fixture_values import PASSWORD, WRONG_PASSWORD


@pytest.fixture(scope="module")
def tmp_data_dir(tmp_path_factory):
    """Module-scoped temporary DATA_DIR so all tests share one DB."""
    return str(tmp_path_factory.mktemp("user_api_data"))


@pytest.fixture(scope="module", autouse=True)
def isolate_data_dir(tmp_data_dir):
    """Set DATA_DIR and reset module state before the module-scoped client."""
    prev_data_dir = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = tmp_data_dir
    # Reset module state so both modules use the tmp path
    import app.core.users as u_mod
    import app.core.jwt_auth as jwt_mod
    u_mod._initialized = False
    jwt_mod._cached_secret = None
    yield
    # Teardown: reset state. Restore the PREVIOUS value — popping DATA_DIR
    # dropped the suite-wide temp dir conftest.py installs, so every later
    # DB access fell back to ./data (the live-data leak conftest exists to
    # prevent) and test_validation_pipeline's ASGI test found no users
    # table there.
    u_mod._initialized = False
    jwt_mod._cached_secret = None
    if prev_data_dir is None:
        os.environ.pop("DATA_DIR", None)
    else:
        os.environ["DATA_DIR"] = prev_data_dir


@pytest.fixture(scope="module")
def client(tmp_data_dir):
    from app.main import app
    with TestClient(app) as c:
        yield c


def _register(client, email, pw=PASSWORD, name="Test"):
    return client.post("/v1/users/register",
                        json={"email": email, "password": pw, "display_name": name})


def test_register_returns_user_no_secrets(client):
    r = _register(client, "reg1@example.com")
    assert r.status_code == 201, r.text
    body = r.json()
    # The one neutral reply (#978): the address as typed, nothing else about
    # the account -- no secrets, no role, no id.
    assert set(body) == {"email", "registration_received", "message"}
    assert body["email"] == "reg1@example.com"


def test_register_duplicate_email_answers_like_a_new_one(client):
    first = _register(client, "dup-api@example.com")
    r = _register(client, "dup-api@example.com")
    assert r.status_code == 201 and r.json() == first.json()


def test_login_returns_jwt(client):
    _register(client, "login1@example.com", pw=PASSWORD)
    r = client.post("/v1/users/login",
                     json={"email": "login1@example.com", "password": PASSWORD})
    assert r.status_code == 200, r.text
    assert "token" in r.json() and r.json()["token"]


def test_login_wrong_password_401(client):
    _register(client, "login2@example.com", pw=PASSWORD)
    r = client.post("/v1/users/login",
                     json={"email": "login2@example.com", "password": WRONG_PASSWORD})
    assert r.status_code == 401


def test_me_with_jwt(client):
    _register(client, "me1@example.com", pw=PASSWORD)
    token = client.post("/v1/users/login",
                         json={"email": "me1@example.com",
                               "password": PASSWORD}).json()["token"]
    r = client.get("/v1/users/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == "me1@example.com"


def test_me_with_legacy_api_key_is_system_user(client):
    r = client.get("/v1/users/me", headers={"Authorization": "Bearer " + DEV_KEY})
    assert r.status_code == 200
    assert r.json()["user_id"] == "system"
