"""BUILD 3 exam: the on-premise profile, part 1 (profile switch + boot check).

Every decision is proven with one setting that must pass and one that must
fail. Each environment below is built by the test itself; nothing is read
from the host. Host names never touch DNS: a fake resolver stands in, so the
test is the same on a laptop, in CI and on an air-gapped box.

Stage names reported on failure: ``profile`` (which profile the switch
reads), ``model_provider``, ``embedder``, ``credential``, ``endpoint``,
``boot`` (the lifespan refuses or starts).
"""

from __future__ import annotations

import pytest

from app.core import deployment_profile as dp

# An environment that keeps every call inside the box: no model provider at
# all (calculators answer without one, see exam 12), offline embedder flags,
# a database and cache on compose service names.
_CLEAN_ONPREM = {
    "DEPLOYMENT_PROFILE": "onprem",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "RAG_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "DATABASE_URL": "postgresql://fork:fork@postgres:5432/fork",
    "REDIS_URL": "redis://cache:6379/0",
    "SECRET_KEY": "local-only",
}


def _resolver(table):
    """A stand-in for DNS: host -> list of addresses; a missing host fails."""

    def resolve(host):
        if host not in table:
            raise OSError(f"cannot resolve {host}")
        return list(table[host])

    return resolve


def _no_dns(host):
    raise AssertionError(f"stage=endpoint: DNS was consulted for {host!r}")


def _offenders(env, resolver=_no_dns):
    return dp.check_onprem_ready(env, resolve=resolver)


def _settings(offenders):
    return sorted(o.setting for o in offenders)


# ── stage: profile ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["", "hosted", "cloud", " Hosted "])
def test_profile_hosted_values(value):
    env = {"DEPLOYMENT_PROFILE": value} if value else {}
    assert dp.deployment_profile(env) == dp.HOSTED, f"stage=profile: {value!r}"


@pytest.mark.parametrize("value", ["onprem", "on-prem", "ON_PREMISE", "on-premise"])
def test_profile_onprem_values(value):
    assert dp.deployment_profile({"DEPLOYMENT_PROFILE": value}) == dp.ONPREM, (
        f"stage=profile: {value!r}"
    )


@pytest.mark.parametrize("value", ["offline", "onprme"])
def test_profile_unknown_value_is_a_refusal(value):
    with pytest.raises(dp.DeploymentProfileError) as err:
        dp.deployment_profile({"DEPLOYMENT_PROFILE": value})
    assert "DEPLOYMENT_PROFILE" in str(err.value), "stage=profile"


# ── hosted: nothing changes ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "env",
    [
        {"LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "k", "SENTRY_DSN": "https://a@b.io/1"},
        {"DEPLOYMENT_PROFILE": "hosted", "LLM_PROVIDER": "ollama",
         "DATABASE_URL": "postgresql://u:p@db.example.net/x"},
    ],
)
def test_hosted_profile_is_untouched(env):
    assert dp.assert_onprem_ready(env, resolve=_no_dns) == [], "stage=boot: hosted must not check"


# ── the clean on-prem box passes ────────────────────────────────────────────


@pytest.mark.parametrize("profile", ["onprem", "on-premise"])
def test_clean_onprem_passes(profile):
    env = {**_CLEAN_ONPREM, "DEPLOYMENT_PROFILE": profile}
    assert _offenders(env) == [], "stage=boot: a clean box must start"
    assert dp.assert_onprem_ready(env, resolve=_no_dns) == []


@pytest.mark.parametrize("sub", ["weights", "bge-small"])
def test_clean_onprem_with_local_embedder_dir_passes(tmp_path, sub):
    (tmp_path / sub).mkdir()
    env = {**_CLEAN_ONPREM, "RAG_EMBEDDING_MODEL": str(tmp_path / sub)}
    env.pop("HF_HUB_OFFLINE")
    env.pop("TRANSFORMERS_OFFLINE")
    assert _offenders(env) == [], "stage=embedder: a local weights dir needs no hub"


# ── stage: model_provider ───────────────────────────────────────────────────


@pytest.mark.parametrize("provider", ["deepseek", "openrouter"])
def test_cloud_provider_named_is_refused(provider):
    env = {**_CLEAN_ONPREM, "LLM_PROVIDER": provider}
    offs = _offenders(env, _resolver({}))
    assert _settings(offs) == ["LLM_PROVIDER"], f"stage=model_provider: {offs}"
    assert "outside" in offs[0].reason


@pytest.mark.parametrize("provider", ["ollama", "llama-local"])
def test_unsupported_provider_is_refused_not_rerouted(provider):
    """The trap: the runtime routes an unknown provider to the cloud default."""
    env = {**_CLEAN_ONPREM, "LLM_PROVIDER": provider}
    offs = _offenders(env)
    assert _settings(offs) == ["LLM_PROVIDER"], f"stage=model_provider: {offs}"
    assert "not a supported provider" in offs[0].reason
    assert provider in offs[0].reason


@pytest.mark.parametrize("fallback", ["ollama", "openrouter"])
def test_fallback_provider_is_checked_too(fallback):
    env = {**_CLEAN_ONPREM, "LLM_FALLBACK_PROVIDER": fallback}
    assert _settings(_offenders(env, _resolver({}))) == ["LLM_FALLBACK_PROVIDER"], (
        "stage=model_provider"
    )


@pytest.mark.parametrize("named", ["", "deepseek"])
def test_default_provider_with_key_is_refused(named):
    """The default provider's key is set: it would call. An unset fallback is
    not an offender of its own."""
    env = {**_CLEAN_ONPREM, "LLM_PROVIDER": named, "DEEPSEEK_API_KEY": "k"}
    offs = _offenders(env, _resolver({}))
    assert _settings(offs) == ["DEEPSEEK_API_KEY", "LLM_PROVIDER"], f"stage=model_provider: {offs}"


@pytest.mark.parametrize("provider_value", ["", "   "])
def test_unset_provider_without_key_is_model_free_and_passes(provider_value):
    env = {**_CLEAN_ONPREM, "LLM_PROVIDER": provider_value, "DEEPSEEK_API_KEY": ""}
    assert _offenders(env) == [], "stage=model_provider: no model configured = no call"


# ── stage: embedder ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("missing", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_hub_embedder_without_offline_flag_is_refused(missing):
    env = {k: v for k, v in _CLEAN_ONPREM.items() if k != missing}
    offs = _offenders(env)
    assert _settings(offs) == ["RAG_EMBEDDING_MODEL"], f"stage=embedder: {offs}"
    assert missing in offs[0].reason


@pytest.mark.parametrize("missing", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_unset_embedder_uses_the_default_and_is_checked(missing):
    env = {k: v for k, v in _CLEAN_ONPREM.items()
           if k not in ("RAG_EMBEDDING_MODEL", missing)}
    assert _settings(_offenders(env)) == ["RAG_EMBEDDING_MODEL"], "stage=embedder"


@pytest.mark.parametrize("flag", ["0", "false"])
def test_offline_flag_must_be_truthy(flag):
    env = {**_CLEAN_ONPREM, "HF_HUB_OFFLINE": flag}
    assert _settings(_offenders(env)) == ["RAG_EMBEDDING_MODEL"], "stage=embedder"


# ── stage: credential ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["SENTRY_DSN", "RESEND_API_KEY", "GOOGLE_ACCESS_TOKEN", "OLLAMA_API_KEY"]
)
def test_remote_service_credential_is_refused(name):
    env = {**_CLEAN_ONPREM, name: "set"}
    offs = _offenders(env)
    assert name in _settings(offs), f"stage=credential: {offs}"


@pytest.mark.parametrize("name,value", [("RESEND_API_KEY", ""), ("SENTRY_DSN", "  ")])
def test_empty_credential_is_not_an_offender(name, value):
    env = {**_CLEAN_ONPREM, name: value}
    assert _offenders(env) == [], "stage=credential: an empty value calls nothing"


def test_offender_message_never_prints_a_value():
    secret = "sk-should-never-print"
    env = {**_CLEAN_ONPREM, "SERPER_API_KEY": secret,
           "DATABASE_URL": f"postgresql://u:{secret}@db.cloud.example/x"}
    with pytest.raises(dp.DeploymentProfileError) as err:
        dp.assert_onprem_ready(env, resolve=_resolver({"db.cloud.example": ["1.1.1.1"]}))
    text = str(err.value)
    assert secret not in text, "stage=boot: a value leaked into the refusal"
    assert "SERPER_API_KEY" in text and "DATABASE_URL" in text


# ── stage: endpoint ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://u:p@127.0.0.1:5432/x",
        "postgresql://u:p@10.2.3.4/x",
        "redis://[::1]:6379",
        "postgresql://u:p@db.corp.example/x",  # resolves to a private address
    ],
)
def test_endpoint_inside_the_box_passes(url):
    env = {**_CLEAN_ONPREM, "DATABASE_URL": url}
    resolve = _resolver({"db.corp.example": ["192.168.10.5"]})
    assert _offenders(env, resolve) == [], f"stage=endpoint: {url}"


@pytest.mark.parametrize(
    "url,table",
    [
        ("postgresql://u:p@ep-x.neon.example/x", {"ep-x.neon.example": ["8.8.8.8"]}),
        ("postgresql://u:p@8.8.4.4/x", {}),
        ("postgresql://u:p@mixed.example/x", {"mixed.example": ["10.0.0.1", "9.9.9.9"]}),
        ("postgresql://u:p@unknown.example/x", {}),  # cannot be shown to stay inside
    ],
)
def test_endpoint_outside_the_box_is_refused(url, table):
    env = {**_CLEAN_ONPREM, "DATABASE_URL": url}
    offs = _offenders(env, _resolver(table))
    assert _settings(offs) == ["DATABASE_URL"], f"stage=endpoint: {url} -> {offs}"


@pytest.mark.parametrize(
    "url,table,expected",
    [
        ("postgresql://u:p@[zz::1/x", {}, ["DATABASE_URL"]),  # unreadable: refused
        ("postgresql://u:p@dead.beef.cafe.face/x", {"dead.beef.cafe.face": ["10.1.1.1"]}, []),
        ("postgresql://u:p@999.1.1.1/x", {}, ["DATABASE_URL"]),  # not an address, no DNS
        ("not a url at all", {}, []),  # nothing is dialled
    ],
)
def test_endpoint_values_that_are_not_plain_addresses(url, table, expected):
    env = {**_CLEAN_ONPREM, "DATABASE_URL": url}
    assert _settings(_offenders(env, _resolver(table))) == expected, f"stage=endpoint: {url}"


@pytest.mark.parametrize("name", ["PUBLIC_BASE_URL", "FRONTEND_URL"])
def test_inbound_self_address_is_not_an_endpoint(name):
    """The service's own public address is where users reach it, not a call out."""
    env = {**_CLEAN_ONPREM, name: "https://shovel.client.example"}
    assert _offenders(env) == [], "stage=endpoint: inbound addresses are not dialled"


@pytest.mark.parametrize("name", ["ACONEX_LOBBY_URL", "SOME_NEW_SERVICE_URL"])
def test_any_dialled_url_setting_is_checked(name):
    env = {**_CLEAN_ONPREM, name: "https://lobby.vendor.example/x"}
    offs = _offenders(env, _resolver({"lobby.vendor.example": ["1.0.0.1"]}))
    assert _settings(offs) == [name], "stage=endpoint"


# ── stage: boot ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "extra,drop,expected",
    [
        ({"LLM_PROVIDER": "ollama", "SENTRY_DSN": "https://a@b.io/1"}, "HF_HUB_OFFLINE",
         {"LLM_PROVIDER", "SENTRY_DSN", "RAG_EMBEDDING_MODEL"}),
        ({"LLM_FALLBACK_PROVIDER": "kimi", "SERPER_API_KEY": "x"}, "TRANSFORMERS_OFFLINE",
         {"LLM_FALLBACK_PROVIDER", "SERPER_API_KEY", "RAG_EMBEDDING_MODEL"}),
    ],
)
def test_assert_names_every_offender_at_once(extra, drop, expected):
    env = {**_CLEAN_ONPREM, **extra}
    env.pop(drop)
    with pytest.raises(dp.DeploymentProfileError) as err:
        dp.assert_onprem_ready(env, resolve=_no_dns)
    names = {o.setting for o in err.value.offenders}
    assert names == expected, "stage=boot"
    for name in expected:
        assert name in str(err.value)


def test_lifespan_guard_reads_the_process_env(monkeypatch):
    """The boot hook refuses on a leaking on-prem env and starts on a clean one."""
    env = dict(_CLEAN_ONPREM)
    monkeypatch.setattr(dp, "_process_env", lambda: env)
    monkeypatch.setattr(dp, "_resolve_host", _no_dns)
    dp.enforce_deployment_profile()  # clean: starts

    env["LLM_PROVIDER"] = "ollama"
    with pytest.raises(dp.DeploymentProfileError):
        dp.enforce_deployment_profile()


def test_lifespan_calls_the_guard():
    import inspect

    from app import main

    assert "enforce_deployment_profile" in inspect.getsource(main.lifespan), "stage=boot"


# ── the plumbing the boot hook relies on ───────────────────────────────────


@pytest.mark.parametrize("value,expected", [("onprem", "onprem"), ("hosted", "hosted")])
def test_profile_reads_the_process_env_by_default(monkeypatch, value, expected):
    monkeypatch.setenv("DEPLOYMENT_PROFILE", value)
    assert dp._process_env()["DEPLOYMENT_PROFILE"] == value
    assert dp.deployment_profile() == expected, "stage=profile"


@pytest.mark.parametrize("host,expected", [("127.0.0.1", "127.0.0.1"), ("::1", "::1")])
def test_resolve_host_returns_addresses(host, expected):
    """Address literals resolve locally; no network is used."""
    assert expected in dp._resolve_host(host), "stage=endpoint"


@pytest.mark.parametrize("offenders", [[], [dp.Offender("SENTRY_DSN", "remote")]])
def test_profile_error_carries_its_offenders(offenders):
    err = dp.DeploymentProfileError.__new__(dp.DeploymentProfileError)
    dp.DeploymentProfileError.__init__(err, "refused", offenders)
    assert str(err) == "refused"
    assert err.offenders == offenders, "stage=boot"
    assert err.offenders is not offenders
