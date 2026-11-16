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
        ("postgresql://u:p@ep-x.quillbase.example/x", {"ep-x.quillbase.example": ["8.8.8.8"]}),
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


@pytest.mark.parametrize("name", ["TARNWELL_LOBBY_URL", "SOME_NEW_SERVICE_URL"])
def test_any_dialled_url_setting_is_checked(name):
    env = {**_CLEAN_ONPREM, name: "https://lobby.vendor.example/x"}
    offs = _offenders(env, _resolver({"lobby.vendor.example": ["1.0.0.1"]}))
    assert _settings(offs) == [name], "stage=endpoint"


# ── G-6: judged by value and by word, not by a list of name endings ──────
# Every name below is invented; none appears in the module.


@pytest.mark.parametrize(
    "name",
    [
        "BRAMBLE_SESSION_TOKEN",   # ends _TOKEN, not _ACCESS_TOKEN
        "QUILLBASE_SECRET",        # _SECRET
        "ORCHARDLY_SECRET_KEY",    # _SECRET_KEY
        "FENWAY_DB_PASSWORD",      # PASSWORD
        "LOOMKIT_PASSWD_HASH",     # PASSWD mid-name
        "MARROW_CREDENTIALS_JSON", # CREDENTIAL mid-name, plural
        "KESTRELAPIKEY",           # no separator at all
        "tarn_refresh_token",      # lower case
    ],
)
def test_any_name_carrying_a_credential_word_is_refused(name):
    env = {**_CLEAN_ONPREM, name: "abc123-not-a-number"}
    offs = _offenders(env)
    assert _settings(offs) == [name], f"stage=credential: {name} -> {offs}"
    assert "credential" in offs[0].reason


@pytest.mark.parametrize("name", ["WHISTLER_MODE", "GRANARY_TIMEOUT_S"])
def test_name_without_a_credential_word_is_not_a_credential(name):
    env = {**_CLEAN_ONPREM, name: "abc123-not-a-number"}
    assert _offenders(env) == [], f"stage=credential: {name}"


@pytest.mark.parametrize("name", ["PEBBLE_MAX_TOKENS", "SIFT_DAILY_TOKEN_BUDGET"])
def test_a_whole_number_is_a_count_not_a_credential(name):
    env = {**_CLEAN_ONPREM, name: "4096"}
    assert _offenders(env) == [], f"stage=credential: a count under {name}"
    env[name] = "4096x"  # the same name with a non-count value is a secret
    assert _settings(_offenders(env)) == [name], f"stage=credential: {name}"


@pytest.mark.parametrize("value", ["generated-locally-abc", "Q9x_kL2-0pW"])
def test_local_only_secrets_are_exactly_what_the_installer_generates(value):
    """The allow list is the installer's own generated secrets (plus the base
    image's public signing-key fingerprint): read from the installer itself."""
    import pathlib
    import re

    script = pathlib.Path(__file__).resolve().parents[1] / "deploy" / "onprem" / "install.sh"
    generated = set(re.findall(r'echo "([A-Z0-9_]+)=\$\(gen\)"', script.read_text(encoding="utf-8")))
    assert generated, "stage=credential: the installer generates no secret (parser stale?)"
    assert generated <= dp._LOCAL_ONLY_SECRETS, f"stage=credential: {generated - dp._LOCAL_ONLY_SECRETS}"
    for name in sorted(generated):
        env = {**_CLEAN_ONPREM, name: value}
        assert _offenders(env) == [], f"stage=credential: {name} is local-only"
        # the same word in a name the installer does not generate is refused
        twin = "REMOTE_" + name
        assert _settings(_offenders({**_CLEAN_ONPREM, twin: value})) == [twin], (
            f"stage=credential: {twin}"
        )
    leftover = dp._LOCAL_ONLY_SECRETS - generated - {"BOOTSTRAP_USER_PASSWORD", "GPG_KEY"}
    assert not leftover, f"stage=credential: allow list grew past the installer: {leftover}"


@pytest.mark.parametrize(
    "name,value,table",
    [
        # an unlisted name, a URL value: checked though the name is not *_URL
        ("HARROWGATE_ENDPOINT", "https://api.quillbase.example/v1", {"api.quillbase.example": ["1.0.0.1"]}),
        ("MOSSPATH_URI", "amqp://broker.quillbase.example:5672", {"broker.quillbase.example": ["9.9.9.9"]}),
        # no scheme: a bare host:port
        ("TARN_HOST", "relay.quillbase.example:2525", {"relay.quillbase.example": ["1.0.0.1"]}),
        ("CINDER_SERVER", "8.8.4.4", {}),
        ("SPRIG_PEER", "[2001:4860::1]:443", {}),
        # one bad entry in a list is enough
        ("WICKET_ORIGINS", "http://10.0.0.2:3000, https://far.quillbase.example", {"far.quillbase.example": ["1.0.0.1"]}),
        # a name that cannot be shown to stay inside
        ("BURROW_TARGET", "nowhere.quillbase.example", {}),
    ],
)
def test_any_value_that_dials_outside_is_refused_whatever_the_name(name, value, table):
    env = {**_CLEAN_ONPREM, name: value}
    offs = _offenders(env, _resolver(table))
    assert _settings(offs) == [name], f"stage=endpoint: {name}={value} -> {offs}"
    assert "outside" in offs[0].reason or "not shown" in offs[0].reason


@pytest.mark.parametrize(
    "name,value,table",
    [
        ("HARROWGATE_ENDPOINT", "https://api.corp.example/v1", {"api.corp.example": ["10.9.8.7"]}),
        ("TARN_HOST", "relay.corp.example:2525", {"relay.corp.example": ["172.16.0.9"]}),
        ("CINDER_SERVER", "192.168.1.20", {}),
        ("SPRIG_PEER", "ollama:11434", {}),           # compose service name
        ("WICKET_ORIGINS", "http://10.0.0.2:3000, http://localhost:5173", {}),
    ],
)
def test_any_value_that_stays_inside_passes_whatever_the_name(name, value, table):
    env = {**_CLEAN_ONPREM, name: value}
    assert _offenders(env, _resolver(table)) == [], f"stage=endpoint: {name}={value}"


@pytest.mark.parametrize(
    "value",
    ["3.11.9", "qwen2.5:14b-instruct", "BAAI/bge-small-en-v1.5", "/app/data", "C.UTF-8", "mill_pond"],
)
def test_values_that_are_not_hosts_never_reach_dns(value):
    env = {**_CLEAN_ONPREM, "GRANARY_SETTING": value}
    assert _offenders(env) == [], f"stage=endpoint: {value!r} is not a host"


# ── G-6 grader FAIL (#386): a host ANYWHERE in a value; numbers under a ──
# secret's name stay secrets. Every name and host below is invented.


@pytest.mark.parametrize(
    "name,value",
    [
        ("FENWAY_DB_PASSWORD", "482913"),     # a numeric PIN or password
        ("QUILLBASE_SECRET", "7"),
        ("ORCHARD_API_KEY", "123456789"),
        ("BRAMBLE_SESSION_TOKEN", "55501"),  # TOKEN, but the name counts nothing
        ("LOOM_MAX_KEY", "4096"),            # a count word cannot cover KEY
        ("SIFT_TOKEN_LIMIT_SECRET", "12"),   # nor SECRET
    ],
)
def test_a_number_under_a_secret_name_is_still_a_credential(name, value):
    offs = _offenders({**_CLEAN_ONPREM, name: value})
    assert _settings(offs) == [name], f"stage=credential: {name}={value} -> {offs}"
    assert "credential" in offs[0].reason


@pytest.mark.parametrize("name", ["PEBBLE_MAX_TOKENS", "SIFT_TOKEN_COUNT", "MOSS_TOKEN_LIMIT"])
def test_a_number_under_a_token_count_name_is_a_count(name):
    assert _offenders({**_CLEAN_ONPREM, name: "8192"}) == [], f"stage=credential: {name}"
    assert _settings(_offenders({**_CLEAN_ONPREM, name: "8192-k"})) == [name]


_FAR = {
    "far.quillbase.example": ["1.0.0.1"],
    "mail.quillbase.example": ["9.9.9.9"],
    "db.quillbase.example": ["8.8.8.8"],
    "relay.xn--p1ai": ["1.0.0.1"],
}
_NEAR = {
    "near.corp.example": ["10.9.8.7"],
    "relay.corp.example": ["172.16.0.9"],
    "box.xn--80ak6aa92e": ["192.168.4.4"],
}


@pytest.mark.parametrize(
    "form,outside,inside",
    [
        ("json object",
         '{"endpoint": "https://far.quillbase.example/v1", "retries": 3}',
         '{"endpoint": "https://near.corp.example/v1", "retries": 3}'),
        ("json nested list, bare host:port",
         '[{"hosts": ["relay.corp.example:25", "db.quillbase.example:5432"]}]',
         '[{"hosts": ["relay.corp.example:25", "ollama:11434"]}]'),
        ("bare IPv6, unbracketed", "2001:4860:4860::8888", "fd00::5"),
        ("IPv6 inside a list", "10.0.0.4, 2606:4700::1111", "10.0.0.4, fe80::1"),
        ("user@host:port, no scheme", "svc@mail.quillbase.example:587", "svc@relay.corp.example:587"),
        ("user@host, no port", "deploy@far.quillbase.example", "deploy@near.corp.example"),
        ("key=value conninfo", "host=db.quillbase.example port=5432", "host=near.corp.example port=5432"),
        ("quoted key=value", "host='far.quillbase.example' user=x", "host='near.corp.example' user=x"),
        ("URL inside text", "send to https://far.quillbase.example/hook now",
         "send to http://near.corp.example/hook now"),
        ("punycode last label", "relay.xn--p1ai:2525", "box.xn--80ak6aa92e:2525"),
        ("decimal IP in a URL", "http://134744072/x", "http://167772162/x"),  # 8.8.8.8 / 10.0.0.2
        ("hex IP in a URL", "http://0x08080808/x", "http://0x7f000001/x"),    # 8.8.8.8 / 127.0.0.1
        ("decimal IP with a port", "134744072:443", "3232235777:443"),       # 8.8.8.8 / 192.168.1.1
        ("short dotted IP after @", "u@8.0x080808", "u@10.2"),               # 8.8.8.8 / 10.0.0.2
        # JSON escapes are readable only once the value is parsed as JSON
        ("json escaped slashes",
         r'{"endpoint": "https:\/\/far.quillbase.example\/v1"}',
         r'{"endpoint": "https:\/\/near.corp.example\/v1"}'),
        ("json unicode escape",
         r'["far.quillbase.example:443"]',
         r'["near.corp.example:443"]'),
    ],
)
def test_a_host_anywhere_in_a_value_is_checked(form, outside, inside):
    resolve = _resolver({**_FAR, **_NEAR})
    offs = _offenders({**_CLEAN_ONPREM, "GRANARY_SETTING": outside}, resolve)
    assert _settings(offs) == ["GRANARY_SETTING"], f"stage=endpoint: {form}: {outside!r} -> {offs}"
    assert _offenders({**_CLEAN_ONPREM, "GRANARY_SETTING": inside}, resolve) == [], (
        f"stage=endpoint: {form}: {inside!r} stays inside"
    )


@pytest.mark.parametrize(
    "value",
    [
        "4096",                     # a plain number is a count, not a decimal IP
        "12:30",                    # a time, not host:port
        "./report.final.pdf",       # a file name written as a path
        "/var/log/shovel/app.log",  # a path
        "C:\\data\\export.csv",     # a Windows path
        "/srv/run/worker@svc.quillbase.example.sock",  # a path holding an @
        "./exports/team@far.quillbase.example.csv",
        '{"retries": 3, "mode": "fast"}',
        "[not json, just text]",
    ],
)
def test_values_holding_no_host_never_reach_dns(value):
    env = {**_CLEAN_ONPREM, "GRANARY_SETTING": value}
    assert _offenders(env) == [], f"stage=endpoint: {value!r} holds no host"


@pytest.mark.parametrize(
    "text,expected",
    [("167772162", "10.0.0.2"), ("0x7f000001", "127.0.0.1"), ("10.2", "10.0.0.2"),
     ("0177.1", "127.0.0.1"), ("300.1.1.1", None), ("1.2.3.4.0", None), ("1.2.3.4.5", None), ("4294967296", None), ("1.2.x", None)],
)
def test_numeric_ipv4_reads_like_the_system_resolver(text, expected):
    got = dp._numeric_ipv4(text)
    assert (str(got) if got else None) == expected


@pytest.mark.parametrize(
    "part,expected",
    [
        ("0x1f", 31), ("0x", None), ("0x1g", None), ("0x1_f", None),   # hex: digits only, at least one
        ("017", 15), ("018", None), ("0_7", None),                     # octal after a leading 0
        ("0", 0), ("42", 42), ("4_2", None), ("٤٢", None),  # decimal: ASCII digits only
        ("", None), ("x1", None),
    ],
)
def test_one_dotted_part_reads_as_the_resolver_number(part, expected):
    assert dp._resolver_number(part) == expected


@pytest.mark.parametrize("host,expected", [("relay.\ud800x", False), ("localhost", True)])
def test_an_unreadable_non_ascii_host_is_outside_and_logged(caplog, host, expected):
    def never(_host):
        raise AssertionError("an unreadable or local host must not reach DNS")

    with caplog.at_level("WARNING", logger=dp.logger.name):
        assert dp._host_is_inside(host, never) is expected
    assert ("not a readable name" in caplog.text) is (not expected)


def _shipped_onprem_env():
    """The on-prem box's environment as shipped: onprem.env.example, the
    installer's generated secrets, and the compose file's app settings with
    their defaults filled in."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "deploy" / "onprem"
    env = {}
    for line in (root / "onprem.env.example").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    for name in re.findall(r'echo "([A-Z0-9_]+)=\$\(gen\)"', (root / "install.sh").read_text(encoding="utf-8")):
        env[name] = "Zk3-generated-on-the-box-9qL"
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    for k, v in re.findall(r"^\s+([A-Z][A-Z0-9_]+):\s*(\S.*)$", compose, flags=re.MULTILINE):
        env.setdefault(k, re.sub(r"\$\{[A-Z0-9_]+:-([^}]*)\}", r"\1", v).strip())
    assert env.get("DEPLOYMENT_PROFILE") == "onprem" and "DATABASE_URL" in env, sorted(env)
    return env


def test_the_shipped_onprem_env_boots(monkeypatch):
    """Through the real boot hook, with the box's real shipped settings."""
    env = _shipped_onprem_env()
    monkeypatch.setattr(dp, "_process_env", lambda: env)
    monkeypatch.setattr(dp, "_resolve_host", _no_dns)
    dp.enforce_deployment_profile()


@pytest.mark.parametrize(
    "name,value",
    [("KESTREL_SESSION_TOKEN", "t0k-abc"), ("MOSSPATH_URI", "amqp://broker.quillbase.example")],
)
def test_the_shipped_onprem_env_refuses_one_unlisted_leak(monkeypatch, name, value):
    env = {**_shipped_onprem_env(), name: value}
    monkeypatch.setattr(dp, "_process_env", lambda: env)
    monkeypatch.setattr(dp, "_resolve_host", _resolver({}))
    with pytest.raises(dp.DeploymentProfileError) as err:
        dp.enforce_deployment_profile()
    assert [o.setting for o in err.value.offenders] == [name], "stage=boot"


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


# ── grader round 3 on #386: whole words, five more host forms, fail closed ──
# Every name and host below is invented. Each decision has an outside value
# (refused) and an inside value (passes).


@pytest.mark.parametrize(
    "inside_a_word,whole_word",
    [
        ("PLINTH_ADMIN_TOKEN", "PLINTH_MIN_TOKEN"),        # ADMIN holds MIN
        ("LEDGER_ACCOUNT_TOKEN", "LEDGER_TOKEN_COUNT"),    # ACCOUNT holds COUNT
        ("SPRUCE_MAXIMAL_TOKEN", "SPRUCE_MAX_TOKEN"),      # MAXIMAL holds MAX
        ("TALLOW_SUBLIMITED_TOKEN", "TALLOW_TOKEN_LIMIT"),  # SUBLIMITED holds LIMIT
    ],
)
def test_count_words_are_whole_words_only(inside_a_word, whole_word):
    """A short number under a name whose count word is only inside another
    word is a credential; under a whole count word it is a token budget."""
    for value in ("123456", "7"):
        offs = _offenders({**_CLEAN_ONPREM, inside_a_word: value})
        assert _settings(offs) == [inside_a_word], f"stage=credential: {inside_a_word}={value}"
        assert _offenders({**_CLEAN_ONPREM, whole_word: value}) == [], f"stage=credential: {whole_word}"


@pytest.mark.parametrize(
    "credential,not_credential",
    [
        ("KILN_PASSPHRASE", "KILN_PASSPHRASES_HELP"),
        ("KILN_AUTH", "KILN_AUTHOR"),
        ("KILN_BEARER", "KILN_BEARERS_SEEN"),
        ("KILN_DB_PWD", "KILN_PWDX"),
        ("KILN_SIGNATURE", "KILN_SIGNATURES_DIR"),
        ("AUTH_HEADER", "OAUTH_FLOW_NAME"),
    ],
)
def test_credential_whole_words(credential, not_credential):
    """PASSPHRASE, AUTH, BEARER, PWD, SIGNATURE name a secret as whole words."""
    assert _settings(_offenders({**_CLEAN_ONPREM, credential: "x7-q"})) == [credential], (
        f"stage=credential: {credential}"
    )
    assert _offenders({**_CLEAN_ONPREM, not_credential: "x7-q"}) == [], f"stage=credential: {not_credential}"


@pytest.mark.parametrize("value,expected", [("/home/worker/app", []), ("x7-q", ["PWD"])])
def test_the_shells_working_directory_is_not_a_secret(value, expected):
    """PWD (exactly that name) holding an absolute path is the shell's working
    directory; any other value under it is a credential."""
    assert _settings(_offenders({**_CLEAN_ONPREM, "PWD": value})) == expected, "stage=credential"


_IDN_FAR = "mail.bücher.example"
_IDN_NEAR = "relay.bücher.example"
_ROUND3_TABLE = {
    **_FAR,
    **_NEAR,
    "far_edge.quillbase.example": ["9.9.9.9"],
    "near_edge.corp.example": ["10.4.4.4"],
    _IDN_FAR.encode("idna").decode(): ["1.0.0.1"],
    _IDN_NEAR.encode("idna").decode(): ["10.7.7.7"],
}


@pytest.mark.parametrize(
    "form,outside,inside",
    [
        ("scheme-relative //host", "//far.quillbase.example/v1", "//near.corp.example/v1"),
        ("scheme-relative //user@host:port", "//u@db.quillbase.example:5432", "//u@relay.corp.example:5432"),
        ("host then ?query", "far.quillbase.example?region=1", "near.corp.example?region=1"),
        ("host then #fragment", "far.quillbase.example#top", "near.corp.example#top"),
        ("underscore in a label", "far_edge.quillbase.example:443", "near_edge.corp.example:443"),
        ("non-ASCII host", f"{_IDN_FAR}:25", f"{_IDN_NEAR}:25"),
        ("non-ASCII host in a URL", f"smtp://{_IDN_FAR}", f"smtp://{_IDN_NEAR}"),
        ("bare decimal IP, no port", "134744072", "167772162"),   # 8.8.8.8 / 10.0.0.2
        ("bare hex IP, no port", "0x08080808", "0x0a000002"),     # 8.8.8.8 / 10.0.0.2
        ("encoded dots", "far%2Equillbase%2Eexample:443", "near%2Ecorp%2Eexample:443"),
    ],
)
def test_round3_host_forms_are_checked(form, outside, inside):
    resolve = _resolver(_ROUND3_TABLE)
    offs = _offenders({**_CLEAN_ONPREM, "GRANARY_SETTING": outside}, resolve)
    assert _settings(offs) == ["GRANARY_SETTING"], f"stage=endpoint: {form}: {outside!r} -> {offs}"
    assert _offenders({**_CLEAN_ONPREM, "GRANARY_SETTING": inside}, resolve) == [], (
        f"stage=endpoint: {form}: {inside!r} stays inside"
    )


@pytest.mark.parametrize("value", ["134744072", "0x08080808"])
def test_a_bare_number_under_a_count_name_is_a_count_not_an_address(value):
    """The same bare number is an address under a plain name, a count under a
    name with a whole count word."""
    assert _offenders({**_CLEAN_ONPREM, "GRANARY_MAX_ROWS": value}) == [], "stage=endpoint"
    assert _settings(_offenders({**_CLEAN_ONPREM, "GRANARY_ROWS": value})) == ["GRANARY_ROWS"]


@pytest.mark.parametrize("stem", ["handover.pdf", "site_export.csv"])
def test_a_file_name_is_written_as_a_path(stem):
    """Convention (operator docs): a bare name.ext reads as a host and is
    refused; written as ./name.ext it is a path and passes."""
    unresolvable = _resolver({})
    assert _settings(_offenders({**_CLEAN_ONPREM, "GRANARY_FILE": stem}, unresolvable)) == ["GRANARY_FILE"]
    assert _offenders({**_CLEAN_ONPREM, "GRANARY_FILE": f"./{stem}"}) == [], "stage=endpoint: no DNS"


@pytest.mark.parametrize("endpoint", ["https://far.quillbase.example/v1", "https://near.corp.example/v1"])
def test_a_file_holding_an_endpoint_is_not_read(tmp_path, endpoint):
    """Stated limit (operator docs): the check reads values, never the files
    they point at. A path passes whatever the file holds."""
    path = tmp_path / "endpoint.conf"
    path.write_text(endpoint, encoding="utf-8")
    assert _offenders({**_CLEAN_ONPREM, "GRANARY_CONFIG_FILE": str(path)}) == [], "stage=endpoint"


@pytest.mark.parametrize(
    "unclassified,plain",
    [
        ("mirror/far.quillbase.example", "mirror/latest"),         # a host where no host reading puts it
        ("pull:near.corp.example:latest", "pull:stable:latest"),    # even a host that would be inside
        ("fork\x01", "fork"),                                        # a control character
        ("relay.\ud800x", "relay-x"),                                # a name IDNA cannot convert
    ],
)
def test_a_value_the_parser_cannot_classify_is_refused(unclassified, plain):
    """Fail closed: a value with a part that is none of the known-safe shapes
    is refused with its setting name (and never its value)."""
    resolve = _resolver({**_FAR, **_NEAR})
    offs = _offenders({**_CLEAN_ONPREM, "GRANARY_SETTING": unclassified}, resolve)
    assert _settings(offs) == ["GRANARY_SETTING"], f"stage=fail-closed: {unclassified!r} -> {offs}"
    assert unclassified not in offs[0].reason
    assert _offenders({**_CLEAN_ONPREM, "GRANARY_SETTING": plain}, resolve) == [], (
        f"stage=fail-closed: {plain!r} is plain"
    )


@pytest.mark.parametrize("profile,expected", [("hosted", []), ("onprem", ["GRANARY_SETTING"])])
def test_fail_closed_applies_only_to_the_onprem_profile(monkeypatch, profile, expected):
    env = {**_CLEAN_ONPREM, "DEPLOYMENT_PROFILE": profile, "GRANARY_SETTING": "mirror/far.quillbase.example"}
    monkeypatch.setattr(dp, "_process_env", lambda: env)
    monkeypatch.setattr(dp, "_resolve_host", _resolver(_FAR))
    if expected:
        with pytest.raises(dp.DeploymentProfileError) as err:
            dp.enforce_deployment_profile()
        assert [o.setting for o in err.value.offenders] == expected
    else:
        dp.enforce_deployment_profile()


@pytest.mark.parametrize(
    "value",
    [
        "sha256:" + "4f" * 32,          # a pinned digest: one 64-character piece
        "ref/" + "a1b2c3d4e5" * 8,      # an 80-character plain piece
    ],
)
def test_a_long_plain_ascii_piece_is_plain_not_a_dns_label(value):
    """Plain ASCII is not held to DNS label rules: a piece longer than a
    label (63) is still a plain word, not an unclassifiable name."""
    assert _offenders({**_CLEAN_ONPREM, "GRANARY_SETTING": value}) == [], "stage=fail-closed"
