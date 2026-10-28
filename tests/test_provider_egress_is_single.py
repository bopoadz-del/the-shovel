"""The provider HTTP egress lives in exactly one place (ORDERS_2050Z #2c).

AST + simple intra-module dataflow. Scan ALL of app/ (no module pre-filter),
every call form (.post/.stream/.request/.send on any receiver). A call OUTSIDE
app/core/llm_client.py FAILS if:
  (i)  its URL (positional or url=) is a string literal / f-string whose literal
       text contains a provider host — and the host list is DERIVED here from
       the app's own provider config (runtime.py *_API_URL / *_BASE_URL), never
       hand-typed; or
  (ii) its URL is non-literal and, by dataflow within its function, originates
       from _llm_config() / cfg["url"] / cfg.get("url") / an env or config key
       whose name contains LLM, PROVIDER, DEEPSEEK, OPENROUTER or MODEL_URL.
Variable URLs with no LLM-config origin pass (Drive, Aconex, OAuth, email).
Every real provider call therefore routes through llm_client.provider_post /
provider_stream, which meter by construction.
"""
import ast
import pathlib
import re
from urllib.parse import urlparse

import app.agents.runtime as _rt

_APP = pathlib.Path(__file__).resolve().parent.parent / "app"
_CHOKEPOINT = (_APP / "core" / "llm_client.py").resolve()
_HTTP = {"post", "stream", "request", "send"}
_VERBS = {"POST", "GET", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
_LLM_KEY_RE = re.compile(r"LLM|PROVIDER|DEEPSEEK|OPENROUTER|MODEL_URL")


def _provider_hosts():
    hosts = set()
    for name in dir(_rt):
        if name.endswith("_API_URL") or name.endswith("_BASE_URL"):
            val = getattr(_rt, name)
            if isinstance(val, str) and "://" in val:
                netloc = urlparse(val).netloc.lower()
                if netloc:
                    hosts.add(netloc)
    return hosts


def _url_arg(call: ast.Call):
    for kw in call.keywords:
        if kw.arg == "url":
            return kw.value
    pos = [a for a in call.args if not isinstance(a, ast.Starred)]
    if not pos:
        return None
    first = pos[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str) \
            and first.value.upper() in _VERBS:
        return pos[1] if len(pos) > 1 else None
    return first


def _literal_text(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return " ".join(v.value for v in node.values
                        if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return None


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


# A provider-URL constant by name: DEEPSEEK_API_URL, rt.OPENROUTER_BASE_URL, …
_LLM_URL_CONST_RE = re.compile(r"_(API|BASE)_URL$")


def _expr_is_llm_origin(node, origin) -> bool:
    """Whether an expression originates from the LLM provider config.

    `origin` is the set of names in the module already known to hold the LLM
    config or a value derived from it (built by whole-module propagation), so
    recognising a config object needs no hand-typed list of variable names:
    a name counts because it was bound from ``_llm_config()`` somewhere, not
    because it is spelled ``cfg``.
    """
    if node is None:
        return False
    # A provider-URL constant, or a name already known to be LLM-derived.
    if isinstance(node, ast.Name):
        return node.id in origin or bool(_LLM_URL_CONST_RE.search(node.id))
    if isinstance(node, ast.Attribute):
        return bool(_LLM_URL_CONST_RE.search(node.attr))
    if isinstance(node, ast.Subscript):
        key = _const_str(node.slice)
        if key and _LLM_KEY_RE.search(key.upper()):     # os.environ["LLM_..."]
            return True
        if key == "url":                                 # <llm-cfg>["url"]
            return _expr_is_llm_origin(node.value, origin)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name) and f.id == "_llm_config":
            return True
        if isinstance(f, ast.Attribute) and f.attr == "_llm_config":
            return True
        if isinstance(f, ast.Attribute) and f.attr == "get" and node.args \
                and _const_str(node.args[0]) == "url":   # <llm-cfg>.get("url")
            return _expr_is_llm_origin(f.value, origin)
        # os.getenv("...LLM...") / os.environ.get("...PROVIDER...")
        if isinstance(f, ast.Attribute) and f.attr in ("getenv", "get") and node.args \
                and _const_str(node.args[0]) and _LLM_KEY_RE.search(_const_str(node.args[0]).upper()):
            return True
    return False


def _origin_names(tree) -> set:
    """Whole-module set of names bound, transitively, from the LLM config.

    Scope-INSENSITIVE on purpose: a ``cfg = _llm_config()`` in one function and
    a ``url = cfg["url"]`` alias consumed inside a nested closure in another are
    the same provider origin (the chat.py ``_call_cloud`` -> ``_stream_generator``
    shape). Over-flagging a non-LLM reuse of the same name is the safe direction
    — it forces the call through the metered chokepoint; a miss would not.
    """
    origin: set = set()
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    changed = True
    while changed:
        changed = False
        for n in assigns:
            if _expr_is_llm_origin(n.value, origin):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id not in origin:
                        origin.add(t.id)
                        changed = True
    return origin


def scan_source(src: str, is_chokepoint: bool, hosts: set) -> list:
    if is_chokepoint:
        return []
    tree = ast.parse(src)
    origin = _origin_names(tree)
    offenders = []
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        if call.func.attr not in _HTTP:
            continue
        url = _url_arg(call)
        if url is None:
            continue
        lit = _literal_text(url)
        if lit is not None:
            if any(h in lit.lower() for h in hosts):
                offenders.append(call.lineno)
            continue
        # Literal-free URL: flag it when it originates from the LLM config —
        # a cfg["url"]/cfg.get("url") on an _llm_config-derived name, a provider
        # URL constant, an LLM env key, or a name aliased from any of those
        # anywhere in the module (closures included).
        if _expr_is_llm_origin(url, origin):
            offenders.append(call.lineno)
    return offenders


_HOSTS = _provider_hosts()


def test_provider_hosts_were_derived_from_the_app_config():
    # The host list must come from the app, not be hand-typed here.
    assert _HOSTS, "no provider hosts found in runtime.py *_API_URL/*_BASE_URL"


def test_no_provider_egress_outside_llm_client():
    offenders = []
    for path in _APP.rglob("*.py"):
        if path.resolve() == _CHOKEPOINT:
            continue
        src = path.read_text(encoding="utf-8", errors="ignore")
        for ln in scan_source(src, False, _HOSTS):
            offenders.append(f"{path.relative_to(_APP.parent)}:{ln}")
    assert not offenders, (
        "provider HTTP egress outside app/core/llm_client.py — route it through "
        "llm_client.provider_post / provider_stream: " + ", ".join(sorted(offenders))
    )


def test_the_chokepoint_actually_makes_the_calls():
    text = _CHOKEPOINT.read_text(encoding="utf-8")
    assert "client.post(url" in text
    assert 'client.stream("POST", url' in text


# ── mutation cases: the guard must catch both shapes ─────────────────────────

def test_guard_flags_a_hardcoded_provider_host_literal():
    host = sorted(_HOSTS)[0]
    src = (
        "import httpx\n"
        "async def f(client):\n"
        f"    return await client.post('https://{host}/v1/chat/completions', json={{}})\n"
    )
    assert scan_source(src, False, _HOSTS), "a literal provider-host URL must FAIL"


def test_guard_flags_a_cfg_url_from_llm_config_on_any_receiver():
    # cfg is the config because it is bound from _llm_config(), not because it
    # is spelled "cfg" (no hand-typed name set).
    src = (
        "async def f(self):\n"
        "    cfg = _llm_config()\n"
        "    return await self.client.post(cfg['url'], json={})\n"
    )
    assert scan_source(src, False, _HOSTS), "self.client.post(cfg['url']) must FAIL"


def test_guard_flags_a_closure_alias_of_the_config_url():
    # The chat.py shape: cfg assigned in the outer function, aliased to url, and
    # the raw stream call sits inside a nested closure. Whole-module propagation
    # must catch it (this is the offender the intra-function guard missed).
    src = (
        "def outer(client):\n"
        "    cfg = _llm_config()\n"
        "    url = cfg['url']\n"
        "    async def inner():\n"
        "        async with client.stream('POST', url, json={}) as r:\n"
        "            return r\n"
        "    return inner\n"
    )
    assert scan_source(src, False, _HOSTS), "closure alias of cfg['url'] must FAIL"


def test_guard_flags_a_provider_url_constant():
    src = (
        "async def f(client):\n"
        "    return await client.post(DEEPSEEK_API_URL, json={})\n"
    )
    assert scan_source(src, False, _HOSTS), "a *_API_URL provider constant must FAIL"


def test_guard_passes_a_non_llm_variable_url():
    # A variable URL with no LLM-config origin (e.g. a search API) passes.
    src = (
        "async def f(client):\n"
        "    url = 'https://search.example/api'\n"
        "    return await client.post(url, json={})\n"
    )
    assert scan_source(src, False, _HOSTS) == []


def test_guard_passes_a_non_llm_dict_get_url():
    # A webhook payload's data.get('url') is not the LLM config (data is not
    # bound from _llm_config), so it must pass — no hand-typed name exceptions.
    src = (
        "async def f(client, data):\n"
        "    url = data.get('url')\n"
        "    return await client.post(url, json={})\n"
    )
    assert scan_source(src, False, _HOSTS) == []
