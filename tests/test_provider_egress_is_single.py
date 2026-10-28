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


# The LLM config object, by convention (cfg = _llm_config()); a module whose
# own dict happens to hold a "url" (e.g. a webhook payload named `data`) is not
# the LLM config and must not be flagged.
_CFG_NAMES = {"cfg", "a_cfg", "fallback_cfg", "active_cfg"}


def _expr_is_llm_origin(node) -> bool:
    """Direct (non-Name) expressions that originate from the LLM config."""
    if node is None:
        return False
    if isinstance(node, ast.Subscript):
        key = _const_str(node.slice)
        # os.environ["LLM_..."] etc. — the KEY itself names the provider.
        if key and _LLM_KEY_RE.search(key.upper()):
            return True
        # cfg["url"] — the LLM config object by name, or _llm_config()["url"].
        if key == "url":
            if isinstance(node.value, ast.Name) and node.value.id in _CFG_NAMES:
                return True
            if isinstance(node.value, ast.Call):
                return _expr_is_llm_origin(node.value)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name) and f.id == "_llm_config":
            return True
        if isinstance(f, ast.Attribute) and f.attr == "_llm_config":
            return True
        # cfg.get("url") — only on the LLM config object.
        if isinstance(f, ast.Attribute) and f.attr == "get" and node.args \
                and _const_str(node.args[0]) == "url" \
                and isinstance(f.value, ast.Name) and f.value.id in _CFG_NAMES:
            return True
        # os.getenv("...LLM...") / os.environ.get("...PROVIDER...")
        if isinstance(f, ast.Attribute) and f.attr in ("getenv", "get") and node.args \
                and _const_str(node.args[0]) and _LLM_KEY_RE.search(_const_str(node.args[0]).upper()):
            return True
    return False


def _llm_origin_names(func) -> set:
    """Names assigned (anywhere in func) from an LLM-origin expr, transitively."""
    origin: set = set()
    assigns = [n for n in ast.walk(func) if isinstance(n, ast.Assign)]
    changed = True
    while changed:
        changed = False
        for n in assigns:
            val = n.value
            if _expr_is_llm_origin(val) or (isinstance(val, ast.Name) and val.id in origin):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id not in origin:
                        origin.add(t.id)
                        changed = True
    return origin


def scan_source(src: str, is_chokepoint: bool, hosts: set) -> list:
    if is_chokepoint:
        return []
    tree = ast.parse(src)
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
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
        if _expr_is_llm_origin(url):
            offenders.append(call.lineno)
            continue
        if isinstance(url, ast.Name):
            cands = [f for f in funcs
                     if f.lineno <= call.lineno <= (f.end_lineno or f.lineno)]
            if cands:
                enc = min(cands, key=lambda f: (f.end_lineno or f.lineno) - f.lineno)
                if url.id in _llm_origin_names(enc):
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


def test_guard_flags_a_cfg_url_on_any_receiver():
    src = (
        "async def f(self, cfg):\n"
        "    return await self.client.post(cfg['url'], json={})\n"
    )
    assert scan_source(src, False, _HOSTS), "self.client.post(cfg['url']) must FAIL"


def test_guard_passes_a_non_llm_variable_url():
    # A variable URL with no LLM-config origin (e.g. a search API) passes.
    src = (
        "async def f(client):\n"
        "    url = 'https://search.example/api'\n"
        "    return await client.post(url, json={})\n"
    )
    assert scan_source(src, False, _HOSTS) == []
