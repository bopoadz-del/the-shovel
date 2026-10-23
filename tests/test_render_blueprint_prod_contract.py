"""Production blueprint must match the live Kimi + Groq ladder.

render.yaml is the reviewable production surface. A committed Ollama intent
model, an enabled ingest worker with no worker service, or a 90s stream
timeout would ship as "Apply Blueprint" and break chat / uploads. These
tests pin the contract that code review cannot see in a comment.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _web_env() -> dict[str, dict]:
    blueprint = yaml.safe_load((REPO / "render.yaml").read_text(encoding="utf-8"))
    web = next(s for s in blueprint["services"] if s.get("name") == "the-fork")
    out: dict[str, dict] = {}
    for item in web.get("envVars") or []:
        key = item.get("key")
        if key:
            out[key] = item
    return out


def test_blueprint_does_not_pin_an_ollama_intent_model():
    env = _web_env()
    intent = env.get("ORCHESTRATOR_INTENT_MODEL") or {}
    value = (intent.get("value") or "").strip()
    assert ":" not in value, (
        f"ORCHESTRATOR_INTENT_MODEL={value!r} is an Ollama-style id. Cloud "
        "prod is Kimi; Moonshot 400s that name every chat turn and predefined "
        "schedule routing never fires. Leave it unset (provider default) or "
        "pin a model the PRIMARY provider serves."
    )


def test_blueprint_lists_openrouter_keys_unsynced():
    """OpenRouter is a first-class cloud provider; keys stay dashboard-only."""
    env = _web_env()
    for key in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
        item = env.get(key)
        assert item is not None, f"{key} missing from render.yaml"
        assert item.get("sync") is False, f"{key} must be sync:false"
        assert not (item.get("value") or "").strip(), (
            f"{key} must not pin a committed value"
        )


def test_blueprint_lists_deepseek_keys_unsynced():
    """DeepSeek is a first-class cloud provider; keys stay dashboard-only."""
    env = _web_env()
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"):
        item = env.get(key)
        assert item is not None, f"{key} missing from render.yaml"
        assert item.get("sync") is False, f"{key} must be sync:false"
        assert not (item.get("value") or "").strip(), (
            f"{key} must not pin a committed value"
        )


def test_blueprint_does_not_commit_ollama_as_cloud_provider():
    env = _web_env()
    for key in ("LLM_PROVIDER", "LLM_FALLBACK_PROVIDER"):
        value = ((env.get(key) or {}).get("value") or "").strip().lower()
        assert value != "ollama", (
            f"{key}={value!r} in render.yaml. Cloud ladder is Kimi primary + "
            "Groq fallback; Ollama is on-prem only."
        )


def test_blueprint_ingest_worker_stays_off():
    env = _web_env()
    item = env.get("INGEST_WORKER_ENABLED")
    assert item is not None, (
        "INGEST_WORKER_ENABLED must be pinned false — a leftover dashboard "
        "true enqueues uploads onto a queue nobody drains."
    )
    value = str(item.get("value") or "").strip().lower()
    assert value in {"", "0", "false", "no"}, (
        f"INGEST_WORKER_ENABLED={value!r}; no worker service is deployed."
    )


def test_blueprint_public_base_url_is_the_client_hostname():
    env = _web_env()
    item = env.get("PUBLIC_BASE_URL")
    assert item is not None, "PUBLIC_BASE_URL missing from render.yaml"
    assert item.get("value") == "https://theshovel.ai"


def test_blueprint_chat_stream_timeout_covers_kimi_reasoning():
    env = _web_env()
    item = env.get("CHAT_STREAM_TIMEOUT_SECONDS")
    assert item is not None, "CHAT_STREAM_TIMEOUT_SECONDS missing from render.yaml"
    seconds = float(item["value"])
    assert seconds >= 240, (
        f"CHAT_STREAM_TIMEOUT_SECONDS={seconds} is below the 240s Kimi "
        "reasoning-burst floor (PLATFORM_HEALTH_REPORT)."
    )


def test_blueprint_auto_deploy_waits_for_ci_checks():
    """Auto-deploy on the-fork must wait for GitHub CI, not fire on commit.

    612a2d89 (#436) is the standing hole: required checks `pip-audit` and
    `test-postgres` went red on main while Render auto-deploy was still
    on-commit, so a red SHA could already be live.

    The gate is Render's native `autoDeployTrigger: checksPass` ("After CI
    Checks Pass"). Render waits for this SHA's GitHub check runs and skips
    the deploy if any conclusion is a failure or if zero checks are
    detected. success / neutral / skipped all count as passed.

    Deprecated `autoDeploy: true` is equivalent to `commit` (no CI gate).
    If both keys are present, `autoDeployTrigger` wins — so a leftover
    `autoDeploy: true` next to `checksPass` is harmless, but a lone
    `autoDeploy: true` (or `autoDeployTrigger: commit`) re-opens the hole.
    Pin the trigger and reject the on-commit forms.

    This file does not change the running service until Dashboard
    Auto-Deploy is set to "After CI Checks Pass" (prefer that one toggle)
    or the blueprint is applied. The test pins the intended live value so
    Apply Blueprint cannot silently revert to on-commit.
    """
    blueprint = yaml.safe_load((REPO / "render.yaml").read_text(encoding="utf-8"))
    web = next(s for s in blueprint["services"] if s.get("name") == "the-fork")
    trigger = web.get("autoDeployTrigger")
    assert trigger == "checksPass", (
        f"the-fork autoDeployTrigger is {trigger!r}; expected 'checksPass' "
        "so a red required GitHub check cannot auto-deploy. Dashboard "
        "equivalent: Auto-Deploy → After CI Checks Pass."
    )
    deprecated = web.get("autoDeploy")
    if deprecated is not None:
        assert str(deprecated).lower() not in {"true", "yes"}, (
            "autoDeploy: true is on-commit (no CI gate). Remove it or set "
            "autoDeployTrigger: checksPass, which takes precedence."
        )
    text = (REPO / "render.yaml").read_text(encoding="utf-8")
    commit_triggers = re.findall(
        r"^\s*autoDeployTrigger:\s*['\"]?(commit)['\"]?",
        text,
        re.MULTILINE,
    )
    assert not commit_triggers, (
        "autoDeployTrigger: commit is on-commit auto-deploy; that ships a "
        "red main SHA. Use checksPass."
    )


# The corpus is embedded with this model and every row carries the stamp.
# VectorStore._verify_embedding_identity refuses a namespace whose stamp
# disagrees with the configured model, so this is not a preference -- it is
# the identity of 172,809 existing vectors.
LIVE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def test_blueprint_embedding_model_matches_the_corpus():
    """render.yaml declared minishlab/potion-base-8M while production ran
    BAAI/bge-small-en-v1.5 and every chunk was stamped with it:

        BAAI/bge-small-en-v1.5   dim 384   normalized True   172,809 chunks

    Applying the blueprint would have set the wrong model, the identity guard
    would have refused the namespace, and retrieval would have gone down
    across the whole corpus until every chunk was re-embedded. The prose above
    that key in render.yaml describes this mechanism exactly -- and the value
    underneath it was still wrong, because nothing tested it.

    Changing this constant is a re-embedding decision, not a config edit.
    """
    env = _web_env()
    item = env.get("RAG_EMBEDDING_MODEL")
    assert item is not None, "RAG_EMBEDDING_MODEL missing from render.yaml"
    assert item.get("value") == LIVE_EMBEDDING_MODEL, (
        f"render.yaml declares {item.get('value')!r} but the live corpus is "
        f"embedded with {LIVE_EMBEDDING_MODEL!r}. Applying this blueprint "
        "would take retrieval down for every chunk."
    )


def _dockerfile_arg_embedding_model() -> str:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(
        r'^ARG RAG_EMBEDDING_MODEL=(?:"([^"]+)"|(\S+))',
        text,
        re.MULTILINE,
    )
    assert match, "Dockerfile is missing ARG RAG_EMBEDDING_MODEL"
    return match.group(1) or match.group(2)


def test_dockerfile_arg_default_matches_the_corpus():
    """Unset --build-arg must not bake potion-base-8M.

    CI and a local `docker build .` do not pass the ARG. The previous
    default (minishlab/potion-base-8M) meant a bake without the live
    model name shipped the wrong weights. HF_HUB_OFFLINE=1 then cannot
    recover: LocalEntryNotFoundError → RAG_INDEX_FAILED.
    """
    assert _dockerfile_arg_embedding_model() == LIVE_EMBEDDING_MODEL


def _service_env(name: str) -> dict[str, dict]:
    blueprint = yaml.safe_load((REPO / "render.yaml").read_text(encoding="utf-8"))
    svc = next(s for s in blueprint["services"] if s.get("name") == name)
    out: dict[str, dict] = {}
    for item in svc.get("envVars") or []:
        key = item.get("key")
        if key:
            out[key] = item
    return out


def test_blueprint_hf_home_stays_on_baked_weights():
    """Dashboard HF_HOME=/app/data/hf-cache overrode onto the empty volume.

    Baked weights live at /opt/hf (outside the /app/data mount). A cache
    path on the volume is empty after every new disk / clear-cache deploy
    and produces 40× RAG_INDEX_FAILED / LocalEntryNotFoundError even when
    the image baked BAAI/bge-small-en-v1.5.
    """
    text = (REPO / "render.yaml").read_text(encoding="utf-8")
    assert "HF_HOME" in text
    assert "/opt/hf" in text
    for name in ("the-fork", "the-fork-ingest"):
        item = _service_env(name).get("HF_HOME")
        assert item is not None, f"HF_HOME missing from {name}"
        assert item.get("value") == "/opt/hf", (
            f"{name} HF_HOME={item.get('value')!r}; must stay /opt/hf so "
            "the volume mount cannot hide the baked embedder."
        )
