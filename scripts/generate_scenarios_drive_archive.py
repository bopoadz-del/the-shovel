#!/usr/bin/env python3
"""Standalone scenario generator for the `drive_archive` corpus.

Why standalone (not generate_training_scenarios.py): that script reads from
``doc_index``, which the Drive archive indexer bypasses (it writes directly
to ``data/rag/vectors.db`` via ``vector_store.upsert_chunks``). Seeding
``doc_index`` from 139k chunks just to feed the existing script is wasteful;
this runner reads chunks directly from the vector store, samples them, and
calls Ollama with the same Q&A prompt shape so the output schema matches
``training_scenarios_rag_grounded.jsonl`` / the trainer's expectations.

Hard rules baked in:
- All LLM calls route through Ollama at http://127.0.0.1:11434. No
  DeepSeek/Anthropic/Groq fallback.
- Primary model: qwen3-coder:480b-cloud. Fallback: qwen2.5:7b-instruct if
  the first 5 cloud calls fail or rate-limit.
- 500 chunks sampled across docs (cap 3/doc, then stratify by discipline).
- Skip chunks <250 chars. Drop unparseable / generic LLM responses.
- Save progress every 50 successful rows to a state file so a crash
  mid-run resumes cleanly.
- Wall-clock budget: 40 minutes.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sqlite3
import sys
import time

# Force UTF-8 on stdout/stderr so we don't crash on the occasional non-cp1252
# char from chunk paths or log lines on Windows.
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Repo-root + DATA_DIR resolution (same pattern as smoke_hybrid_retrieval /
# _probe_hybrid_vs_vector.py — cwd-independent on Linux and Windows).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DATA_DIR = os.path.abspath(os.environ.setdefault("DATA_DIR", os.path.join(_REPO, "data")))

# ── Config ────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://127.0.0.1:11434"
PRIMARY_MODEL = "qwen3-coder:480b-cloud"
FALLBACK_MODEL = "qwen2.5:7b-instruct"
VECTORS_DB = os.path.join(_DATA_DIR, "rag", "vectors.db")
OUT_PATH = os.path.join(_DATA_DIR, "learning", "training_scenarios_drive_archive.jsonl")
STATE_PATH = os.path.join(_DATA_DIR, "logs", "scenario_gen_state_drive_archive.json")
TARGET_SCENARIOS = 500
PER_DOC_CAP = 3
MIN_CHUNK_CHARS = 250
WALL_CLOCK_BUDGET_SEC = 40 * 60  # 40 minutes
SAVE_EVERY = 50

random.seed(42)

# ── Discipline classification (folder keyword → bucket) ───────────────────
DISCIPLINE_RULES: List[Tuple[str, str]] = [
    # order matters — first hit wins
    (r"structural|\\ST\\|\\ST-|\bSTR\b", "structural"),
    (r"street\s*light|lighting|\\LI-|\\LI\\", "lighting"),
    (r"\\MEP\\|mechanical|electrical|plumbing|HVAC", "mep"),
    (r"contract\s*docs?|PMC|PSA|signed\s*contract", "contract"),
    (r"BOQ|bill\s*of\s*quantit|pricing|tender|priced", "boq"),
    (r"specification|\bspec\b|\\SPEC\\", "spec"),
    (r"schedule|programme|baseline\s*p6", "schedule"),
    (r"geotech|borehole|BH-\d+|soil\s*invest", "geotech"),
    (r"survey|topograph|level\s*plan", "survey"),
    (r"drainage|storm\s*water|sewerage|\\DR\\", "drainage"),
    (r"road|highway|pavement|asphalt", "roads"),
    (r"landscape|hardscape|softscape|irrigation", "landscape"),
    (r"architectur|\\AR\\|\\AR-", "architecture"),
    (r"drawing|drwg|\\DWG-|design\s*drawings", "drawings"),
    (r"safety|HSE|risk\s*assessment|method\s*statement", "hse"),
    (r"report|study|analysis", "report"),
]


def classify_discipline(source_path: str) -> str:
    for pattern, bucket in DISCIPLINE_RULES:
        if re.search(pattern, source_path, re.IGNORECASE):
            return bucket
    return "other"


# ── Sampling ──────────────────────────────────────────────────────────────
def sample_chunks() -> List[Dict[str, object]]:
    print(f"[sample] opening {VECTORS_DB}", flush=True)
    con = sqlite3.connect(VECTORS_DB)
    cur = con.cursor()
    # Pull (doc_id, chunk_index, text) for all drive_archive chunks ≥ 250 chars.
    cur.execute(
        "SELECT doc_id, chunk_index, text FROM chunks "
        "WHERE project_id='drive_archive' AND length(text) >= ?",
        (MIN_CHUNK_CHARS,),
    )
    by_doc: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for doc_id, chunk_index, text in cur:
        by_doc[doc_id].append((chunk_index, text))
    con.close()

    print(f"[sample] {len(by_doc)} docs with >=1 chunk >={MIN_CHUNK_CHARS} chars", flush=True)

    # For each doc, pick first / middle / last chunk (or random if ≤3 total).
    capped: List[Dict[str, object]] = []
    for doc_id, chunks in by_doc.items():
        chunks.sort(key=lambda r: r[0])
        n = len(chunks)
        if n <= PER_DOC_CAP:
            pick = chunks
        else:
            pick = [chunks[0], chunks[n // 2], chunks[-1]]
        for ci, text in pick:
            source_path = parse_source(text)
            capped.append({
                "doc_id": doc_id,
                "chunk_index": ci,
                "text": text,
                "source_path": source_path,
                "discipline": classify_discipline(source_path),
            })

    print(f"[sample] capped pool: {len(capped)} chunks (<={PER_DOC_CAP}/doc)", flush=True)

    # Stratified down-sample to TARGET_SCENARIOS by discipline.
    if len(capped) <= TARGET_SCENARIOS:
        print(f"[sample] pool <= target, taking all {len(capped)}", flush=True)
        random.shuffle(capped)
        return capped

    by_disc: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for c in capped:
        by_disc[c["discipline"]].append(c)
    n_disc = len(by_disc)
    quota_per = TARGET_SCENARIOS // n_disc
    out: List[Dict[str, object]] = []
    leftovers: List[Dict[str, object]] = []
    for disc, items in by_disc.items():
        random.shuffle(items)
        out.extend(items[:quota_per])
        leftovers.extend(items[quota_per:])
    # Top up from leftovers (random) until we hit TARGET.
    random.shuffle(leftovers)
    out.extend(leftovers[: max(0, TARGET_SCENARIOS - len(out))])
    random.shuffle(out)
    print(f"[sample] stratified across {n_disc} disciplines, final: {len(out)}", flush=True)
    counts = defaultdict(int)
    for c in out:
        counts[c["discipline"]] += 1
    for d, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {d:14s} {n}", flush=True)
    return out


def parse_source(text: str) -> str:
    # text starts with "[source: <path>]\n..."
    m = re.match(r"\[source:\s*([^\]]+)\]", text)
    return m.group(1).strip() if m else ""


# ── Ollama call ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are generating training data for a construction-domain RAG LLM. "
    "Given a chunk from a real construction document, produce ONE Q&A pair: "
    "a realistic engineer question answerable from the chunk + a grounded "
    "answer. Do not invent facts. Return JSON only with keys instruction "
    "and response."
)


def call_ollama(model: str, chunk_text: str, source_path: str, timeout: int = 90) -> Optional[str]:
    user = (
        f"Context (chunk from {source_path}):\n{chunk_text}\n\n"
        'Return JSON only: {"instruction": "...", "response": "..."}'
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 600},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        return (resp.get("message", {}) or {}).get("content")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        print(f"[ollama] ERROR {model}: {exc}", flush=True)
        return None


def parse_qa(raw: str) -> Optional[Dict[str, str]]:
    if not raw:
        return None
    # Strip code fences.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # Extract the first {...} block.
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        # Try to fix common issue: trailing comma, then bail.
        try:
            obj = json.loads(re.sub(r",\s*\}", "}", blob))
        except json.JSONDecodeError:
            return None
    instr = (obj.get("instruction") or "").strip()
    resp = (obj.get("response") or "").strip()
    if len(instr) < 10 or len(resp) < 30:
        return None
    # Reject generic boilerplate.
    lower_q = instr.lower()
    if any(p in lower_q for p in ("what does this say", "summarize this", "what is this about")):
        return None
    return {"instruction": instr, "response": resp}


# ── State / progress ──────────────────────────────────────────────────────
def load_state() -> Dict[str, object]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {"completed_keys": []}


def save_state(state: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def append_row(row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    # Force LLM env (even though we call urllib directly, keep the contract).
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["OLLAMA_URL"] = OLLAMA_URL
    os.environ["LOCAL_LLM_MODEL"] = PRIMARY_MODEL

    t0 = time.time()
    samples = sample_chunks()

    # Resume support: skip keys already written.
    state = load_state()
    done_keys = set(state.get("completed_keys", []))
    if done_keys:
        print(f"[resume] {len(done_keys)} already done — skipping those", flush=True)

    model = PRIMARY_MODEL
    primary_failures = 0
    primary_calls = 0
    fallback_calls = 0
    switched_at: Optional[int] = None
    written = 0
    dropped_parse = 0
    dropped_short_or_generic = 0

    for i, c in enumerate(samples, start=1):
        elapsed = time.time() - t0
        if elapsed > WALL_CLOCK_BUDGET_SEC:
            print(f"[budget] hit {WALL_CLOCK_BUDGET_SEC}s wall — stopping at {i-1}/{len(samples)}", flush=True)
            break

        key = f"{c['doc_id']}:{c['chunk_index']}"
        if key in done_keys:
            continue

        raw = call_ollama(model, c["text"], c["source_path"] or c["doc_id"])
        if model == PRIMARY_MODEL:
            primary_calls += 1
            if raw is None:
                primary_failures += 1
                # Switch to fallback after 5 consecutive cloud failures.
                if primary_failures >= 5 and switched_at is None:
                    print(f"[switch] {primary_failures} primary failures → switching to {FALLBACK_MODEL}", flush=True)
                    model = FALLBACK_MODEL
                    switched_at = i
                continue
            primary_failures = 0  # reset on success
        else:
            fallback_calls += 1
            if raw is None:
                dropped_parse += 1
                continue

        qa = parse_qa(raw)
        if qa is None:
            # Either parse failed or post-filter rejected it.
            if raw and ("{" in raw or "instruction" in raw.lower()):
                dropped_short_or_generic += 1
            else:
                dropped_parse += 1
            continue

        row = {
            "instruction": qa["instruction"],
            "context": c["text"][:600],
            "response": qa["response"],
            "source": f"drive_archive:{c['doc_id']}:{c['chunk_index']}",
            "source_doc_path": c["source_path"],
            "discipline": c["discipline"],
        }
        append_row(row)
        done_keys.add(key)
        written += 1

        if written % SAVE_EVERY == 0:
            state["completed_keys"] = sorted(done_keys)
            save_state(state)
            rate = written / max(1.0, time.time() - t0)
            print(
                f"[progress] {written} written | {i}/{len(samples)} processed | "
                f"model={model} | {rate:.2f} rows/s | "
                f"elapsed={int(time.time()-t0)}s",
                flush=True,
            )

    # Final state save.
    state["completed_keys"] = sorted(done_keys)
    save_state(state)

    total = time.time() - t0
    print("\n── SUMMARY ──", flush=True)
    print(f"  written:          {written}", flush=True)
    print(f"  primary calls:    {primary_calls} ({PRIMARY_MODEL})", flush=True)
    print(f"  fallback calls:   {fallback_calls} ({FALLBACK_MODEL})", flush=True)
    print(f"  switched at:      {switched_at}", flush=True)
    print(f"  dropped (parse):  {dropped_parse}", flush=True)
    print(f"  dropped (qual):   {dropped_short_or_generic}", flush=True)
    print(f"  wall-clock:       {total:.1f}s ({total/60:.1f} min)", flush=True)
    if written:
        print(f"  avg per row:      {total/written:.2f}s", flush=True)
    print(f"  output:           {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
