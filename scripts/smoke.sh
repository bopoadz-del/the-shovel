#!/usr/bin/env bash
# scripts/smoke.sh - pre-deploy smoke: the deliverable discriminator.
#
# Drives fork_cli against a live Fork instance N times with a deliverable
# (tool-calling) prompt and prints a per-run summary + a single verdict line.
# This is the tripwire for the reasoning-field 400 class of bug AND for silent
# provider fallbacks: a deliverable turn that errors or degenerates produces a
# short/empty answer, and the tool path is exactly what breaks.
#
# VERDICT: PASS iff all N runs succeed (answer_chars >= MIN_ANSWER_CHARS,
#          default 1500) AND >= max(1, floor(N*0.3)) runs executed a REAL
#          deliverable (1 for N=3, 3 for N=10) — either via the agent tool-loop
#          (tool_call/tool_result) OR via predefined dispatch (mode:predefined
#          with a non-empty plan_steps). Otherwise FAIL
#          with a nonzero exit code. Each run prints the served model, so a
#          silent fallback (e.g. Kimi K2 -> Ollama) is visible, not hidden.
#
# Credentials come from the environment ONLY - never hardcode a key here:
#   export FORK_API_KEY="<master key>"           # preferred
#   -- or -- FORK_EMAIL + FORK_PASSWORD
#   -- or -- FORK_TOKEN
# Target defaults to prod (fork_cli's own default); override with FORK_BASE_URL.
#
# Usage:
#   FORK_API_KEY="..." ./scripts/smoke.sh                 # routine: 3 runs
#   SMOKE_RUNS=10 FORK_API_KEY="..." ./scripts/smoke.sh   # release: 10 runs
#   ./scripts/smoke.sh master_corpus
#
# Long release runs may exceed a shell/CI timeout. Use --background <file> to
# detach and write results to a file; the caller polls the file for completion.
set -u

PROJECT="${1:-${FORK_PROJECT:-master_corpus}}"
BACKGROUND_FILE=""
if [ "${1:-}" = "--background" ]; then
  BACKGROUND_FILE="${2:-}"
  if [ -z "$BACKGROUND_FILE" ]; then
    echo "Usage: ./scripts/smoke.sh --background <output-file> [project]" >&2
    exit 2
  fi
  shift 2
  PROJECT="${1:-${FORK_PROJECT:-master_corpus}}"
fi

RUNS="${SMOKE_RUNS:-3}"                       # routine default; release uses 10
MIN_CHARS="${MIN_ANSWER_CHARS:-1500}"         # a degenerate short answer FAILs
MESSAGE="${SMOKE_MESSAGE:-generate a commissioning checklist for the MV substation}"
export PYTHONIOENCODING=utf-8

# --- background mode: detach, run, write to file, print PID ---
if [ -n "$BACKGROUND_FILE" ]; then
  nohup bash "$0" "$PROJECT" > "$BACKGROUND_FILE" 2>&1 &
  pid=$!
  echo "smoke backgrounded (PID $pid) -> $BACKGROUND_FILE"
  exit 0
fi

# --- credential guard: env only, never hardcoded ---
if [ -z "${FORK_API_KEY:-}" ] \
   && { [ -z "${FORK_EMAIL:-}" ] || [ -z "${FORK_PASSWORD:-}" ]; } \
   && [ -z "${FORK_TOKEN:-}" ]; then
  echo "No credentials in env. Set FORK_API_KEY (or FORK_EMAIL + FORK_PASSWORD, or FORK_TOKEN)." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="$SCRIPT_DIR/fork_cli.py"
[ -f "$CLI" ] || { echo "fork_cli.py not found next to smoke.sh ($CLI)" >&2; exit 2; }

# Prefer explicit PYTHON, then repo venv, then system python.
if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
elif [ -f "$SCRIPT_DIR/../.venv/Scripts/python.exe" ]; then
  PYTHON_BIN="$SCRIPT_DIR/../.venv/Scripts/python.exe"
elif [ -f "$SCRIPT_DIR/../.venv/bin/python" ]; then
  PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"
else
  PYTHON_BIN="python"
fi

BASE_ARGS=()
[ -n "${FORK_BASE_URL:-}" ] && BASE_ARGS=(--base "$FORK_BASE_URL")

# Tool-run threshold scales with N: 1 for the routine N=3 smoke, 3 for N=10.
tool_needed=$(( RUNS * 3 / 10 ))
[ "$tool_needed" -lt 1 ] && tool_needed=1

successes=0
tool_runs=0

echo "Pre-deploy smoke: $RUNS runs, project=$PROJECT, min_answer_chars=$MIN_CHARS"
echo "message: $MESSAGE"
printf '=%.0s' $(seq 1 72); echo

for i in $(seq 1 "$RUNS"); do
  conv_id="$("$PYTHON_BIN" -c 'import uuid; print(uuid.uuid4())')"
  out="$("$PYTHON_BIN" "$CLI" "${BASE_ARGS[@]}" chat "$MESSAGE" --project "$PROJECT" --conversation "$conv_id" --events 2>&1)"

  line="$(printf '%s\n' "$out" | grep -oE 'total=[0-9.]+s +first_token=([0-9.]+s?|-) +events=[0-9]+ +answer_chars=[0-9]+' | head -1)"
  chars="$(printf '%s\n' "$line" | grep -oE 'answer_chars=[0-9]+' | grep -oE '[0-9]+$')"
  total="$(printf '%s\n' "$line" | grep -oE 'total=[0-9.]+' | grep -oE '[0-9.]+')"
  ftok="$(printf '%s\n' "$line" | sed -nE 's/.*first_token=([0-9.]+s?|-).*/\1/p')"
  model="$(printf '%s\n' "$out" | grep -oE 'model=[^ ]+' | head -1 | sed 's/model=//')"
  chars="${chars:-0}"
  model="${model:-?}"

  # A "deliverable run" is one where the orchestrator executed a REAL action,
  # by EITHER path:
  #   (a) the agent tool-loop — a TOOL_CALL + TOOL_RESULT pair; or
  #   (b) predefined dispatch — the END event carries mode:predefined with a
  #       non-empty plan_steps (the container action ran via run_workflow).
  # (b) was invisible to the old detector, so a healthy predefined deliverable
  # (e.g. commissioning_checklist -> mode:predefined) wrongly read as tool=n.
  has_tool=n
  if printf '%s\n' "$out" | grep -q 'TOOL_CALL' && printf '%s\n' "$out" | grep -q 'TOOL_RESULT'; then
    has_tool=Y
  elif printf '%s\n' "$out" | grep -q '"mode": "predefined"' \
       && printf '%s\n' "$out" | grep -qE '"plan_steps": \[[[:space:]]*"[A-Za-z]'; then
    has_tool=Y
  fi

  ok="FAIL"
  if [ "${chars:-0}" -ge "$MIN_CHARS" ] 2>/dev/null; then
    ok="OK  "
    successes=$((successes + 1))
  fi
  [ "$has_tool" = "Y" ] && tool_runs=$((tool_runs + 1))

  printf 'run%2d  %s  tool=%s  total=%ss  first_token=%s  answer_chars=%6s  model=%s\n' \
    "$i" "$ok" "$has_tool" "${total:-?}" "${ftok:-?}" "$chars" "$model"
  if [ "$ok" != "OK  " ]; then
    echo "----- run $i cli output (truncated) -----"
    printf '%s\n' "$out" | head -n 40
    echo "----- end run $i -----"
  fi
done

printf '=%.0s' $(seq 1 72); echo
if [ "$successes" -eq "$RUNS" ] && [ "$tool_runs" -ge "$tool_needed" ]; then
  echo "VERDICT: PASS  ($successes/$RUNS succeeded [answer_chars>=$MIN_CHARS], $tool_runs tool runs >= $tool_needed)"
  exit 0
else
  echo "VERDICT: FAIL  ($successes/$RUNS succeeded [need $RUNS/$RUNS with answer_chars>=$MIN_CHARS], $tool_runs tool runs [need >=$tool_needed])"
  exit 1
fi
