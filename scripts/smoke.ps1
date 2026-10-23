#!/usr/bin/env pwsh
# scripts/smoke.ps1 - pre-deploy smoke: the 10-run deliverable discriminator.
#
# Drives fork_cli against a live Fork instance ten times with a deliverable
# (tool-calling) prompt and prints a per-run summary + a single verdict line.
# This is the tripwire for the reasoning-field 400 class of bug: a deliverable
# turn that errors produces an empty answer (answer_chars=0), and the tool path
# is exactly what breaks - so we require both a full success rate AND that the
# tool path actually fired on several runs.
#
# VERDICT: PASS iff all N runs succeed (answer_chars >= MinChars, default 1500)
#          AND >= ceil-ish(N*0.3) runs contain a tool_call/tool_result pair
#          (1 for N=3, 3 for N=10). Otherwise FAIL with a nonzero exit code.
#          Each run also prints the served model, so a silent fallback (e.g.
#          Kimi K2 -> Ollama) is visible instead of hidden behind a PASS.
#
# Credentials come from the environment ONLY - never hardcode a key here:
#   $env:FORK_API_KEY = "<master key>"          # preferred
#   -- or -- $env:FORK_EMAIL + $env:FORK_PASSWORD
#   -- or -- $env:FORK_TOKEN
# Target defaults to prod (fork_cli's own default); override with $env:FORK_BASE_URL.
#
# Usage:
#   $env:FORK_API_KEY="..."; ./scripts/smoke.ps1
#   ./scripts/smoke.ps1 -Project master_corpus -Runs 10
[CmdletBinding()]
param(
    [string]$Project  = $(if ($env:FORK_PROJECT) { $env:FORK_PROJECT } else { "master_corpus" }),
    [string]$Base     = $(if ($env:FORK_BASE_URL) { $env:FORK_BASE_URL } else { "" }),
    # Default 3 = routine pre-deploy smoke (keeps LLM quota use light). Release
    # smokes use -Runs 10. Verdict thresholds scale with N (see below).
    [int]$Runs        = $(if ($env:SMOKE_RUNS) { [int]$env:SMOKE_RUNS } else { 3 }),
    # A run only counts as success if answer_chars >= MinChars. Guards against a
    # degenerate short answer (a 93-char "checklist" passed the old >0 oracle).
    [int]$MinChars    = $(if ($env:MIN_ANSWER_CHARS) { [int]$env:MIN_ANSWER_CHARS } else { 1500 }),
    [string]$Message  = "generate a commissioning checklist for the MV substation"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"   # Windows console chokes on symbols (Ohm etc.) in checklists

# --- credential guard: env only, never hardcoded ---
$haveKey   = [bool]$env:FORK_API_KEY
$haveLogin = [bool]$env:FORK_EMAIL -and [bool]$env:FORK_PASSWORD
$haveTok   = [bool]$env:FORK_TOKEN
if (-not ($haveKey -or $haveLogin -or $haveTok)) {
    Write-Error "No credentials in env. Set FORK_API_KEY (or FORK_EMAIL + FORK_PASSWORD, or FORK_TOKEN) before running smoke."
    exit 2
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$cli = Join-Path $scriptDir "fork_cli.py"
if (-not (Test-Path $cli)) { Write-Error "fork_cli.py not found next to smoke.ps1 ($cli)"; exit 2 }

$baseArgs = @()
if ($Base) { $baseArgs += @("--base", $Base) }

$successes = 0
$toolRuns  = 0

Write-Host "Pre-deploy smoke: $Runs runs, project=$Project, min_answer_chars=$MinChars" -ForegroundColor Cyan
Write-Host ("message: {0}" -f $Message)
Write-Host ("=" * 72)

for ($i = 1; $i -le $Runs; $i++) {
    # PS 5.1: 2>&1 on a native exe wraps each stderr line in an ErrorRecord;
    # with EAP=Stop the first one (fork_cli's "[auth] logged in" on the
    # email/password path) becomes a TERMINATING error and kills the loop.
    # Relax EAP for the invocation only — stderr text still lands in $out.
    $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $out = & python $cli @baseArgs chat $Message --project $Project --events 2>&1 | Out-String
    $ErrorActionPreference = $eap

    $m = [regex]::Match($out, "total=([\d.]+)s\s+first_token=([\d.\-]+s?|-)\s+events=(\d+)\s+answer_chars=(\d+)")
    $total = if ($m.Success) { [double]$m.Groups[1].Value } else { $null }
    $ftok  = if ($m.Success) { $m.Groups[2].Value } else { $null }
    $chars = if ($m.Success) { [int]$m.Groups[4].Value } else { 0 }
    $mm    = [regex]::Match($out, "model=(\S+)")
    $model = if ($mm.Success) { $mm.Groups[1].Value } else { "?" }

    # A deliverable run = the orchestrator executed a REAL action, via EITHER
    # the agent tool-loop (TOOL_CALL + TOOL_RESULT) OR predefined dispatch
    # (END event with mode:predefined + a non-empty plan_steps). The latter was
    # invisible to the old detector, so a healthy predefined deliverable
    # (e.g. commissioning_checklist) wrongly read as tool=n.
    $hasTool = (($out -match "TOOL_CALL") -and ($out -match "TOOL_RESULT")) -or `
               (($out -match '"mode":\s*"predefined"') -and ($out -match '"plan_steps":\s*\[\s*"[A-Za-z]'))
    # Success requires a substantive answer, not just non-empty — a degenerate
    # short answer must FAIL.
    $ok = $m.Success -and ($chars -ge $MinChars)

    if ($ok)      { $successes++ }
    if ($hasTool) { $toolRuns++ }

    $status   = if ($ok) { "OK  " } else { "FAIL" }
    $toolFlag = if ($hasTool) { "tool=Y" } else { "tool=n" }
    $totalTxt = if ($total) { "{0,6:N2}" -f $total } else { "     ?" }
    $ftokTxt  = if ($ftok)  { "{0,7}" -f $ftok }     else { "      ?" }
    Write-Host ("run{0,2}  {1}  {2}  total={3}s  first_token={4}  answer_chars={5,6}  model={6}" -f `
        $i, $status, $toolFlag, $totalTxt, $ftokTxt, $chars, $model)
}

Write-Host ("=" * 72)
# Tool-run threshold scales with N: 1 for the routine N=3 smoke, 3 for N=10.
$toolNeeded = [math]::Max(1, [math]::Floor($Runs * 0.3))
$pass = ($successes -eq $Runs) -and ($toolRuns -ge $toolNeeded)
if ($pass) {
    Write-Host ("VERDICT: PASS  ({0}/{1} succeeded [answer_chars>={2}], {3} tool runs >= {4})" -f $successes, $Runs, $MinChars, $toolRuns, $toolNeeded) -ForegroundColor Green
    exit 0
} else {
    Write-Host ("VERDICT: FAIL  ({0}/{1} succeeded [need {1}/{1} with answer_chars>={2}], {3} tool runs [need >={4}])" -f $successes, $Runs, $MinChars, $toolRuns, $toolNeeded) -ForegroundColor Red
    exit 1
}
