# fork-tunnel-start.ps1 — start cloudflared quick-tunnel pointed at local Ollama.
#
# Runs cloudflared as a detached, hidden-window background process whose stderr
# (cloudflared writes its INF logs to stderr) lands in $LogDir\cloudflared.log.
# Returns immediately so the caller (a Scheduled Task or a one-shot .cmd) can
# move on. A second script (fork-tunnel-update.ps1) reads the log a few seconds
# later to grab the URL and push it to Render.

$CloudflaredExe = (Join-Path $env:USERPROFILE 'Downloads\ollama-setup\cloudflared.exe')
$OllamaUrl      = 'http://localhost:11434'
$LogDir         = "$env:LOCALAPPDATA\fork-tunnel"
$TunnelLogFile  = "$LogDir\cloudflared.log"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Kill any prior cloudflared so we don't end up with stale URLs.
Get-Process -Name cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

if (Test-Path $TunnelLogFile) { Remove-Item $TunnelLogFile -Force }

$argList = @('tunnel', '--url', $OllamaUrl, '--http-host-header', 'localhost')
$proc = Start-Process -FilePath $CloudflaredExe -ArgumentList $argList `
    -RedirectStandardError $TunnelLogFile -RedirectStandardOutput "$LogDir\cf-stdout.log" `
    -PassThru -WindowStyle Hidden

$proc.Id | Out-File "$LogDir\cloudflared.pid" -Encoding ASCII
Write-Host "started cloudflared pid=$($proc.Id) log=$TunnelLogFile"
