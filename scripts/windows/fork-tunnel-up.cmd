@echo off
REM fork-tunnel-up.cmd - one-click "bring the Fork chat back online".
REM Starts cloudflared, waits, pushes the new URL to Render, returns.
REM Double-click to run. Output stays on-screen so you can see success/failure.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0fork-tunnel-start.ps1"
echo.
echo Waiting ~10 seconds for cloudflared to print its URL...
timeout /t 10 /nobreak >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0fork-tunnel-update.ps1"
echo.
echo Done. Chat will be live again ~90 seconds after Render redeploys.
pause
