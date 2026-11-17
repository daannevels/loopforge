@echo off
REM Loopforge launcher for Windows - double-click this file.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo First start: installing uv ^(one time only^)...
  powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
echo Preparing Loopforge (first start can take a minute)...
uv sync --quiet
if errorlevel 1 (
  echo Setup failed - see above.
  pause
  exit /b 1
)
uv run loopforge ui
pause
