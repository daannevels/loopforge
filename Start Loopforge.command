#!/bin/bash
# Loopforge launcher for macOS — double-click this file.
# First run installs uv (Python manager) and the app's dependencies; later runs start in seconds.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "First start: installing uv (one time only)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Could not install uv. Check your internet connection."; read -r -p "Press Enter to close"; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "Preparing Loopforge (first start can take a minute)…"
uv sync --quiet || { echo "Setup failed (see above)."; read -r -p "Press Enter to close"; exit 1; }
uv run loopforge ui
