#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MARKER="$SCRIPT_DIR/.venv/.installed"

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/.venv"
fi

# Only run pip install if marker file doesn't exist (first setup)
if [ ! -f "$MARKER" ]; then
    echo "Installing dependencies..."
    "$SCRIPT_DIR/.venv/bin/pip" install -q ollama rich prompt-toolkit schedule plyer && touch "$MARKER"
fi

# Kill any other running instances of superboof cli.py
pkill -9 -f "$SCRIPT_DIR/cli.py" 2>/dev/null

"$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/cli.py"
