#!/bin/bash
# ═══════════════════════════════════════════
# ⚽ Soccer Predictions Pro - Launch Script
# ═══════════════════════════════════════════

cd "$(dirname "$0")"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "⚽ Starting Soccer Predictions Pro..."
echo ""

python main.py "$@"
