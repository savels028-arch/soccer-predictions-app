#!/bin/bash
# ═══════════════════════════════════════════
# ⚽ Soccer Predictions Pro - Setup Script
# ═══════════════════════════════════════════

set -e

echo ""
echo "⚽ Soccer Predictions Pro - Setup"
echo "══════════════════════════════════"
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python er ikke installeret!"
    echo "   Installér Python 3.9+ fra https://python.org"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Python fundet: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "📦 Opretter virtual environment..."
$PYTHON_CMD -m venv venv

# Activate
source venv/bin/activate
echo "✅ Virtual environment aktiveret"

# Upgrade pip
echo ""
echo "⬆️  Opgraderer pip..."
pip install --upgrade pip -q

# Install dependencies
echo ""
echo "📥 Installerer dependencies..."
pip install -r requirements.txt

echo ""
echo "══════════════════════════════════"
echo "✅ Setup færdig!"
echo ""
echo "Sådan starter du appen:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Eller kør: ./run.sh"
echo "══════════════════════════════════"
echo ""
