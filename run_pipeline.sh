#!/bin/bash
# Run the AIBets prediction pipeline
# Loads Firebase credentials from deploy/.env.local

set -euo pipefail

cd "$(dirname "$0")"

# Load environment variables from deploy/.env.local
if [ -f deploy/.env.local ]; then
    # Handle multi-line and quoted values
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        # Export the variable
        export "$line" 2>/dev/null || true
    done < deploy/.env.local
fi

echo "🚀 Starting AIBets Prediction Pipeline..."
echo "   Firebase project: aibets-5943b"
echo ""

python3 run_pipeline.py "$@"
