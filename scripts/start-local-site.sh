#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$ROOT/deploy"
PORT="${PORT:-3000}"

cd "$DEPLOY_DIR"

if [ ! -d node_modules ]; then
  npm install
fi

export NEXT_PUBLIC_LOCAL_PRIVATE_MODE="${NEXT_PUBLIC_LOCAL_PRIVATE_MODE:-true}"

echo "Starting AIBets locally"
echo "URL: http://127.0.0.1:$PORT"
echo "Local private mode: $NEXT_PUBLIC_LOCAL_PRIVATE_MODE"
echo

npx next dev -H 127.0.0.1 -p "$PORT"
