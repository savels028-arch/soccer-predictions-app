#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/logs"

if [ -f "$ROOT/.env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi

export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$ROOT/service-account.json}"

if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "Missing Firebase service account: $GOOGLE_APPLICATION_CREDENTIALS" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

STAMP="$(date +"%Y%m%d-%H%M%S")"
LOG_FILE="$ROOT/logs/local-pipeline-$STAMP.log"

echo "Running AIBets pipeline locally: $*" | tee "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"

"$PYTHON_BIN" run_pipeline.py "$@" --verbose 2>&1 | tee -a "$LOG_FILE"
