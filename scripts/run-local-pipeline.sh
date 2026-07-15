#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/.runtime"

# Every launchd mode shares this lock. Overlap is expected (for example a
# weekly training run crossing an odds refresh), so the second process exits
# successfully instead of producing a failure notification.
LOCK_DIR="$ROOT/.runtime/pipeline.lock"
LOCK_PID="$LOCK_DIR/pid"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  existing_pid=""
  if [ -f "$LOCK_PID" ]; then
    existing_pid="$(tr -dc '0-9' < "$LOCK_PID")"
  fi
  if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "AIBets pipeline already running (pid $existing_pid); skipping overlapping job."
    exit 0
  fi

  # Recover a lock left by an interrupted process, then retry atomically.
  rm -f "$LOCK_PID"
  rmdir "$LOCK_DIR" 2>/dev/null || true
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "AIBets pipeline lock is busy; skipping overlapping job."
    exit 0
  fi
fi
printf '%s\n' "$$" > "$LOCK_PID"

cleanup_lock() {
  current_pid=""
  if [ -f "$LOCK_PID" ]; then
    current_pid="$(tr -dc '0-9' < "$LOCK_PID")"
  fi
  if [ "$current_pid" = "$$" ]; then
    rm -f "$LOCK_PID"
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}
trap cleanup_lock EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM

if [ -f "$ROOT/.env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi

export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$ROOT/service-account.json}"

# The Cloudflare ingest credential lives in macOS Keychain rather than in the
# repository or runtime env file.  Never print the value to stdout/logs.
if [ -z "${AIBETS_CACHE_SYNC_SECRET:-}" ] && command -v security >/dev/null 2>&1; then
  AIBETS_CACHE_SYNC_SECRET="$(
    security find-generic-password -w -a "$USER" -s aibets-cache-sync 2>/dev/null || true
  )"
  if [ -n "$AIBETS_CACHE_SYNC_SECRET" ]; then
    export AIBETS_CACHE_SYNC_SECRET
  fi
fi

if [ -z "${AIBETS_CACHE_SYNC_SECRET:-}" ]; then
  echo "Missing AIBets public-cache sync credential in environment or macOS Keychain." >&2
  exit 1
fi

if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "Missing Firebase service account: $GOOGLE_APPLICATION_CREDENTIALS" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

if [ "${1:-}" = "full" ]; then
  shift
fi

STAMP="$(date +"%Y%m%d-%H%M%S")"
LOG_FILE="$ROOT/logs/local-pipeline-$STAMP.log"

echo "Running AIBets pipeline locally: $*" | tee "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"

train_requested=false
for arg in "$@"; do
  if [ "$arg" = "--train" ]; then
    train_requested=true
    break
  fi
done

if [ "$train_requested" = true ]; then
  year="$(date +%Y)"
  month="$((10#$(date +%m)))"
  season="$year"
  if [ "$month" -lt 8 ]; then
    season="$((year - 1))"
  fi
  echo "Refreshing Football-Data cache for season $season/$((season + 1)) before training." | tee -a "$LOG_FILE"
  if ! "$PYTHON_BIN" scripts/refresh-football-data-cache.py --season "$season" 2>&1 | tee -a "$LOG_FILE"; then
    echo "Football-Data refresh failed; refusing to train on a stale or incomplete cache." | tee -a "$LOG_FILE" >&2
    exit 1
  fi
  echo "Rebuilding the point-in-time Strategy Zoo artifact." | tee -a "$LOG_FILE"
  if ! "$PYTHON_BIN" -m research.run_pattern_zoo 2>&1 | tee -a "$LOG_FILE"; then
    echo "Strategy Zoo rebuild failed; preserving the last validated artifact and failing the train job." | tee -a "$LOG_FILE" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" run_pipeline.py "$@" --verbose 2>&1 | tee -a "$LOG_FILE"
