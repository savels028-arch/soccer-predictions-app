#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${AIBETS_RUNTIME_DIR:-$HOME/AIBets/soccer-predictions-app}"

mkdir -p "$DEST"

# Copy source/config while preserving runtime-owned credentials, models,
# databases, caches and logs.  There is deliberately no --delete.
rsync -a \
  --exclude '.git/' \
  --exclude '.env*' \
  --exclude 'service-account.json' \
  --exclude 'venv/' \
  --exclude 'node_modules/' \
  --exclude 'deploy/node_modules/' \
  --exclude 'deploy/.next/' \
  --exclude 'deploy/.open-next/' \
  --exclude 'deploy/.wrangler/' \
  --exclude 'data/' \
  --exclude 'logs/' \
  "$ROOT/" "$DEST/"

# The international model is a small, checksum-verified runtime artifact, not
# mutable pipeline history. Keep it in sync while preserving every other
# runtime-owned data/cache/database file.
if [ -d "$ROOT/data/international" ]; then
  mkdir -p "$DEST/data/international"
  rsync -a "$ROOT/data/international/" "$DEST/data/international/"
fi

# The public Strategy Zoo is rebuilt inside the runtime during weekly train
# jobs.  Keep its validator-owned, network-free Football-Data source complete
# without touching any other mutable cache, model or credential directory.
if [ -d "$ROOT/data/cache/football_data_csv" ]; then
  mkdir -p "$DEST/data/cache/football_data_csv"
  rsync -a \
    "$ROOT/data/cache/football_data_csv/" \
    "$DEST/data/cache/football_data_csv/"
fi

# Strategy Zoo is a compact, validator-owned public research artifact.  Copy
# only this immutable file, never the mutable backtest/model data directory.
if [ -f "$ROOT/data/strategy_zoo_public.json" ]; then
  mkdir -p "$DEST/data"
  rsync -a "$ROOT/data/strategy_zoo_public.json" "$DEST/data/strategy_zoo_public.json"
fi
if [ -f "$ROOT/data/strategy_zoo_public.sha256" ]; then
  mkdir -p "$DEST/data"
  rsync -a "$ROOT/data/strategy_zoo_public.sha256" "$DEST/data/strategy_zoo_public.sha256"
fi

echo "AIBets source synced to: $DEST"
echo "Runtime credentials, mutable data, models and logs were preserved."
echo "The validated international model artifact was refreshed."
echo "The canonical Football-Data research cache was refreshed without deleting runtime files."
echo "The validated Strategy Zoo artifact was refreshed when present."
