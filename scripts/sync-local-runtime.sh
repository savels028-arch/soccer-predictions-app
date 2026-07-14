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

echo "AIBets source synced to: $DEST"
echo "Runtime credentials, mutable data, models and logs were preserved."
echo "The validated international model artifact was refreshed."
