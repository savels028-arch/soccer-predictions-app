#!/usr/bin/env bash
set -euo pipefail

LAUNCH_DIR="$HOME/Library/LaunchAgents"

for label in \
  dk.aibets.pipeline.full \
  dk.aibets.pipeline.evaluate \
  dk.aibets.pipeline.odds \
  dk.aibets.pipeline.train
do
  plist="$LAUNCH_DIR/$label.plist"
  if [ -f "$plist" ]; then
    launchctl unload "$plist" >/dev/null 2>&1 || true
    rm -f "$plist"
    echo "Removed $label"
  fi
done
