#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$ROOT/logs"

mkdir -p "$LAUNCH_DIR" "$LOG_DIR"

write_plist() {
  local label="$1"
  local hour="$2"
  local minute="$3"
  local weekday="${4:-}"
  shift 4
  local plist="$LAUNCH_DIR/$label.plist"

  {
    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$label</string>
  <key>WorkingDirectory</key>
  <string>$ROOT</string>
  <key>ProgramArguments</key>
  <array>
    <string>$ROOT/scripts/run-local-pipeline.sh</string>
EOF
    for arg in "$@"; do
      printf '    <string>%s</string>\n' "$arg"
    done
    cat <<EOF
  </array>
  <key>StartCalendarInterval</key>
  <dict>
EOF
    if [ -n "$weekday" ]; then
      cat <<EOF
    <key>Weekday</key>
    <integer>$weekday</integer>
EOF
    fi
    cat <<EOF
    <key>Hour</key>
    <integer>$hour</integer>
    <key>Minute</key>
    <integer>$minute</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/$label.out.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/$label.err.log</string>
</dict>
</plist>
EOF
  } > "$plist"

  launchctl unload "$plist" >/dev/null 2>&1 || true
  launchctl load "$plist"
  echo "Installed $label"
}

# Times are local macOS time.
write_plist "dk.aibets.pipeline.full" 8 30 ""
write_plist "dk.aibets.pipeline.evaluate" 23 15 "" "--evaluate-only"
write_plist "dk.aibets.pipeline.odds" 12 0 "" "--odds-only"
write_plist "dk.aibets.pipeline.train" 7 0 "1" "--train"

echo
echo "Local AIBets scheduler installed."
echo "It runs when this Mac is awake:"
echo "- Full predictions: daily 08:30"
echo "- Odds snapshot: daily 12:00"
echo "- Result evaluation: daily 23:15"
echo "- Retrain: Monday 07:00"
