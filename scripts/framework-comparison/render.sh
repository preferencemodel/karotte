#!/usr/bin/env bash
# Renders comparison.html to docs/assets/framework-comparison.png with headless Chrome.
set -euo pipefail
cd "$(dirname "$0")"
chrome="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$chrome" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --window-size=1600,900 --virtual-time-budget=5000 \
  --screenshot="$PWD/../../docs/assets/framework-comparison.png" "file://$PWD/comparison.html" 2>/dev/null
