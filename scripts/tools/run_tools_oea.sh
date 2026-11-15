#!/usr/bin/env bash

# Some tools access Google Earth Engine; ensure this machine has network access,
# for example by enabling a VPN when required.

set -euo pipefail

# Fixed launch setting.
CONFIG_NAME="skillearth.yaml"

# Paths derived from this script location.
PROJECT_ROOT="Path/to/SkillEarth"
LAUNCH_DIR="$PROJECT_ROOT/tool_server/oea/tool_workers/scripts/launch_scripts"
CONFIG_PATH="$LAUNCH_DIR/config/$CONFIG_NAME"

# Start the local OEA tool server.
cd "$LAUNCH_DIR"
exec python start_server_local.py --config "$CONFIG_PATH"
