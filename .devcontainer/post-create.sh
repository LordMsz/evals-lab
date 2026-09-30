#!/usr/bin/env bash
set -euo pipefail
cd /workspaces/eval-lab

# Fresh named volumes are created root-owned.
sudo chown vscode:vscode .venv node_modules "$CLAUDE_CONFIG_DIR"
git config --global --add safe.directory /workspaces/eval-lab

make setup
