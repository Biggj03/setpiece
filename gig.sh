#!/usr/bin/env bash
# One-command launcher for the Resolume control rig (Linux/macOS).
# Run ./gig.sh — close the terminal or Ctrl-C to stop.
# (For the AVL MXE / Debian 12 migration.)
cd "$(dirname "$0")" || exit 1
exec python3 gig.py "$@"
