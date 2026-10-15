#!/usr/bin/env bash
set -euo pipefail
CHECKER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$CHECKER_DIR/.venv/bin/python" ]]; then
  python3 -m venv "$CHECKER_DIR/.venv"
fi
if ! cmp -s "$CHECKER_DIR/requirements.txt" "$CHECKER_DIR/.venv/installed-requirements.txt"; then
  "$CHECKER_DIR/.venv/bin/python" -m pip install -r "$CHECKER_DIR/requirements.txt"
  cp "$CHECKER_DIR/requirements.txt" "$CHECKER_DIR/.venv/installed-requirements.txt"
fi
cd "$CHECKER_DIR/.."
exec "$CHECKER_DIR/.venv/bin/python" -m arm_checker "$@"
