#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=configs/main.yaml
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
exec bash "$PROJECT_DIR/run.sh" --config "$CONFIG" evaluate "$@"
