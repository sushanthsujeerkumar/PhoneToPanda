#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: bash setup.sh [--help]'
  echo 'Create a local Python 3.12 environment and install pinned dependencies.'
  exit 0
fi
if [[ $# -ne 0 ]]; then echo 'Usage: bash setup.sh [--help]' >&2; exit 2; fi
PYTHON=python3.12
if ! command -v "$PYTHON" >/dev/null; then PYTHON=python3; fi
if ! "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
  echo 'This tested setup needs Python 3.12 on Linux.' >&2
  exit 1
fi
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install 'uv==0.12.22'
REQUIREMENTS=requirements.txt
if [[ -f requirements.lock ]]; then REQUIREMENTS=requirements.lock; fi
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
.venv/bin/python -m uv pip install --python .venv/bin/python -r "$REQUIREMENTS" \
  --find-links https://download.pytorch.org/whl/cu128/torch/ \
  --find-links https://download.pytorch.org/whl/cu128/torchvision/
.venv/bin/python -m uv pip install --python .venv/bin/python --no-deps -e .
mkdir -p data/raw data/calibration data/annotations data/processed data/robot runs
bash run.sh doctor
echo 'Setup complete. See README.md for dataset, training, and demo commands.'
