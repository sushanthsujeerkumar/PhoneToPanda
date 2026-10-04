#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo 'First run: bash setup.sh' >&2
  exit 1
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export LIBERO_CONFIG_PATH="$PROJECT_DIR/.cache/libero"
if [[ $# -eq 0 ]]; then set -- --help; fi
exec .venv/bin/python -m phone_to_panda.cli "$@"
