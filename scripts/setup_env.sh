#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install .

echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Optional Jupyter kernel setup: scripts/register_kernel.sh"
