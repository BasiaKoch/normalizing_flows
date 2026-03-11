#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run scripts/setup_env.sh first."
  exit 1
fi

source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace coursework.ipynb

echo "Notebook executed successfully: coursework.ipynb"
