#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run scripts/setup_env.sh first."
  exit 1
fi

source .venv/bin/activate

python -m ipykernel install \
  --user \
  --name m2-normalizing-flows \
  --display-name "Python (m2-normalizing-flows)"

echo "Registered kernel: Python (m2-normalizing-flows)"
echo "Available kernels:"
jupyter kernelspec list
