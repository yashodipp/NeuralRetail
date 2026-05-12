#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Virtual environment not found. Create it first with:"
  echo "pyenv exec python -m venv .venv"
  exit 1
fi

exec .venv/bin/python -m streamlit run run.py
