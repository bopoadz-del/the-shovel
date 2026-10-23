#!/usr/bin/env bash
# Regenerate pinned requirements*.txt from human-edited *.in files.
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="${HOME}/.local/bin:${PATH}"

pip install -q pip-tools

pip-compile requirements.in -o requirements.txt --strip-extras
pip-compile requirements-rag.in -o requirements-rag.txt --strip-extras
pip-compile requirements-cv.in -o requirements-cv.txt --strip-extras
pip-compile requirements-ml.in -o requirements-ml.txt --strip-extras

echo "Pinned requirements written."
