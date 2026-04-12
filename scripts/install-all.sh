#!/usr/bin/env bash
# Editable install of every Kairo subpackage.
# Run from the repo root:  bash scripts/install-all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Order matters: core first, then anything that depends on core, then things
# that depend on those. Same as the dependency table in the README.
PACKAGES=(
    core
    agents
    ingest
    retrieval
    flow
    rag
    connectors
    edge
    cli
)

for pkg in "${PACKAGES[@]}"; do
    echo ">>> installing kairo-$pkg (editable)"
    pip install -e "./$pkg"
done

echo ">>> installing root meta + dev deps"
pip install -e ".[dev]"

echo "done."
