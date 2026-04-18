# Editable install of every Kairo subpackage.
# Run from the repo root:  powershell -File scripts/install-all.ps1
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Order matters: core first, then anything that depends on core.
$packages = @(
    "core",
    "agents",
    "ingest",
    "retrieval",
    "flow",
    "rag",
    "analytics",
    "connectors",
    "edge",
    "cli"
)

foreach ($pkg in $packages) {
    Write-Host ">>> installing kairo-$pkg (editable)"
    pip install -e "./$pkg"
}

Write-Host ">>> installing root meta + dev deps"
pip install -e ".[dev]"

Write-Host "done."
