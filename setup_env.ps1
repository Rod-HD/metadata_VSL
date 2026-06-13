<#
    setup_env.ps1 - Recreate the project-local Python environment on D: (never C:).

    This script:
      * Redirects the pip cache to D:\...\.pip-cache (keeps pip cache off C:).
      * Creates a virtual environment at D:\...\.venv using "py -3 -m venv".
      * Upgrades pip and installs the pinned requirements.txt into that venv.

    Run from anywhere:  powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
#>

$ErrorActionPreference = "Stop"

# Project root on D: (this script's own directory).
$proj = $PSScriptRoot
if ([string]::IsNullOrEmpty($proj)) {
    $proj = (Get-Location).Path
}

Write-Host "Project root: $proj"

# Keep pip's cache off C: by redirecting it to D:.
$env:PIP_CACHE_DIR = Join-Path $proj ".pip-cache"
Write-Host "PIP_CACHE_DIR = $env:PIP_CACHE_DIR"

# Paths to the venv and its interpreter.
$venv = Join-Path $proj ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"

# Create the venv on D: (uses the Windows "py" launcher, Python 3).
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $venv ..."
    py -3 -m venv $venv
} else {
    Write-Host "Virtual environment already exists at $venv"
}

# Upgrade pip, then install pinned requirements into the venv.
Write-Host "Upgrading pip ..."
& $venvPython -m pip install --upgrade pip

$req = Join-Path $proj "requirements.txt"
Write-Host "Installing requirements from $req ..."
& $venvPython -m pip install -r $req

Write-Host ""
Write-Host "Done. Interpreter: $venvPython"
& $venvPython -c "import sys; print('sys.executable =', sys.executable)"
