$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python was not found. Install Python 3.10 or 3.11 and enable 'Add Python to PATH'."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating the virtual environment..."
    python -m venv .venv
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
Write-Host "Installing required packages..."
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements-classifier.txt

if (-not (Test-Path "classifier_output\best_model.pth")) {
    Write-Warning "The trained model was not found at classifier_output\best_model.pth."
}

Write-Host ""
Write-Host "Setup completed."
Write-Host 'Example: .\classify.bat "C:\images\mushroom.jpg"'
