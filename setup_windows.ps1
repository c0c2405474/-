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
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment creation failed. Check that Python is installed correctly, then run: py -3.11 -m venv .venv"
}

Write-Host "Installing required packages..."
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upgrade pip."
}

& $python -m pip install -r requirements-classifier.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install the required packages."
}

if (-not (Test-Path "classifier_output\best_model.pth")) {
    Write-Warning "The trained model was not found at classifier_output\best_model.pth."
}

Write-Host ""
Write-Host "Setup completed."
Write-Host 'Example: .\classify.bat "C:\images\mushroom.jpg"'
