$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Pythonが見つかりません。Python 3.10または3.11をインストールし、インストール時に「Add Python to PATH」を選択してください。"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "仮想環境を作成しています..."
    python -m venv .venv
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
Write-Host "必要なライブラリをインストールしています..."
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements-classifier.txt

if (-not (Test-Path "classifier_output\best_model.pth")) {
    Write-Warning "学習済みモデルが見つかりません。classifier_output\best_model.pth をプロジェクトに配置してください。"
}

Write-Host ""
Write-Host "セットアップが完了しました。"
Write-Host '実行例: .\classify.bat "C:\画像\mushroom.jpg"'
