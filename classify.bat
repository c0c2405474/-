@echo off
setlocal
cd /d "%~dp0"
chcp 932 >nul

if not exist ".venv\Scripts\python.exe" (
    echo First-time setup is required.
    echo Run .\setup_windows.ps1 in PowerShell.
    exit /b 1
)

if "%~1"=="" (
    echo Specify one image file.
    echo Usage: classify.bat "image-file-path"
    echo Example: classify.bat "C:\Users\name\Pictures\mushroom.jpg"
    exit /b 1
)

if not "%~2"=="" (
    echo Specify only one image file at a time.
    exit /b 1
)

echo Checking brightness and classifying...
".venv\Scripts\python.exe" classify_image.py --input "%~1" --auto-enhance
if errorlevel 1 (
    echo.
    echo Classification failed. Check the image path and model.
    exit /b 1
)

endlocal
