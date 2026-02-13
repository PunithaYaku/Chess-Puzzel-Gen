# Helper script to run the Chess Puzzle Generator on Windows
# This script handles environment variable cleanup to prevent Python version mismatches

Write-Host "Cleaning up environment variables..." -ForegroundColor Cyan
$env:PYTHONPATH = ""
$env:PYTHONHOME = ""

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    . .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Warning: Virtual environment not found at .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}

Write-Host "Starting application..." -ForegroundColor Green
python app.py
