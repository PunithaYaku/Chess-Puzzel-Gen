@echo off
setlocal
echo ==========================================
echo   Chess AI Puzzle Generator - Installer
echo ==========================================
echo.

:: Fix for SRE module mismatch (unset global python path/home)
set PYTHONPATH=
set PYTHONHOME=

echo [1/3] Checking dependencies...
python -m pip install fastapi uvicorn torch python-chess pandas numpy

echo.
echo [2/3] Checking models...
if not exist fen_generator.pth (
    echo [!] fen_generator.pth not found. 
    echo Training a small baseline model first...
    python train_gen.py
)

echo.
echo [3/3] Starting the Interactive GUI...
echo Access the GUI at: http://localhost:8000
echo.
python app.py
endlocal
pause
