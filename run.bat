@echo off
echo ============================================================
echo  Emotion-Aware Sarcasm Detection System - Setup and Run
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not on PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/4] Virtual environment already exists.
)

:: Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo [3/4] Installing dependencies...
pip install -r requirements.txt --quiet

:: Check if models exist
if not exist "saved_models\baseline_classifier.pkl" (
    echo [4/4] Training models (quick mode)...
    python train.py --quick
) else (
    echo [4/4] Models already trained. Skipping training.
)

echo.
echo ============================================================
echo  Starting web server at http://localhost:5000
echo  Press Ctrl+C to stop.
echo ============================================================
echo.

python app.py

pause
