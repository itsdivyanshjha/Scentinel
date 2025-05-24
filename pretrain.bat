@echo off
REM Pre-training script for Scentinel (Windows version)
REM This script sets up a virtual environment and runs the pre-training

echo Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python could not be found. Please install Python.
    exit /b 1
)

echo Setting up environment...

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install pandas numpy torch scikit-learn gensim python-dotenv

REM Run the pre-training
echo Running pre-training...
python standalone_pretrain.py

REM Check if pre-training was successful
if %ERRORLEVEL% EQU 0 (
    echo Pre-training completed successfully!
    echo Pre-trained models are saved in backend\app\data\models\
) else (
    echo Pre-training failed! Check the error messages above.
)

REM Deactivate virtual environment
call deactivate
echo Virtual environment deactivated.

echo.
echo Press any key to exit...
pause >nul 