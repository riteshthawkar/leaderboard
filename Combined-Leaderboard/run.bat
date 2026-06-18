@echo off
REM Combined Vision Leaderboard - Windows Startup Script

echo.
echo ================================
echo Combined Vision Leaderboard
echo ================================
echo.

REM Check if .env exists
if not exist .env (
    echo Creating .env from .env.example...
    copy .env.example .env
    echo.
    echo [OK] .env created. Please update paths in .env file.
    echo.
)

REM Check if venv exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
    echo.
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt -q
echo [OK] Dependencies installed
echo.

REM Run the Flask app
echo Starting Combined Leaderboard Server...
echo.
echo Server running at http://localhost:5000
echo.
cd backend\web
python -m flask run
