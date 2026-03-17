@echo off
REM ============================================================
REM AEGIS SME — One-Command Setup Script (Windows)
REM Team Finvee | Varsity Hackathon 2026
REM Usage: Double-click or run in Command Prompt: setup.bat
REM ============================================================

echo.
echo ⚔️  AEGIS SME — Setup Script
echo    Team Finvee ^| Varsity Hackathon 2026
echo ============================================

REM Step 1: Check if UV is installed
where uv >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo 📦 Installing UV package manager...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo ✅ UV installed. Please restart this script after installation.
    pause
    exit /b
) ELSE (
    echo ✅ UV already installed
)

echo.
echo 🔧 Creating virtual environment with UV...
uv venv .venv --python 3.11

echo.
echo 📥 Installing all dependencies (this is fast with UV)...
uv pip install -r requirements.txt

echo.
echo ============================================
echo ✅ Setup complete!
echo.
echo To run AEGIS SME, open 2 Command Prompt windows:
echo.
echo   Window 1 (API Backend):
echo     set PYTHONPATH=.
echo     python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
echo.
echo   Window 2 (Dashboard):
echo     set PYTHONPATH=.
echo     python -m streamlit run dashboard/app.py --server.port 8501
echo.
echo   Open browser: http://localhost:8501
echo ============================================
pause
