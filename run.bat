@echo off
title StudyBuddy Local
echo ============================================================
echo  Starting StudyBuddy Local - Private AI Study Partner
echo ============================================================
echo.

if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Virtual environment not found or dependencies not installed.
    echo Please run: python -m venv .venv ^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Launching Streamlit interface at http://127.0.0.1:8501 ...
.\.venv\Scripts\streamlit.exe run app.py
pause
