# StudyBuddy Local PowerShell Launcher
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Starting StudyBuddy Local - Private AI Study Partner" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path ".\.venv\Scripts\streamlit.exe")) {
    Write-Host "[ERROR] Virtual environment or dependencies not found in .venv." -ForegroundColor Red
    exit 1
}

Write-Host "Launching Streamlit interface at http://127.0.0.1:8501 ..." -ForegroundColor Green
& .\.venv\Scripts\streamlit.exe run app.py
