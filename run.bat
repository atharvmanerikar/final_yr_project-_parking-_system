@echo off
title NextGen Smart Parking Control panel
echo ===================================================
echo   Starting NextGen Smart Parking System...
echo ===================================================

:: Start FastAPI Backend using Python venv
echo [1/2] Starting FastAPI Backend on Port 8000...
start "Smart Parking - Backend" cmd /k "venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

:: Start React Frontend Dev Server
echo [2/2] Starting Vite Frontend on Port 5173...
start "Smart Parking - Frontend" cmd /k "cd frontend && npm run dev"

echo ===================================================
echo   System running!
echo   - Backend API Docs: http://localhost:8000/docs
echo   - Frontend Dashboard: http://localhost:5173
echo ===================================================
pause
