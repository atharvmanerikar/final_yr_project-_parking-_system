@echo off
title NextGen Smart Parking System Setup
echo =======================================================
echo   NextGen Smart Parking System Setup
echo =======================================================
echo.

:: 1. Verify Python Installation
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your PATH environment variable.
    echo Please install Python (3.10 to 3.14) and check the "Add Python to PATH" box.
    goto :error
)
echo Python detected.

:: 2. Verify Node/NPM Installation
echo.
echo [2/6] Checking Node.js and NPM...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not added to your PATH.
    echo Please install Node.js (v18+) to set up the React frontend.
    goto :error
)
echo Node.js and NPM detected.

:: 3. Setup Virtual Environment
echo.
echo [3/6] Setting up Python virtual environment (venv)...
if not exist venv (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        goto :error
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists (skipping creation).
)

:: 4. Install Backend Dependencies
echo.
echo [4/6] Installing backend dependencies...
venv\Scripts\python.exe -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARNING] Failed to upgrade pip. Proceeding anyway...
)

venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements from requirements.txt.
    goto :error
)

:: 5. Install PyTorch with GPU or CPU support
echo.
echo [5/6] Detecting NVIDIA GPU and configuring PyTorch...
nvidia-smi >nul 2>&1
if %errorlevel% eq 0 (
    echo NVIDIA GPU detected! Installing PyTorch with CUDA 12.6 acceleration...
    venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
    if %errorlevel% neq 0 (
        echo [WARNING] CUDA-enabled PyTorch installation failed. Falling back to default CPU version...
        venv\Scripts\python.exe -m pip install torch torchvision
    )
) else (
    echo No NVIDIA GPU detected. Installing standard CPU version of PyTorch...
    venv\Scripts\python.exe -m pip install torch torchvision
)

:: 6. Setup Environment Config File
if not exist .env (
    echo.
    echo Setting up default configuration (.env)...
    copy .env.example .env >nul
    echo Created .env config file.
)

:: 7. Build Frontend
echo.
echo [6/6] Installing Node modules and building React frontend...
cd frontend
echo Installing frontend dependencies...
call npm install
if %errorlevel% neq 0 (
    echo [ERROR] Frontend npm install failed.
    cd ..
    goto :error
)

echo Building production React package...
call npm run build
if %errorlevel% neq 0 (
    echo [ERROR] Frontend production build failed.
    cd ..
    goto :error
)
cd ..

echo.
echo =======================================================
echo   Setup Completed Successfully!
echo =======================================================
echo   To launch the system:
echo     1. Open run.bat to start both backend and dev server.
echo     2. Or run: venv\Scripts\python.exe -m uvicorn backend.main:app
echo        and open http://localhost:8000 in your browser.
echo =======================================================
pause
exit /b 0

:error
echo.
echo [FATAL ERROR] Setup failed. Please check the logs above and try again.
pause
exit /b 1
