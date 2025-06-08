@echo off
echo Setting up Diabetes-Check Application...
echo =====================================

:: Create necessary directories
if not exist "data\uploads" mkdir "data\uploads"
if not exist "static" mkdir "static"

echo.
echo Installing Python dependencies...
echo ===============================
pip install -r requirements.txt

if errorlevel 1 (
    echo Failed to install dependencies. Please check your Python and pip installation.
    pause
    exit /b 1
)

echo.
echo Starting Backend Server...
echo ========================
start "" cmd /k "python run_backend.py"

timeout /t 5 /nobreak >nul

echo.
echo Starting Frontend...
echo ===================
start "" cmd /k "streamlit run src/frontend/app.py"

echo.
echo Application is starting up...
echo Backend will be available at: http://127.0.0.1:5000
echo Frontend will be available at: http://localhost:8501
echo.
pause
