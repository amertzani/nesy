@echo off
echo ========================================
echo Starting NesyX Application
echo ========================================
echo.

REM Check if requirements are installed
echo Checking dependencies...
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install -r requirements.txt
    echo.
)

REM Run the application
echo Starting application...
echo.
python app.py

pause

