@echo off
echo ========================================
echo Starting NesyX API Server
echo ========================================
echo.
echo This server allows your Replit frontend to connect
echo Server will be accessible at: http://0.0.0.0:8000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Check if requirements are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing API dependencies...
    python -m pip install fastapi uvicorn python-multipart
    echo.
)

REM Run the API server
python api_server.py

pause

