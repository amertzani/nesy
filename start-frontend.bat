@echo off
echo ========================================
echo Starting NesyX Frontend (Local)
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "RandDKnowledgeGraph\package.json" (
    echo Error: RandDKnowledgeGraph folder not found!
    echo Make sure you're in the xNeSy2 folder.
    pause
    exit /b 1
)

REM Navigate to frontend folder
cd RandDKnowledgeGraph

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
    echo.
)

REM Install cross-env if not already installed
echo Checking dependencies...
npm list cross-env >nul 2>&1
if errorlevel 1 (
    echo Installing cross-env for Windows compatibility...
    call npm install cross-env --save-dev
    if errorlevel 1 (
        echo Warning: cross-env installation failed, will use Windows syntax
    )
    echo.
)

REM Start the development server
echo Starting frontend development server...
echo.
echo Frontend will be available at: http://localhost:5005
echo Make sure backend is running at: http://192.168.1.92:8000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Try Windows syntax first (most reliable on Windows)
npm run dev:win
if errorlevel 1 (
    echo.
    echo Trying with cross-env...
    npm run dev
)

pause

