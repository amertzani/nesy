@echo off
REM Enable Triplex for knowledge extraction
REM This sets the environment variable for the current session

echo Enabling Triplex for knowledge extraction...
set USE_TRIPLEX=true
echo Triplex enabled! You can now start the API server.
echo.
echo To start the API server with Triplex:
echo   python api_server.py
echo.
echo Note: The first time you use Triplex, it will download the model (~4GB)
echo       This may take several minutes depending on your internet connection.
echo.
pause

