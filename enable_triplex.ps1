# Enable Triplex for knowledge extraction
# This sets the environment variable for the current PowerShell session

Write-Host "Enabling Triplex for knowledge extraction..." -ForegroundColor Green
$env:USE_TRIPLEX = "true"
Write-Host "Triplex enabled! You can now start the API server." -ForegroundColor Green
Write-Host ""
Write-Host "To start the API server with Triplex:" -ForegroundColor Yellow
Write-Host "  python api_server.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: The first time you use Triplex, it will download the model (~4GB)" -ForegroundColor Yellow
Write-Host "      This may take several minutes depending on your internet connection." -ForegroundColor Yellow
Write-Host ""
Write-Host "To verify Triplex is enabled, run:" -ForegroundColor Yellow
Write-Host "  python -c \"from knowledge import USE_TRIPLEX; print('Enabled:', USE_TRIPLEX)\"" -ForegroundColor Cyan

