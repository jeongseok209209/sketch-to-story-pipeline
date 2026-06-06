@echo off
setlocal

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Setup failed. Read the message above, fix the issue, then run .\setup.bat again.
    exit /b %EXIT_CODE%
)

echo.
echo Setup finished successfully.
echo Next smoke test:
echo .\.venv\Scripts\python.exe run.py a --story 1 --image 1 --story-max-new-tokens 20 --output-dir outputs\smoke_A

endlocal
