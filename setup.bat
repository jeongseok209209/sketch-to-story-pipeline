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
echo Next: download models + verify (one command):
echo   .\.venv\Scripts\python.exe run.py doctor
echo Then run the full demo (story 7 + evaluation dashboard):
echo   .\.venv\Scripts\python.exe run.py demo

endlocal
