@echo off
REM Restore Cursor to normal (disable proxy)
REM Run as Administrator!

echo ========================================
echo   Restore Cursor to Normal
echo ========================================
echo.

REM Check admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Please run as Administrator!
    pause
    exit /b 1
)

echo Removing hosts entry...
powershell -Command "(Get-Content C:\Windows\System32\drivers\etc\hosts) | Where-Object {$_ -notmatch 'api2.cursor.sh'} | Set-Content C:\Windows\System32\drivers\etc\hosts"

echo.
echo Done! Cursor will use normal API.
echo.
echo Next: Restart Cursor IDE
echo.
pause
