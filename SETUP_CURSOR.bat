@echo off
REM ONE-CLICK SETUP for Cursor MITM Proxy
REM Run as Administrator!

echo ========================================
echo   Cursor Smart Routing Setup
echo ========================================
echo.

REM Check admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Please run as Administrator!
    echo Right-click -^> "Run as Administrator"
    pause
    exit /b 1
)

echo [1/3] Adding hosts entry...
findstr /C:"api2.cursor.sh" C:\Windows\System32\drivers\etc\hosts >nul
if %errorLevel% neq 0 (
    echo 127.0.0.1 api2.cursor.sh >> C:\Windows\System32\drivers\etc\hosts
    echo Done!
) else (
    echo Already exists, skipping
)

echo.
echo [2/3] Installing SSL certificate...
if exist cursor_cert.pem (
    certutil -addstore -f "Root" cursor_cert.pem >nul 2>&1
    echo Done!
) else (
    echo Certificate will be generated on first run
)

echo.
echo [3/3] Starting proxy with SMART ROUTING...
echo.
echo ========================================
echo SMART ROUTING ACTIVE:
echo   Simple questions   -^> cursor-small (FREE)
echo   Tool-use/reading   -^> cursor-small (FREE)
echo   Code generation    -^> claude-sonnet
echo ========================================
echo.
echo Next: Restart Cursor IDE
echo.

cd /d "%~dp0src"
..\venv\Scripts\python.exe -m distiq_code.cli cursor-proxy --port 443 --passthrough=false

pause
