@echo off
REM Set system proxy to route Cursor through our proxy
REM Run as Administrator!

echo Setting system proxy to localhost:443...

REM Check admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Please run as Administrator!
    echo Right-click -^> "Run as Administrator"
    pause
    exit /b 1
)

REM Enable proxy
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyEnable /t REG_DWORD /d 1 /f
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyServer /t REG_SZ /d "127.0.0.1:8888" /f

echo.
echo System proxy enabled: 127.0.0.1:8888
echo.
echo Now start the proxy on port 8888 (not 443)
echo Then restart Cursor IDE
echo.

pause
