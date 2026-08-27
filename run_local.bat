@echo off
setlocal

set "PORT=8000"
set "OPEN_BROWSER=1"

if not "%~1"=="" (
    if /i "%~1"=="--no-open" (
        set "OPEN_BROWSER=0"
    ) else (
        set "PORT=%~1"
    )
)

if /i "%~2"=="--no-open" set "OPEN_BROWSER=0"

pushd "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
    ) else (
        echo Python was not found. Please install Python 3 or add it to PATH.
        popd
        exit /b 1
    )
)

echo Starting local preview at http://127.0.0.1:%PORT%/
echo Press Ctrl+C to stop the server.
if "%OPEN_BROWSER%"=="1" start "" "http://127.0.0.1:%PORT%/"

%PYTHON_CMD% -m http.server %PORT% --bind 127.0.0.1

popd
