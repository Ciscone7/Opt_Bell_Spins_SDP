@echo off
REM setup_env.bat — Create the SpinsSDP virtual environment (Windows)
REM Usage: double-click this file, or run it from Command Prompt / PowerShell

setlocal EnableDelayedExpansion

set VENV_DIR=.venv
set PYTHON=python

echo === SpinsSDP environment setup (Windows) ===

REM Check Python is available
%PYTHON% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: 'python' not found. Install Python 3.10+ from python.org and ensure it is on PATH.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('%PYTHON% --version') do echo Using %%v

REM Create virtual environment
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment '%VENV_DIR%' already exists -- skipping creation.
) else (
    echo Creating virtual environment in '%VENV_DIR%'...
    %PYTHON% -m venv %VENV_DIR%
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

REM Activate
call %VENV_DIR%\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing runtime dependencies from requirements.txt...
pip install --prefer-binary -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip install failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo === Setup complete! ===
echo Activate the environment with:
echo     %VENV_DIR%\Scripts\activate.bat
echo.
pause
