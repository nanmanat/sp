@echo off
REM Quick Start Script for Object Detection GUI
REM Activates environment and launches the GUI

echo ========================================
echo Object Detection GUI - Quick Start
echo ========================================
echo.

REM Check if sp-gui environment exists
conda info --envs | findstr "sp-gui" >nul
if errorlevel 1 (
    echo ERROR: Environment 'sp-gui' not found!
    echo.
    echo Please run setup first:
    echo   setup_python311_env.bat
    echo.
    pause
    exit /b 1
)

echo Activating environment 'sp-gui'...
call conda activate sp-gui

echo.
echo Launching GUI...
echo.
python run_gui.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch GUI
    echo.
    echo Troubleshooting:
    echo   1. Make sure environment is set up: setup_python311_env.bat
    echo   2. Check for missing dependencies
    echo   3. See PYTHON_311_SETUP.md for help
    echo.
    pause
)
