@echo off
REM Setup Script for Python 3.11 Environment
REM Run this script to create a proper environment for the project

echo Creating Python 3.11 environment for Object Detection GUI...
echo.

echo Step 1: Creating conda environment 'sp-gui' with Python 3.11...
call conda create -n sp-gui python=3.11 -y

echo.
echo Step 2: Activating environment...
call conda activate sp-gui

echo.
echo Step 3: Installing core dependencies...
call conda install -c conda-forge numpy scipy scikit-learn pillow -y

echo.
echo Step 4: Installing PyTorch (CPU version)...
call pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo Step 5: Installing additional requirements...
call pip install timm

echo.
echo Step 6: Installing optional detection dependencies...
call pip install effdet transformers

echo.
echo ======================================================================
echo Setup Complete!
echo ======================================================================
echo.
echo To use the environment:
echo   1. Run: conda activate sp-gui
echo   2. Then: python run_gui.py
echo.
echo To verify installation:
echo   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo   python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
echo.
pause
