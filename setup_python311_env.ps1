# Setup Script for Python 3.11 Environment
# Run this script to create a proper environment for the project

Write-Output "Creating Python 3.11 environment for Object Detection GUI..."
Write-Output ""

# Create a new conda environment with Python 3.11
Write-Output "Step 1: Creating conda environment 'sp-gui' with Python 3.11..."
conda create -n sp-gui python=3.11 -y

Write-Output ""
Write-Output "Step 2: Activating environment..."
conda activate sp-gui

Write-Output ""
Write-Output "Step 3: Installing core dependencies..."
conda install -c conda-forge numpy scipy scikit-learn pillow -y

Write-Output ""
Write-Output "Step 4: Installing PyTorch..."
# Install PyTorch (CPU version - change if you need CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Output ""
Write-Output "Step 5: Installing additional requirements..."
pip install timm

Write-Output ""
Write-Output "Step 6: Installing optional detection dependencies..."
pip install effdet transformers

Write-Output ""
Write-Output "="*70
Write-Output "Setup Complete!"
Write-Output "="*70
Write-Output ""
Write-Output "To use the environment:"
Write-Output "  1. Run: conda activate sp-gui"
Write-Output "  2. Then: python run_gui.py"
Write-Output ""
Write-Output "To verify installation:"
Write-Output "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
Write-Output "  python -c 'import numpy; print(f\"NumPy: {numpy.__version__}\")'"
Write-Output ""
