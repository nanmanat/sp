# Python 3.11 Environment Setup Guide

## Problem
You're encountering NumPy/SciPy compatibility issues in your base conda environment because:
- Base environment has NumPy 2.3.3
- SciPy requires NumPy <2.0
- scikit-learn has incompatibilities

## Solution: Create a New Python 3.11 Environment

### Option 1: Automated Setup (Recommended)

#### Windows PowerShell:
```powershell
.\setup_python311_env.ps1
```

#### Windows Command Prompt:
```cmd
setup_python311_env.bat
```

### Option 2: Manual Setup

#### Step 1: Create Environment
```bash
conda create -n sp-gui python=3.11 -y
```

#### Step 2: Activate Environment
```bash
conda activate sp-gui
```

#### Step 3: Install Core Dependencies
```bash
conda install -c conda-forge numpy scipy scikit-learn pillow -y
```

#### Step 4: Install PyTorch

**For CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA 11.8 (if you have NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1 (if you have newer NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Step 5: Install Additional Requirements
```bash
pip install timm
```

#### Step 6: Install Optional Detection Dependencies (Optional)
```bash
pip install effdet transformers
```

### Verify Installation

After setup, verify everything is installed correctly:

```bash
# Check Python version
python --version
# Should show: Python 3.11.x

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check NumPy
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Check if CUDA is available (if you installed GPU version)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running the GUI

### Every Time You Use the Project:

1. **Activate the environment:**
   ```bash
   conda activate sp-gui
   ```

2. **Run the GUI:**
   ```bash
   python run_gui.py
   ```

### Or use the convenience script:
```bash
python run_gui.py
```
(Make sure the environment is activated first!)

## Creating a Startup Shortcut

### Windows Batch Script
Create `start_gui.bat`:
```batch
@echo off
call conda activate sp-gui
python run_gui.py
pause
```

### Windows PowerShell Script
Create `start_gui.ps1`:
```powershell
conda activate sp-gui
python run_gui.py
```

Then run:
```powershell
.\start_gui.ps1
```

## Troubleshooting

### Issue: "conda: command not found"
**Solution:** Make sure conda is in your PATH or use Anaconda Prompt

### Issue: "Environment already exists"
**Solution:** Remove the old environment first:
```bash
conda env remove -n sp-gui
```
Then create it again.

### Issue: Still getting NumPy errors
**Solution:** Make sure you're in the correct environment:
```bash
conda activate sp-gui
python -c "import numpy; print(numpy.__version__)"
```
Should show NumPy version <2.0 (e.g., 1.26.x)

### Issue: Import errors when running GUI
**Solution:** Install missing packages in the sp-gui environment:
```bash
conda activate sp-gui
pip install <package-name>
```

## Package Versions (Recommended)

The environment will install these compatible versions:
- Python: 3.11.x
- NumPy: 1.26.x (automatically managed by conda)
- SciPy: 1.11.x
- scikit-learn: 1.3.x
- PyTorch: 2.1.x or later
- torchvision: 0.16.x or later
- Pillow: 10.x
- timm: 0.9.x or later

## Alternative: Using venv Instead of Conda

If you prefer using Python's built-in venv:

```bash
# Create virtual environment
python3.11 -m venv sp-env

# Activate (Windows)
sp-env\Scripts\activate

# Activate (Linux/Mac)
source sp-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Complete Requirements File

See `requirements.txt` for the complete list of dependencies.

For object detection models, also install:
```bash
pip install -r requirements_detection.txt
```

## Summary

✅ **Recommended Setup:**
1. Create new Python 3.11 conda environment
2. Install dependencies in that environment
3. Always activate environment before running
4. Use provided setup scripts for convenience

✅ **Key Points:**
- Don't use base conda environment (has conflicts)
- Use Python 3.11 for best compatibility
- Install NumPy <2.0 to avoid SciPy issues
- GPU support is optional (CPU works fine for GUI)

---

**Quick Start:**
```bash
# One-time setup
.\setup_python311_env.bat

# Every time you use
conda activate sp-gui
python run_gui.py
```

**That's it! You're ready to use the Object Detection GUI with Python 3.11!** 🎉
