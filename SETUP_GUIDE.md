# 🚀 Quick Setup & Run Guide

## First Time Setup (One Time Only)

### Step 1: Run the Setup Script

**Option A - Double-click the file:**
- Double-click `setup_python311_env.bat`

**Option B - Run from command line:**
```cmd
setup_python311_env.bat
```

This will:
- ✅ Create a new Python 3.11 environment named `sp-gui`
- ✅ Install all required dependencies
- ✅ Install PyTorch and torchvision
- ✅ Install optional object detection libraries
- ✅ Fix NumPy/SciPy compatibility issues

**Time:** About 5-10 minutes depending on your internet speed

---

## Running the GUI (Every Time)

### Option 1: Use the Quick Start Script (Easiest)

**Double-click:**
- `start_gui.bat`

**Or run from command line:**
```cmd
start_gui.bat
```

### Option 2: Manual Method

```cmd
conda activate sp-gui
python run_gui.py
```

---

## 📋 What's Included

After setup, you'll have:

### ✅ Classification Models (10 models)
- VGG16-BN
- ResNet50
- EfficientNet-V2-S
- ConvNeXt-Tiny
- DenseNet121
- RegNet-Y-8GF
- MobileNet-V3-Large
- ViT-Small
- Swin-Tiny
- DeiT-Small

### ✅ Object Detection Models (14 models)
- SSD300
- EfficientDet-D0/D1/D2/D3
- RetinaNet R50-FPN
- Faster R-CNN R50-FPN
- Cascade R-CNN R50-FPN
- FCOS R50-FPN
- ATSS R50-FPN
- CenterNet Hourglass-104
- DETR R50
- Deformable DETR R50
- RT-DETR R50

### ✅ Utilities
- Dataset format converter (COCO ↔ VOC ↔ YOLO)
- Training queue system
- Real-time logging
- Model checkpointing

---

## 🔧 Troubleshooting

### Problem: "conda: command not found"
**Solution:** Use **Anaconda Prompt** or **Anaconda PowerShell** instead of regular Command Prompt

### Problem: "Environment already exists"
**Solution:** Remove and recreate:
```cmd
conda env remove -n sp-gui
setup_python311_env.bat
```

### Problem: GUI shows NumPy errors
**Solution:** Make sure you activated the environment:
```cmd
conda activate sp-gui
python -c "import numpy; print(numpy.__version__)"
```
Should show version 1.26.x (not 2.x)

### Problem: "torch module not found"
**Solution:** Reinstall PyTorch in the environment:
```cmd
conda activate sp-gui
pip install torch torchvision
```

### Problem: Slow startup or missing models
**Solution:** Install optional dependencies:
```cmd
conda activate sp-gui
pip install effdet timm transformers
```

---

## 📚 Documentation

- **Setup Guide**: See `PYTHON_311_SETUP.md`
- **Object Detection Guide**: See `OBJECT_DETECTION_GUIDE.md`
- **Quick Start**: See `QUICKSTART_DETECTION.md`
- **Dataset Formats**: See `DATASET_FORMATS.md`
- **Full Documentation Index**: See `DETECTION_DOCS_INDEX.md`

---

## 🎯 Quick Command Reference

```cmd
# Setup (one time)
setup_python311_env.bat

# Start GUI
start_gui.bat

# Or manually
conda activate sp-gui
python run_gui.py

# View available models
conda activate sp-gui
python run_detection_example.py

# Test format converter
conda activate sp-gui
python run_format_converter.py
```

---

## 💡 Tips

1. **Always activate the environment** before running Python commands
2. **Use start_gui.bat** for the easiest launch
3. **Check the Log tab** in GUI to monitor training
4. **Start with smaller models** (SSD300, EfficientDet-D0) to test your setup
5. **Read OBJECT_DETECTION_GUIDE.md** for detailed model information

---

## ✅ Verify Setup

After running setup, verify everything works:

```cmd
conda activate sp-gui

REM Check Python version
python --version

REM Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

REM Check NumPy (should be <2.0)
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

REM Check GUI modules
python -c "from code.gui import main; print('GUI imports OK!')"
```

If all checks pass, you're ready to go! 🎉

---

## 🆘 Need Help?

1. **Setup Issues**: See `PYTHON_311_SETUP.md`
2. **Detection Questions**: See `OBJECT_DETECTION_GUIDE.md`
3. **General Questions**: See `README.md`

---

**Ready to start?** Run `setup_python311_env.bat` now! ⚡
