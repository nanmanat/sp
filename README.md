## Joint Acne Image Grading and Counting via Label Distri#### Supported Models

##### Classification Models

The following classification models are supported:
- vgg16_bn
- resnet50
- efficientnet_v2_s
- convnext_tiny
- densenet121
- regnet_y_8gf
- mobilenet_v3_large
- vit_small_patch16_224
- swin_tiny_patch4_window7_224
- deit_small_patch16_224

##### Object Detection Models (NEW!)

The following object detection models are supported:
- **SSD300** - Single Shot MultiBox Detector
- **EfficientDet-D0/D1/D2/D3** - Efficient detection models
- **RetinaNet R50-FPN** - One-stage with focal loss
- **Faster R-CNN R50-FPN** - Classic two-stage detector
- **Cascade R-CNN R50-FPN** - Cascaded refinement
- **FCOS R50-FPN** - Anchor-free detector
- **ATSS R50-FPN** - Adaptive training sample selection
- **CenterNet Hourglass-104** - Keypoint-based detector
- **DETR R50** - Detection Transformer
- **Deformable DETR R50** - DETR with deformable attention
- **RT-DETR R50** - Real-time DETR

For detailed information about object detection models, see [OBJECT_DETECTION_GUIDE.md](OBJECT_DETECTION_GUIDE.md).

To run the detection models example:
```bash
python run_detection_example.py
```ng
Pytorch implementation of "Joint Acne Image Grading and Counting via Label Distribution Learning"

This work was accepted by ICCV 2019 [[paper](http://xiaopingwu.cn/assets/paper/iccv2019_ldl.pdf)].

### ACNE04 Dataset

The ACNE04 dataset can be downloaded from [Baidu](https://pan.baidu.com/s/15JQlymnhnEmEt8Q5zpJQDw) (pw: fbrm) or [Google](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ?usp=sharing).

### Experiment GUI

A graphical user interface is provided to easily run experiments with a queue feature. This allows you to configure and queue multiple experiments to run sequentially.

#### Setup (First Time)

**Quick Setup with Python 3.11:**
```bash
# Run the setup script (Windows)
setup_python311_env.bat

# Or see SETUP_GUIDE.md for detailed instructions
```

For detailed setup instructions, see:
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Quick setup guide
- **[PYTHON_311_SETUP.md](PYTHON_311_SETUP.md)** - Detailed environment setup

#### Running the GUI

**Option 1: Use the quick start script (easiest)**
```bash
start_gui.bat
```

**Option 2: Manual method**
```bash
conda activate sp-gui
python run_gui.py
```

or on Unix-like systems:
```bash
./run_gui.py
```

#### Using the GUI

The GUI has four tabs:

1. **Classification**: Set up image classification experiments
   - Select a model architecture
   - Choose cross-validation folds
   - Configure batch size, learning rate, and number of workers
   - Set the data path
   - Configure early stopping parameters
   - Click "Add to Queue" to add the experiment to the queue

2. **Object Detection**: Set up object detection experiments (NEW!)
   - Choose from 14 detection models (SSD, EfficientDet, Faster R-CNN, DETR, etc.)
   - Configure detection-specific parameters
   - Support for multiple detection architectures
   - See [OBJECT_DETECTION_GUIDE.md](OBJECT_DETECTION_GUIDE.md) for detailed information

3. **Queue**: Manage the experiment queue
   - View queued experiments (both classification and detection)
   - Start processing the queue
   - Stop after the current experiment
   - Clear the queue

4. **Log**: View training output
   - Real-time log of training progress
   - Save logs to a file

#### Supported Models

##### Classification Models

The following models are supported:
- vgg16_bn
- resnet50
- efficientnet_v2_s
- convnext_tiny
- densenet121
- regnet_y_8gf
- mobilenet_v3_large
- vit_small_patch16_224
- swin_tiny_patch4_window7_224
- deit_small_patch16_224

### Additional Information
If you find this work helpful, please cite it as
```
@InProceedings{Wu_2019_ICCV,
  author = {Wu, Xiaoping and Ni, Wen and Jie, Liang and Lai, Yu-Kun and Cheng, Dongyu, She and Ming-Ming and Yang, Jufeng},
  title = {Joint Acne Image Grading and Counting via Label Distribution Learning},
  booktitle = {IEEE International Conference on Computer Vision},
  year = {2019}
}
```

ATTN: This work is free for academic usage. For other purposes, please contact Xiaoping Wu (xpwu95@163.com).
