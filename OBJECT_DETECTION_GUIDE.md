# Object Detection Models Guide

This guide provides information about the object detection models available in the GUI and how to use them.

## ⚠️ Important: Custom Number of Classes

**If you have a custom number of classes (not 91/COCO), please read:**

Most pretrained object detection models are trained on the COCO dataset with **91 classes**. If you're training with a different number of classes:

### Quick Reference for num_classes:
- **2 object classes** → `num_classes = 3` (2 + 1 background)
- **5 object classes** → `num_classes = 6` (5 + 1 background)
- **10 object classes** → `num_classes = 11` (10 + 1 background)
- **Formula**: `num_classes = YOUR_CLASSES + 1`

### Important Notes:
- ⚠️ **Pretrained weights expect 91 classes** (COCO dataset)
- ✅ **For custom classes**: Models will train **from scratch** (no pretrained)
- 📝 **You'll see a warning** when training starts, this is normal
- ⏱️ **Training takes longer** without pretrained weights
- 📊 **Need more data** when training from scratch (1000+ images/class recommended)

**See `CUSTOM_CLASSES_FIX.md` for detailed explanation and troubleshooting.**

---

## Available Models

### One-Stage Detectors (Fast Inference)

| Model | Description | Framework | Notes |
|-------|-------------|-----------|-------|
| **SSD300** | Single Shot MultiBox Detector with VGG16 | torchvision | Fast one-stage detector, good for real-time |
| **EfficientDet-D0** | Smallest EfficientDet variant | effdet | Most efficient, fastest inference |
| **EfficientDet-D1** | Balanced EfficientDet variant | effdet | Good speed/accuracy tradeoff |
| **EfficientDet-D2** | Medium EfficientDet variant | effdet | Better accuracy, moderate speed |
| **EfficientDet-D3** | Larger EfficientDet variant | effdet | High accuracy, slower inference |
| **RetinaNet R50-FPN** | RetinaNet with ResNet50-FPN | torchvision | One-stage with focal loss |
| **FCOS R50-FPN** | Fully Convolutional One-Stage | torchvision | Anchor-free detector |
| **ATSS R50-FPN** | Adaptive Training Sample Selection | torchvision | Improved anchor assignment |

### Two-Stage Detectors (High Accuracy)

| Model | Description | Framework | Notes |
|-------|-------------|-----------|-------|
| **Faster R-CNN R50-FPN** | Faster R-CNN with ResNet50-FPN | torchvision | Classic two-stage detector |
| **Cascade R-CNN R50-FPN** | Cascaded refinement detector | torchvision | Improved Faster R-CNN |

### Transformer-Based Detectors (Modern Architecture)

| Model | Description | Framework | Notes |
|-------|-------------|-----------|-------|
| **DETR R50** | DEtection TRansformer | transformers | End-to-end transformer |
| **Deformable DETR R50** | DETR with deformable attention | transformers | Faster convergence |
| **RT-DETR R50** | Real-Time DETR | transformers | Optimized for speed |

### Anchor-Free Detectors

| Model | Description | Framework | Notes |
|-------|-------------|-----------|-------|
| **CenterNet Hourglass-104** | Keypoint-based detector | mmdetection | Requires mmcv-full, mmdet |
| **FCOS R50-FPN** | Fully Convolutional One-Stage | torchvision | Also listed above |

## Installation Requirements

### Basic Requirements (Required)
```bash
pip install torch>=1.10.0
pip install torchvision>=0.14.0
```

### Optional Requirements (Model-Specific)

**For EfficientDet models (D0-D3):**
```bash
pip install effdet
pip install timm
```

**For Transformer-based models (DETR, Deformable DETR, RT-DETR):**
```bash
pip install transformers
```

**For advanced models (CenterNet):**
```bash
pip install mmcv-full
pip install mmdet
```

## Using the GUI

### 1. Start the GUI
```bash
python run_gui.py
```

### 2. Navigate to Object Detection Tab
Click on the "Object Detection" tab in the main window.

### 3. Configure Your Training

#### Model Selection
Choose from 14 available detection models. Click "Model Info" to see detailed information about each model.

#### Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| **Number of Classes** | Total classes including background | 91 (COCO), 21 (VOC), **YOUR_CLASSES + 1** for custom |
| **Batch Size** | Images per batch | 4 (adjust based on GPU memory) |
| **Learning Rate** | Optimizer learning rate | 0.001 |
| **Epochs** | Number of training epochs | 50-100 |
| **Num Workers** | Data loading threads | 4 |
| **Dataset Path** | Path to your dataset | ./datasets/coco |

**Important: Number of Classes Calculation**
- COCO dataset: 80 object classes + background = **91 classes**
- Pascal VOC: 20 object classes + background = **21 classes**  
- **Custom dataset**: YOUR object classes + background = **YOUR_CLASSES + 1**
  - Example: 2 object classes → **num_classes = 3**
  - Example: 5 object classes → **num_classes = 6**
  - Example: 10 object classes → **num_classes = 11**

#### Dataset Format
Your dataset should follow standard object detection format:
- **COCO format** (recommended) - JSON-based with rich metadata
- **Pascal VOC format** - XML-based, simple and readable
- **YOLO format** - TXT-based, lightweight and fast
- **Custom format** with bounding box annotations

*Note: You may need to implement custom dataset loaders for specific formats.*

### 4. Training Options

#### Option A: Add to Queue
Click "Add Detection Training to Queue" to queue the experiment. You can queue multiple experiments and run them sequentially.

#### Option B: Start Immediately
Click "Start Detection Training Now" to begin training immediately in the background.

## Model Selection Guide

### Choose Based on Your Needs:

#### Speed is Priority (Real-time Applications)
- **SSD300**: Fastest, good for embedded systems
- **EfficientDet-D0**: Best efficiency/accuracy ratio
- **RT-DETR R50**: Modern architecture with real-time inference

#### Accuracy is Priority
- **Cascade R-CNN R50-FPN**: Highest accuracy two-stage detector
- **EfficientDet-D3**: Best one-stage accuracy
- **Deformable DETR R50**: Strong transformer-based option

#### Balanced Performance
- **Faster R-CNN R50-FPN**: Classic choice, well-tested
- **EfficientDet-D1 or D2**: Good middle ground
- **RetinaNet R50-FPN**: Solid all-around performer

#### Research/Experimentation
- **DETR R50**: End-to-end learning
- **Deformable DETR R50**: Improved attention mechanism
- **FCOS R50-FPN**: Anchor-free approach

## Code Examples

### Using Object Detection Models Programmatically

```python
from code.object_detection_models import create_detection_model

# Create a model for COCO dataset
model = create_detection_model('faster_rcnn_r50_fpn', num_classes=91)

# Create a model for Pascal VOC
model = create_detection_model('ssd300', num_classes=21)

# Create a model for custom dataset (10 classes + background)
model = create_detection_model('retinanet_r50_fpn', num_classes=11)
```

### Training Programmatically

```python
from code.train_detection import run_detection_training

run_detection_training(
    model_name='faster_rcnn_r50_fpn',
    num_classes=91,
    batch_size=4,
    epochs=50,
    lr=0.001,
    num_workers=4,
    data_path='./datasets/coco'
)
```

### Getting Model Information

```python
from code.object_detection_models import get_model_info, list_available_models

# List all available models
models = list_available_models()
print(f"Available models: {models}")

# Get detailed information about a specific model
info = get_model_info('faster_rcnn_r50_fpn')
print(f"Description: {info['description']}")
print(f"Framework: {info['framework']}")
print(f"Requirements: {info['requirements']}")
```

## Performance Characteristics

### Inference Speed (relative, faster → slower)
1. SSD300
2. EfficientDet-D0
3. RT-DETR R50
4. EfficientDet-D1
5. FCOS R50-FPN, RetinaNet R50-FPN
6. Faster R-CNN R50-FPN
7. EfficientDet-D2
8. Deformable DETR R50
9. EfficientDet-D3
10. Cascade R-CNN R50-FPN
11. DETR R50

### Typical Accuracy (lower → higher mAP)
1. SSD300
2. EfficientDet-D0
3. RetinaNet R50-FPN
4. Faster R-CNN R50-FPN
5. EfficientDet-D1
6. FCOS R50-FPN
7. EfficientDet-D2
8. RT-DETR R50
9. Deformable DETR R50
10. EfficientDet-D3
11. Cascade R-CNN R50-FPN
12. DETR R50

### GPU Memory Usage (lower → higher)
1. SSD300
2. EfficientDet-D0
3. EfficientDet-D1
4. RetinaNet R50-FPN, FCOS R50-FPN
5. Faster R-CNN R50-FPN
6. EfficientDet-D2
7. RT-DETR R50
8. EfficientDet-D3
9. Deformable DETR R50
10. Cascade R-CNN R50-FPN
11. DETR R50

## Common Issues and Solutions

### Out of Memory (OOM) Errors
**Solution**: Reduce batch size
- Start with batch_size=2 for large models
- Use batch_size=4-8 for efficient models
- Enable gradient accumulation if needed

### Model Not Available
**Solution**: Install required dependencies
```bash
# For EfficientDet models
pip install effdet timm

# For DETR models
pip install transformers

# For CenterNet
pip install mmcv-full mmdet
```

### Slow Training
**Solutions**:
- Reduce image resolution
- Use a smaller model variant (e.g., EfficientDet-D0 instead of D3)
- Enable mixed precision training (FP16)
- Use more workers for data loading

### Import Errors
**Solution**: Make sure you're in the correct directory and modules are available
```bash
cd /path/to/sp
python run_gui.py
```

## Dataset Preparation

### COCO Format (Recommended)
```
datasets/
└── coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    │   └── [images]
    └── val2017/
        └── [images]
```

**Format Details:**
- JSON-based annotations with bounding boxes in `[x, y, width, height]` format
- Supports instance segmentation masks
- Widely used in research and competitions
- Rich metadata support

### Pascal VOC Format
```
datasets/
└── VOC2012/
    ├── Annotations/
    │   └── [xml files]
    ├── JPEGImages/
    │   └── [images]
    └── ImageSets/
        └── Main/
            ├── train.txt
            └── val.txt
```

**Format Details:**
- XML-based annotations with bounding boxes in `[xmin, ymin, xmax, ymax]` format
- Each image has a corresponding XML file
- Simple and human-readable
- Good for small to medium datasets

### YOLO Format (TXT-based)
```
datasets/
└── yolo/
    ├── images/
    │   ├── train/
    │   │   └── [image files]
    │   └── val/
    │       └── [image files]
    ├── labels/
    │   ├── train/
    │   │   └── [txt files]
    │   └── val/
    │       └── [txt files]
    └── data.yaml  # Dataset configuration
```

**Format Details:**
- One `.txt` file per image with the same filename
- Each line in the txt file represents one object: `class_id center_x center_y width height`
- All coordinates are normalized (0-1 range)
- Example annotation line: `0 0.5 0.5 0.3 0.4`
  - class_id: 0 (first class)
  - center_x: 0.5 (center at 50% of image width)
  - center_y: 0.5 (center at 50% of image height)
  - width: 0.3 (30% of image width)
  - height: 0.4 (40% of image height)

**Example data.yaml:**
```yaml
# Dataset root directory
path: ../datasets/yolo

# Relative paths from path
train: images/train
val: images/val
test: images/test  # optional

# Number of classes
nc: 80

# Class names
names: ['person', 'bicycle', 'car', 'motorcycle', ...]
```

**Converting to YOLO Format:**
```python
# Example: Convert COCO to YOLO format
def convert_coco_to_yolo(coco_box, img_width, img_height):
    """
    Convert COCO format [x, y, width, height] to YOLO format
    [center_x, center_y, width, height] (normalized)
    """
    x, y, w, h = coco_box
    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    return center_x, center_y, norm_w, norm_h
```

*For more conversion utilities, see `examples/dataset_format_converter.py`*

**Advantages:**
- Lightweight and fast to parse
- Simple text-based format
- Easy to create and modify manually
- Minimal storage space
- Popular in YOLO-based workflows

### Format Comparison

| Feature | COCO | Pascal VOC | YOLO |
|---------|------|------------|------|
| **File Type** | JSON | XML | TXT |
| **Parsing Speed** | Medium | Slow | Fast |
| **File Size** | Medium | Large | Small |
| **Readability** | Good | Excellent | Good |
| **Coordinate Format** | `[x, y, w, h]` | `[xmin, ymin, xmax, ymax]` | `[cx, cy, w, h]` normalized |
| **Metadata Support** | Extensive | Good | Minimal |
| **Segmentation** | ✅ Yes | ❌ No | ❌ No |
| **Easy Manual Edit** | ❌ No | ✅ Yes | ✅ Yes |
| **Industry Standard** | Research | Academic | Real-time Apps |
| **Best For** | Large datasets, research | Small datasets, education | Fast training, YOLO models |

### Which Format to Choose?

- **Choose COCO** if:
  - Working with large-scale datasets
  - Need instance segmentation support
  - Publishing research or benchmarking
  - Using pretrained models from research

- **Choose Pascal VOC** if:
  - Working with small to medium datasets
  - Need human-readable annotations
  - Teaching or learning object detection
  - Converting from existing VOC datasets

- **Choose YOLO** if:
  - Training YOLO-based models
  - Need fast data loading
  - Working with limited storage
  - Manually creating/editing annotations
  - Deploying real-time applications

## Tips for Best Results

1. **Start Small**: Begin with EfficientDet-D0 or SSD300 to verify your pipeline works
2. **Monitor Training**: Use the Log tab to watch training progress
3. **Use Pretrained Weights**: All models support pretrained initialization
4. **Adjust Learning Rate**: Lower LR (0.0001) for fine-tuning, higher (0.001) for training from scratch
5. **Data Augmentation**: Consider adding augmentation for better generalization
6. **Class Balance**: Ensure your dataset has balanced class distribution
7. **Validation**: Always use a separate validation set to monitor overfitting

## References

- **SSD**: Liu et al., "SSD: Single Shot MultiBox Detector" (ECCV 2016)
- **EfficientDet**: Tan et al., "EfficientDet: Scalable and Efficient Object Detection" (CVPR 2020)
- **RetinaNet**: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
- **Faster R-CNN**: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection" (NIPS 2015)
- **FCOS**: Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection" (ICCV 2019)
- **DETR**: Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020)
- **Deformable DETR**: Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection" (ICLR 2021)

## Support

For more examples, see:
- `examples/detection_models_example.py` - Model listing and usage examples
- `examples/dataset_format_converter.py` - Convert between COCO, VOC, and YOLO formats
- `code/object_detection_models.py` - Model implementations
- `code/train_detection.py` - Training pipeline

For issues or questions, please check the main README.md or open an issue on the repository.
