# Custom Classes Fix for Object Detection

## Problem

When training object detection models with a custom number of classes (e.g., 3 classes for 2 objects + background), you were getting an error:

```
✗ Detection training error: The parameter 'num_classes' expected value 91 but got 3 instead.
```

## Root Cause

The pretrained object detection models in torchvision (SSD300, RetinaNet, Faster R-CNN, FCOS, etc.) are trained on the COCO dataset, which has **91 classes** (80 object classes + 11 for various categories including background).

When you set `pretrained=True` and `num_classes=3`, torchvision tries to load pretrained weights that expect 91 output classes, which causes a mismatch error.

## Solution

The `code/object_detection_models.py` file has been updated to automatically disable pretrained weights when using a custom number of classes that doesn't match the COCO dataset (91 classes).

### What Changed

When you request a model with:
- `num_classes != 91` (custom number of classes)
- `pretrained=True`

The code will now:
1. Display a warning message
2. Automatically set `pretrained=False`
3. Train the model from scratch with your custom number of classes

```python
⚠ Warning: Pretrained weights are for 91 classes (COCO).
⚠ Training from scratch with 3 classes instead.
```

## How to Use

### For 2 Object Classes (e.g., "dog" and "cat")

Set **num_classes = 3** (2 objects + 1 background)

```python
# In the GUI or programmatically:
num_classes = 3  # 2 object classes + background
```

### For 5 Object Classes

Set **num_classes = 6** (5 objects + 1 background)

```python
num_classes = 6  # 5 object classes + background
```

### General Formula

```
num_classes = YOUR_OBJECT_CLASSES + 1
```

The "+1" is for the background class that object detection models use internally.

## Models Affected

This fix applies to all torchvision-based models:
- SSD300
- RetinaNet R50-FPN
- Faster R-CNN R50-FPN
- Cascade R-CNN R50-FPN
- FCOS R50-FPN
- ATSS R50-FPN

## Training Considerations

### Training from Scratch

When using custom classes with pretrained disabled:
- **Training will take longer** - You're training from random initialization
- **More data needed** - Pretrained models benefit from ImageNet/COCO pretraining
- **Lower initial accuracy** - The model starts with no knowledge
- **Adjust hyperparameters** - You may need more epochs or different learning rates

### Recommendations

1. **Use more epochs**: Start with 100-150 epochs instead of 50
2. **More data**: Aim for at least 1000+ images per class if possible
3. **Data augmentation**: Use augmentation to increase effective dataset size
4. **Lower learning rate initially**: Try 0.0001-0.0005 to start
5. **Patience**: Training from scratch needs more time to converge

### Alternative: Transfer Learning (Advanced)

If you want to use pretrained weights with custom classes, you can:
1. Load a model with 91 classes (pretrained)
2. Manually replace only the final classification layer
3. Freeze backbone layers initially
4. Fine-tune in stages

This is more complex but can give better results with less data. See advanced tutorials for implementation details.

## Example Usage

### GUI Usage

1. Open the GUI: `python run_gui.py`
2. Go to "Object Detection" tab
3. Select your model (e.g., SSD300)
4. **Set Number of Classes = YOUR_CLASSES + 1**
   - For 2 classes: Set to 3
   - For 5 classes: Set to 6
   - For 10 classes: Set to 11
5. Configure other parameters
6. Start training

You'll see the warning message that it's training from scratch.

### Programmatic Usage

```python
from code.object_detection_models import create_detection_model

# Create model for 2 object classes (+ background = 3 total)
model = create_detection_model('ssd300', num_classes=3, pretrained=True)
# Will automatically train from scratch with warning message
```

## Testing

After this fix:
- ✅ You can now train with any number of classes
- ✅ The error about "expected value 91" is resolved
- ✅ Models train from scratch with custom classes
- ✅ All torchvision detection models are supported

## Questions?

If you still encounter issues:
1. Check that you've calculated num_classes correctly (objects + 1)
2. Verify your dataset format matches expected format
3. Check dataset paths are correct
4. Review the training logs for other errors

For COCO dataset (91 classes), pretrained weights will still be used normally.
