# EfficientDet Target Format Fix

## Problem
When training EfficientDet models, encountered error:
```
❌ Error during training: list indices must be integers or slices, not str
```

## Root Cause
EfficientDet models (using `DetBenchTrain` wrapper from `effdet` library) require a different target format than torchvision detection models:

**Torchvision Format** (list of dicts):
```python
targets = [
    {'boxes': tensor([N, 4]), 'labels': tensor([N])},  # Image 1
    {'boxes': tensor([M, 4]), 'labels': tensor([M])},  # Image 2
    ...
]
```

**EfficientDet Format** (single dict):
```python
targets = {
    'bbox': tensor([total_boxes, 5]),  # [batch_idx, x1, y1, x2, y2]
    'cls': tensor([total_boxes]),       # class labels
    'img_scale': tensor([B, 2])         # image scales
}
```

## Solution

### 1. Updated `collate_fn_efficientdet` in `detection_dataset.py`
The collate function now converts targets from torchvision format to effdet format:
- Extracts boxes and labels from each sample
- Adds batch indices to boxes
- Concatenates all boxes across the batch
- Returns single dict with 'bbox', 'cls', 'img_scale'

### 2. Updated Training Loop in `train_detection.py`
Modified both `_train_epoch` and `_validate_epoch` to handle two formats:
```python
if isinstance(targets, dict) and 'bbox' in targets:
    # EfficientDet format - single dict
    targets = {k: v.to(device) for k, v in targets.items()}
else:
    # Torchvision format - list of dicts
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
```

## Target Format Details

### Torchvision Models (SSD, Faster R-CNN, RetinaNet, FCOS)
```python
targets = [
    {
        'boxes': tensor([[x1, y1, x2, y2], ...]),  # [N, 4]
        'labels': tensor([class_id, ...]),          # [N]
        'image_id': tensor([img_id])
    },
    ...  # one dict per image in batch
]
```

### EfficientDet Models (D0-D3)
```python
targets = {
    'bbox': tensor([
        [batch_idx, x1, y1, x2, y2],  # Box 1 from image batch_idx
        [batch_idx, x1, y1, x2, y2],  # Box 2 from image batch_idx
        ...
    ]),  # [total_boxes, 5]
    'cls': tensor([class_id, ...]),    # [total_boxes]
    'img_scale': tensor([[h, w], ...])  # [B, 2]
}
```

## Files Modified
1. **code/detection_dataset.py**: `collate_fn_efficientdet()` - Converts target format
2. **code/train_detection.py**: `_train_epoch()` and `_validate_epoch()` - Handles both formats

## Result
✅ EfficientDet models now train successfully with proper target format
✅ Backward compatibility maintained with torchvision models
✅ Automatic format detection based on target structure

## Usage
No changes needed from user perspective. The system automatically:
1. Detects model type (efficientdet vs others)
2. Uses appropriate collate function
3. Handles target format in training loop

```python
# Works for both model types
run_detection_training(
    model_name='efficientdet_d0',  # or 'ssd300', 'faster_rcnn_r50_fpn', etc.
    num_classes=2,
    batch_size=16,
    epochs=50,
    data_path='E:/Buoy/dataset_coco',
    dataset_format='coco'
)
```
