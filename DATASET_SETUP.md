# How to Provide Your Dataset for Object Detection

## What You're Seeing

If you see this message:
```
⚠️  No valid dataset path provided

✓ Model initialized successfully!
Provide dataset path to begin training.
```

This means **the model is ready**, but you need to provide your dataset so it can actually train!

## Quick Start - 3 Steps

### Step 1: Organize Your Dataset

Choose one of these three formats:

#### Option A: YOLO Format (Easiest for Custom Data)
```
my_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        └── ...
```

**Label format** (one .txt file per image):
```
class_id center_x center_y width height
```
All coordinates are normalized (0-1 range).

**Example `image1.txt`** (for a dog at center with 2 classes):
```
0 0.5 0.5 0.3 0.4
```
- `0` = class ID (0 for first class, 1 for second class)
- `0.5 0.5` = center at 50% of image width/height
- `0.3 0.4` = box is 30% of width, 40% of height

#### Option B: Pascal VOC Format
```
my_dataset/
├── Annotations/
│   ├── image1.xml
│   ├── image2.xml
│   └── ...
├── JPEGImages/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ImageSets/
    └── Main/
        ├── train.txt
        └── val.txt
```

**Annotation XML example:**
```xml
<annotation>
  <object>
    <name>dog</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
</annotation>
```

#### Option C: COCO Format
```
my_dataset/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/  (or images/)
│   ├── image1.jpg
│   └── ...
└── val/
    ├── image1.jpg
    └── ...
```

### Step 2: Update Your Training Call

#### In GUI
When using the GUI, you need to:
1. Set **Dataset Path** to your dataset folder
2. Set **Dataset Format** to one of: `coco`, `voc`, or `yolo`
3. Set **Number of Classes** correctly (YOUR_CLASSES + 1)

#### Programmatically
```python
from code.train_detection import run_detection_training

run_detection_training(
    model_name='ssd300',
    num_classes=3,  # 2 object classes + 1 background
    batch_size=16,
    epochs=50,
    lr=0.001,
    data_path='E:/my_dataset',  # ← YOUR DATASET PATH
    dataset_format='yolo',      # ← YOUR FORMAT: 'coco', 'voc', or 'yolo'
    num_workers=4
)
```

### Step 3: Run Training

Once you've organized your dataset and updated the code, run training again!

## Class Mapping for VOC Format

If using VOC format with custom class names, you need to provide a class mapping:

```python
from code.detection_dataset import VOCDetectionDataset

# Define your class mapping
class_mapping = {
    'dog': 1,
    'cat': 2
}

# Use it when creating dataset
dataset = VOCDetectionDataset(
    data_path='E:/my_dataset',
    split='train',
    class_mapping=class_mapping
)
```

## Converting Your Data

If your data is in a different format, use the converter:

```python
# See examples/dataset_format_converter.py
```

Or create annotations manually using tools like:
- **LabelImg** (for VOC/YOLO format)
- **CVAT** (for COCO format)
- **Labelme** (for general annotations)

## Example: Creating YOLO Format Manually

For each image, create a matching .txt file:

**image1.jpg** (contains 1 dog):
```
# image1.txt
0 0.5 0.5 0.4 0.6
```

**image2.jpg** (contains 1 cat and 1 dog):
```
# image2.txt
0 0.3 0.4 0.2 0.3
1 0.7 0.6 0.25 0.35
```

Class IDs:
- 0 = first class (e.g., dog)
- 1 = second class (e.g., cat)

## Minimum Dataset Size

For good results, you need:
- **Minimum**: 100+ images per class
- **Recommended**: 500+ images per class
- **Best**: 1000+ images per class

When training from scratch (no pretrained weights), more data is better!

## Common Errors

### "Dataset path does not exist"
- Check the path is correct and exists
- Use absolute paths (e.g., `E:/datasets/my_data`)
- Check folder permissions

### "Could not find COCO annotation file"
- Make sure `annotations/instances_train.json` exists
- Check file naming matches expected format

### "Split file not found"
- For VOC: make sure `ImageSets/Main/train.txt` and `val.txt` exist
- These files should contain list of image IDs (without extensions)

### "Images directory not found"
- For YOLO: make sure `images/train/` and `images/val/` exist
- For COCO: make sure `train/` or `images/` folder exists

## Testing Your Dataset

Before starting long training, test that your dataset loads:

```python
from code.detection_dataset import create_detection_dataset

# Try loading dataset
dataset = create_detection_dataset(
    data_path='E:/my_dataset',
    dataset_format='yolo',
    split='train'
)

print(f"Dataset size: {len(dataset)}")

# Load first sample
image, target = dataset[0]
print(f"Image size: {image.size}")
print(f"Number of objects: {len(target['boxes'])}")
print(f"Labels: {target['labels']}")
```

## Ready to Train?

Once your dataset is ready:

1. ✅ Dataset organized in correct format
2. ✅ `data_path` points to your dataset
3. ✅ `dataset_format` matches your dataset ('coco', 'voc', or 'yolo')
4. ✅ `num_classes` = YOUR_CLASSES + 1

Run your training code again, and this time it should actually start training!

## Need Help?

See:
- `OBJECT_DETECTION_GUIDE.md` - Full guide with all details
- `CUSTOM_CLASSES_FIX.md` - Explanation of custom classes
- `examples/dataset_format_converter.py` - Dataset conversion utilities
