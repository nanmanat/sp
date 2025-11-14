# Dataset-Specific Model Naming for Object Detection

## Overview

Object detection models are now automatically saved with dataset-specific names to help you organize and identify which dataset each model was trained on.

## Model Naming Convention

**Format:** `{model_name}_{dataset_name}/best_model.pth`

**Examples:**
- `ssd300_dataset_sep_1_2_aug_coco/best_model.pth`
- `faster_rcnn_r50_fpn_dataset_sep_1_aug_coco/best_model.pth`
- `retinanet_r50_fpn_dataset_sep_2_aug_coco/best_model.pth`

## Usage

### 1. Automatic Dataset Name Detection

The dataset name is automatically extracted from the `data_path` if not explicitly provided:

```python
from code.train_detection import run_detection_training

run_detection_training(
    model_name='ssd300',
    num_classes=3,
    data_path='E:\\dataset_sep_1_2_aug\\dataset_sep_1_2_aug_coco',
    # dataset_name is auto-detected as 'dataset_sep_1_2_aug_coco'
)
```

**Result:** Model saved to `./saved_models/ssd300_dataset_sep_1_2_aug_coco/best_model.pth`

### 2. Explicit Dataset Name

You can explicitly specify the dataset name:

```python
run_detection_training(
    model_name='faster_rcnn_r50_fpn',
    num_classes=3,
    data_path='E:\\dataset_sep_1_2_aug\\dataset_sep_1_aug_coco',
    dataset_name='sep1_aug'  # Custom short name
)
```

**Result:** Model saved to `./saved_models/faster_rcnn_r50_fpn_sep1_aug/best_model.pth`

### 3. Without Dataset Name (Backward Compatible)

If you don't provide a dataset name and no data_path, it falls back to the old behavior:

```python
run_detection_training(
    model_name='ssd300',
    num_classes=91,
    # No data_path or dataset_name
)
```

**Result:** Model saved to `./saved_models/ssd300/best_model.pth`

## Complete Training Example

### Training on Your Buoy Datasets

```python
from code.train_detection import run_detection_training

# Configuration
datasets = [
    {
        'name': 'dataset_sep_1_2_aug_coco',
        'path': 'E:\\dataset_sep_1_2_aug\\dataset_sep_1_2_aug_coco'
    },
    {
        'name': 'dataset_sep_1_aug_coco',
        'path': 'E:\\dataset_sep_1_2_aug\\dataset_sep_1_aug_coco'
    },
    {
        'name': 'dataset_sep_2_aug_coco',
        'path': 'E:\\dataset_sep_1_2_aug\\dataset_sep_2_aug_coco'
    }
]

models = ['ssd300', 'faster_rcnn_r50_fpn', 'retinanet_r50_fpn']

# Train all combinations
for dataset in datasets:
    for model_name in models:
        print(f"Training {model_name} on {dataset['name']}...")
        
        run_detection_training(
            model_name=model_name,
            num_classes=3,  # background + drowning + swimming
            batch_size=4,
            epochs=50,
            lr=0.001,
            data_path=dataset['path'],
            dataset_format='coco',
            dataset_name=dataset['name']
        )
        
        print(f"✓ Saved to: ./saved_models/{model_name}_{dataset['name']}/best_model.pth")
```

### Quick Single Training Example

Use the provided example script:

```bash
# Train single model
python train_detection_with_dataset_name.py --single

# Train all combinations (3 datasets × 3 models = 9 trainings)
python train_detection_with_dataset_name.py
```

## Using Trained Models

### Loading a Model for Inference

```python
from code.detection_inference import DetectionInference

# Load model trained on specific dataset
model_path = './saved_models/ssd300_dataset_sep_1_2_aug_coco/best_model.pth'
detector = DetectionInference(
    model_name='ssd300',
    model_path=model_path,
    num_classes=3
)

# Run prediction
results = detector.predict('path/to/image.jpg')
```

## Directory Structure

After training, your `saved_models` directory will look like:

```
saved_models/
├── ssd300_dataset_sep_1_2_aug_coco/
│   └── best_model.pth
├── ssd300_dataset_sep_1_aug_coco/
│   └── best_model.pth
├── ssd300_dataset_sep_2_aug_coco/
│   └── best_model.pth
├── faster_rcnn_r50_fpn_dataset_sep_1_2_aug_coco/
│   └── best_model.pth
├── faster_rcnn_r50_fpn_dataset_sep_1_aug_coco/
│   └── best_model.pth
└── ...
```

## GUI Usage

The GUI automatically uses the dataset name from the data path field. Simply:

1. Select your model (e.g., `ssd300`)
2. Set the data path (e.g., `E:\dataset_sep_1_2_aug\dataset_sep_1_2_aug_coco`)
3. Click "Start Training"

The model will automatically be saved as `ssd300_dataset_sep_1_2_aug_coco/best_model.pth`

## API Reference

### `DetectionTrainer.__init__`

```python
DetectionTrainer(
    model_name,
    num_classes=91,
    device='cuda',
    dataset_name=None  # NEW parameter
)
```

**Parameters:**
- `dataset_name` (str, optional): Name of the dataset for model naming. If None, model is saved with just the model name.

### `run_detection_training`

```python
run_detection_training(
    model_name='faster_rcnn_r50_fpn',
    num_classes=91,
    batch_size=4,
    epochs=50,
    lr=0.001,
    num_workers=4,
    data_path=None,
    dataset_format='coco',
    save_dir='./saved_models',
    dataset_name=None  # NEW parameter
)
```

**Parameters:**
- `dataset_name` (str, optional): Name of the dataset for model naming. If None and `data_path` is provided, the dataset name is auto-detected from the path.

## Benefits

1. **Easy Organization**: Quickly identify which dataset a model was trained on
2. **Prevent Overwriting**: Different datasets produce different model files
3. **Reproducibility**: Clear naming helps track experiments
4. **Comparison**: Easy to compare models trained on different datasets

## Notes

- Dataset name is automatically extracted from the last directory in `data_path`
- You can override the auto-detected name by explicitly setting `dataset_name`
- Backward compatible: works without dataset names (original behavior)
- Works with GUI, queue system, and direct API calls
