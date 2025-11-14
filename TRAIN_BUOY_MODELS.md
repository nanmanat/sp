# Quick Start: Train Detection Models with Your Buoy Datasets

## Your Datasets

Located in `E:\dataset_sep_1_2_aug\`:

1. **dataset_sep_1_2_aug_coco** - Combined dataset (Sep 1 + 2 Augmented)
2. **dataset_sep_1_aug_coco** - Dataset Sep 1 Augmented  
3. **dataset_sep_2_aug_coco** - Dataset Sep 2 Augmented

Classes: **drowning** and **swimming** (+ background = 3 classes total)

## Quick Training Commands

### Option 1: Train All Models on All Datasets (Recommended)

```bash
python train_detection_with_dataset_name.py
```

This will train 9 models (3 models × 3 datasets):
- `ssd300_dataset_sep_1_2_aug_coco`
- `ssd300_dataset_sep_1_aug_coco`
- `ssd300_dataset_sep_2_aug_coco`
- `faster_rcnn_r50_fpn_dataset_sep_1_2_aug_coco`
- `faster_rcnn_r50_fpn_dataset_sep_1_aug_coco`
- `faster_rcnn_r50_fpn_dataset_sep_2_aug_coco`
- `retinanet_r50_fpn_dataset_sep_1_2_aug_coco`
- `retinanet_r50_fpn_dataset_sep_1_aug_coco`
- `retinanet_r50_fpn_dataset_sep_2_aug_coco`

### Option 2: Train Single Model

```bash
python train_detection_with_dataset_name.py --single
```

This trains just `ssd300` on `dataset_sep_1_2_aug_coco`

### Option 3: Custom Python Script

```python
from code.train_detection import run_detection_training

# Train SSD300 on combined dataset
run_detection_training(
    model_name='ssd300',
    num_classes=3,  # background + drowning + swimming
    batch_size=4,
    epochs=50,
    lr=0.001,
    num_workers=4,
    data_path='E:\\dataset_sep_1_2_aug\\dataset_sep_1_2_aug_coco',
    dataset_format='coco'
    # dataset_name is auto-detected as 'dataset_sep_1_2_aug_coco'
)
```

**Result:** Saved to `./saved_models/ssd300_dataset_sep_1_2_aug_coco/best_model.pth`

## Model Locations After Training

All models saved in `./saved_models/`:

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
└── ...
```

## Using Trained Models

```python
from code.detection_inference import DetectionInference

# Load your trained model
detector = DetectionInference(
    model_name='ssd300',
    model_path='./saved_models/ssd300_dataset_sep_1_2_aug_coco/best_model.pth',
    num_classes=3
)

# Detect drowning/swimming
results = detector.predict('path/to/pool_image.jpg')

# Results contain bounding boxes, classes, and confidence scores
for bbox, label, score in zip(results['boxes'], results['labels'], results['scores']):
    print(f"Detected: {label} (confidence: {score:.2f})")
```

## Recommended Training Order

1. **Start with SSD300** (fastest)
   ```bash
   python -c "from code.train_detection import run_detection_training; run_detection_training('ssd300', 3, data_path='E:\\dataset_sep_1_2_aug\\dataset_sep_1_2_aug_coco', dataset_format='coco')"
   ```

2. **Then Faster R-CNN** (better accuracy)
   ```bash
   python -c "from code.train_detection import run_detection_training; run_detection_training('faster_rcnn_r50_fpn', 3, data_path='E:\\dataset_sep_1_2_aug\\dataset_sep_1_2_aug_coco', dataset_format='coco')"
   ```

3. **Finally RetinaNet** (good balance)
   ```bash
   python -c "from code.train_detection import run_detection_training; run_detection_training('retinanet_r50_fpn', 3, data_path='E:\\dataset_sep_1_2_aug\\dataset_sep_1_2_aug_coco', dataset_format='coco')"
   ```

## Model Characteristics

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| SSD300 | ⚡⚡⚡ Fast | ⭐⭐ Good | Real-time detection |
| Faster R-CNN | ⚡ Slow | ⭐⭐⭐ Best | Highest accuracy |
| RetinaNet | ⚡⚡ Medium | ⭐⭐⭐ Excellent | Balance of speed/accuracy |

## Training Parameters

Default settings (you can adjust these):
- **num_classes**: 3 (background, drowning, swimming)
- **batch_size**: 4
- **epochs**: 50
- **lr**: 0.001
- **num_workers**: 4

To change parameters:
```python
run_detection_training(
    model_name='ssd300',
    num_classes=3,
    batch_size=8,      # ← Increase if you have more GPU memory
    epochs=100,        # ← More epochs for better training
    lr=0.0005,         # ← Lower learning rate for fine-tuning
    data_path='E:\\dataset_sep_1_2_aug\\dataset_sep_1_2_aug_coco',
    dataset_format='coco'
)
```

## Troubleshooting

### Out of Memory
Reduce batch_size:
```python
batch_size=2  # or 1
```

### Training Too Slow
Use smaller model or reduce image size:
```python
model_name='ssd300'  # Fastest model
```

### Need Better Accuracy
- Increase epochs: `epochs=100`
- Use larger model: `model_name='faster_rcnn_r50_fpn'`
- Train on combined dataset: `dataset_sep_1_2_aug_coco`

## Next Steps

After training:
1. Check model performance in logs
2. Use best performing model for inference
3. Deploy to detection queue system
4. Integrate with GUI for real-time detection

For more details, see `DATASET_MODEL_NAMING.md`
