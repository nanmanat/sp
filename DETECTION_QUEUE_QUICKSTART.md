# Detection Queue System - Quick Start

## What is it?

A complete queue-based system for processing object detection tasks on images. Perfect for:
- Batch processing large image datasets
- Real-time detection pipelines
- Prioritized image processing
- Multi-threaded detection workflows

## Installation

Already included in the SP project. Just ensure you have the required dependencies:

```bash
pip install torch torchvision pillow matplotlib numpy
```

For EfficientDet models:
```bash
pip install effdet
```

## Quick Start

### 1. Basic Detection

```python
from code.detection_inference import DetectionInference

# Initialize detector
detector = DetectionInference(
    model_name='faster_rcnn_r50_fpn',
    num_classes=91,
    device='cuda'
)

# Run detection
result = detector.detect('image.jpg')
print(f"Found {result['num_detections']} objects")
```

### 2. Queue Processing

```python
from code.detection_queue import DetectionQueue
from code.detection_inference import DetectionInference

# Setup
detector = DetectionInference(model_name='faster_rcnn_r50_fpn')
queue = DetectionQueue()
queue.set_detector(detector)

# Add images
queue.add_task('image1.jpg')
queue.add_task('image2.jpg')
queue.add_task('image3.jpg')

# Process
queue.start_processing()
```

### 3. Batch Processing

```python
# Add many images at once
image_paths = ['img1.jpg', 'img2.jpg', ..., 'img100.jpg']
task_ids = queue.add_batch_tasks(image_paths)

# Monitor progress
while queue.is_running:
    info = queue.get_queue_info()
    print(f"Progress: {info['completed']}/{len(task_ids)}")
    time.sleep(1)
```

### 4. Priority Queue

```python
from code.detection_queue import PriorityDetectionQueue

queue = PriorityDetectionQueue()
queue.set_detector(detector)

# Higher priority tasks (lower number) processed first
queue.add_task('urgent.jpg', priority=0)
queue.add_task('normal.jpg', priority=1)
queue.add_task('low.jpg', priority=2)
```

## Available Models

- **faster_rcnn_r50_fpn** - Balanced speed/accuracy (recommended)
- **ssd300** - Fast, good for real-time
- **efficientdet_d0** to **efficientdet_d3** - Scalable efficiency
- **retinanet_r50_fpn** - High accuracy
- **fcos_r50_fpn** - Anchor-free detection

## Key Features

✅ Thread-safe queue management
✅ Priority-based processing
✅ Batch operations
✅ Progress monitoring
✅ Callback system
✅ Result storage (JSON)
✅ Visualization support
✅ Error handling

## Examples

Run the comprehensive examples:

```bash
python run_detection_queue.py
```

Or test the system:

```bash
python test_detection_queue.py
```

## Callbacks

Monitor processing with custom callbacks:

```python
def on_complete(task):
    print(f"Done: {task.task_id} - {task.result['num_detections']} objects")

queue.on_task_complete = on_complete
```

## Visualization

Visualize detection results:

```python
detector.visualize_detections(
    'image.jpg',
    save_path='output.jpg',
    show=True
)
```

## Custom Models

Use your trained models:

```python
detector = DetectionInference(
    model_name='faster_rcnn_r50_fpn',
    num_classes=5,  # Your classes
    model_path='saved_models/my_model/best_model.pth'
)
```

## Documentation

Full documentation: `DETECTION_QUEUE_GUIDE.md`

## Files Created

- `code/detection_queue.py` - Queue management system
- `code/detection_inference.py` - Inference engine
- `examples/detection_queue_example.py` - Usage examples
- `run_detection_queue.py` - Quick runner
- `test_detection_queue.py` - Test suite
- `DETECTION_QUEUE_GUIDE.md` - Complete documentation
- `DETECTION_QUEUE_QUICKSTART.md` - This file

## Support

For issues or questions, refer to the full documentation in `DETECTION_QUEUE_GUIDE.md`.
