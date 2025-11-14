# Object Detection Queue System 🎯

A complete, production-ready queue system for processing object detection tasks.

## Quick Start

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

# Process
queue.start_processing()
```

## Features

✅ **Thread-Safe Queue** - Safe concurrent operations  
✅ **Priority Support** - Process important images first  
✅ **Batch Processing** - Add multiple images at once  
✅ **Progress Monitoring** - Real-time statistics  
✅ **Callback System** - Custom event handlers  
✅ **Result Storage** - Export to JSON  
✅ **Visualization** - Draw bounding boxes  
✅ **Multiple Models** - Faster R-CNN, SSD, EfficientDet, etc.

## Files

| File | Description |
|------|-------------|
| `code/detection_queue.py` | Queue management system |
| `code/detection_inference.py` | Inference engine |
| `examples/detection_queue_example.py` | Usage examples |
| `run_detection_queue.py` | Quick runner |
| `test_detection_queue_basic.py` | Test suite |
| `DETECTION_QUEUE_GUIDE.md` | Complete documentation |
| `DETECTION_QUEUE_QUICKSTART.md` | Quick reference |

## Installation

```bash
pip install torch torchvision pillow matplotlib
```

## Usage Examples

### Basic Queue
```python
queue = DetectionQueue()
queue.set_detector(detector)
queue.add_task('image.jpg')
queue.start_processing()
```

### Priority Queue
```python
priority_queue = PriorityDetectionQueue()
priority_queue.add_task('urgent.jpg', priority=0)
priority_queue.add_task('normal.jpg', priority=1)
```

### Batch Processing
```python
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
task_ids = queue.add_batch_tasks(image_paths)
```

### With Callbacks
```python
def on_complete(task):
    print(f"Found {task.result['num_detections']} objects")

queue.on_task_complete = on_complete
```

### Monitor Progress
```python
while queue.is_running:
    info = queue.get_queue_info()
    print(f"Completed: {info['completed']}/{info['queue_size']}")
```

## Run Examples

```bash
python run_detection_queue.py
```

Choose from:
1. Basic Queue
2. Priority Queue
3. Batch Processing
4. Queue with Visualization

## Run Tests

```bash
python test_detection_queue_basic.py
```

## Models Supported

- `faster_rcnn_r50_fpn` - Balanced (recommended)
- `ssd300` - Fast
- `efficientdet_d0` to `efficientdet_d3` - Efficient
- `retinanet_r50_fpn` - Accurate
- `fcos_r50_fpn` - Anchor-free

## API Quick Reference

### DetectionQueue

```python
# Create queue
queue = DetectionQueue(max_queue_size=1000)

# Add tasks
task_id = queue.add_task(image_path, metadata={})
task_ids = queue.add_batch_tasks(image_paths, metadata_list)

# Control
queue.start_processing()
queue.stop_processing(wait=True)

# Monitor
status = queue.get_task_status(task_id)
info = queue.get_queue_info()

# Save
queue.save_results('results.json')
```

### DetectionInference

```python
# Create detector
detector = DetectionInference(
    model_name='faster_rcnn_r50_fpn',
    num_classes=91,
    device='cuda',
    confidence_threshold=0.5
)

# Detect
result = detector.detect(image_path)
results = detector.detect_batch(image_paths)

# Visualize
detector.visualize_detections(
    image_path,
    save_path='output.jpg'
)
```

## Documentation

📖 **[Complete Guide](DETECTION_QUEUE_GUIDE.md)** - Full documentation  
⚡ **[Quick Start](DETECTION_QUEUE_QUICKSTART.md)** - Quick reference  
📊 **[Implementation](DETECTION_QUEUE_IMPLEMENTATION.md)** - Technical details

## Examples

See `examples/detection_queue_example.py` for:
- Basic queue usage
- Priority processing
- Batch operations
- Visualization

## Status

✅ **PRODUCTION READY**

All tests passing. Fully documented. Ready to use.

## License

Part of the SP project.
