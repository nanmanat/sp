# Object Detection Queue System

## Overview

The Object Detection Queue System provides a robust, thread-safe queue implementation for processing object detection tasks on images. It supports both synchronous and asynchronous processing, priority queuing, and batch operations.

## Features

- **Queue-based Processing**: Add multiple detection tasks and process them sequentially
- **Priority Queue**: Process important images first with priority levels
- **Batch Processing**: Efficiently add and process multiple images at once
- **Callbacks**: Custom callbacks for task start, completion, and error handling
- **Thread-safe**: Safe for concurrent operations
- **Statistics**: Track processing times, success/failure rates
- **Result Storage**: Save detection results to JSON

## Components

### 1. DetectionQueue (`code/detection_queue.py`)

Main queue manager for processing detection tasks.

**Key Features:**
- Thread-safe task management
- Configurable queue size
- Progress tracking and statistics
- Callback system for monitoring

**Basic Usage:**
```python
from code.detection_queue import DetectionQueue
from code.detection_inference import DetectionInference

# Initialize detector
detector = DetectionInference(
    model_name='faster_rcnn_r50_fpn',
    num_classes=91,
    device='cuda'
)

# Initialize queue
queue = DetectionQueue(max_queue_size=100)
queue.set_detector(detector)

# Add tasks
task_id = queue.add_task('path/to/image.jpg')

# Start processing
queue.start_processing()
```

### 2. PriorityDetectionQueue

Extended queue with priority support.

**Usage:**
```python
from code.detection_queue import PriorityDetectionQueue

queue = PriorityDetectionQueue(max_queue_size=100)
queue.set_detector(detector)

# Add tasks with priorities (lower number = higher priority)
queue.add_task('urgent_image.jpg', priority=0)
queue.add_task('normal_image.jpg', priority=1)
```

### 3. DetectionInference (`code/detection_inference.py`)

Inference engine for running object detection.

**Supported Models:**
- Faster R-CNN (faster_rcnn_r50_fpn)
- SSD300 (ssd300)
- RetinaNet (retinanet_r50_fpn)
- EfficientDet (efficientdet_d0 to efficientdet_d3)
- FCOS (fcos_r50_fpn)

**Usage:**
```python
from code.detection_inference import DetectionInference

detector = DetectionInference(
    model_name='faster_rcnn_r50_fpn',
    num_classes=91,
    model_path='path/to/weights.pth',  # Optional
    device='cuda',
    confidence_threshold=0.5
)

# Run detection
result = detector.detect('image.jpg')
print(f"Found {result['num_detections']} objects")

# Visualize
detector.visualize_detections(
    'image.jpg',
    detections=result,
    save_path='output.jpg'
)
```

## Queue Callbacks

Set custom callbacks to monitor processing:

```python
def on_task_start(task):
    print(f"Starting: {task.task_id}")

def on_task_complete(task):
    print(f"Completed: {task.task_id} - {task.result['num_detections']} objects")

def on_task_error(task, error):
    print(f"Error: {task.task_id} - {error}")

def on_queue_empty():
    print("Queue is empty!")

queue.on_task_start = on_task_start
queue.on_task_complete = on_task_complete
queue.on_task_error = on_task_error
queue.on_queue_empty = on_queue_empty
```

## Batch Processing

Process multiple images efficiently:

```python
# Add multiple tasks at once
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
metadata_list = [
    {'source': 'camera1'},
    {'source': 'camera2'},
    {'source': 'camera3'}
]

task_ids = queue.add_batch_tasks(image_paths, metadata_list)

# Start processing
queue.start_processing()

# Monitor progress
while queue.is_running:
    info = queue.get_queue_info()
    print(f"Progress: {info['completed']}/{len(task_ids)}")
    time.sleep(1)
```

## Queue Information

Get real-time queue statistics:

```python
info = queue.get_queue_info()
print(f"Queue size: {info['queue_size']}")
print(f"Completed: {info['completed']}")
print(f"Failed: {info['failed']}")
print(f"Average time: {info['avg_processing_time']:.3f}s")
```

## Task Status

Check individual task status:

```python
task_id = queue.add_task('image.jpg')

# Get status
status = queue.get_task_status(task_id)
print(f"Status: {status['status']}")  # pending, processing, completed, failed

# Once completed
if status['status'] == 'completed':
    result = status['result']
    print(f"Found {result['num_detections']} objects")
```

## Save Results

Save all detection results to JSON:

```python
queue.save_results('detection_results.json')
```

Output format:
```json
[
  {
    "task_id": "task_20250110_120000_123456",
    "image_path": "image.jpg",
    "status": "completed",
    "result": {
      "boxes": [[x1, y1, x2, y2], ...],
      "scores": [0.95, 0.87, ...],
      "labels": [1, 3, ...],
      "num_detections": 5
    },
    "created_at": "2025-01-10T12:00:00",
    "completed_at": "2025-01-10T12:00:02"
  }
]
```

## Visualization

Visualize detection results with bounding boxes:

```python
from code.detection_inference import get_coco_class_names

class_names = get_coco_class_names()

detector.visualize_detections(
    'image.jpg',
    class_names=class_names,
    save_path='output.jpg',
    show=True
)
```

## Examples

Run the provided examples:

```bash
python run_detection_queue.py
```

Available examples:
1. **Basic Queue**: Simple sequential processing
2. **Priority Queue**: Process high-priority images first
3. **Batch Processing**: Efficiently process large image sets
4. **Queue with Visualization**: Process and save visualizations

## Performance Tips

1. **Batch Size**: Use larger batches for GPU efficiency
   ```python
   from code.detection_inference import BatchDetectionInference
   
   detector = BatchDetectionInference(
       batch_size=8,
       model_name='faster_rcnn_r50_fpn'
   )
   ```

2. **Model Selection**: Choose model based on speed/accuracy tradeoff
   - Fast: `ssd300`, `efficientdet_d0`
   - Balanced: `faster_rcnn_r50_fpn`
   - Accurate: `efficientdet_d3`, `retinanet_r50_fpn`

3. **Confidence Threshold**: Adjust based on use case
   ```python
   detector.set_confidence_threshold(0.7)  # Higher = fewer false positives
   ```

4. **Queue Size**: Set appropriate queue size
   ```python
   queue = DetectionQueue(max_queue_size=1000)  # For large batches
   ```

## Integration with GUI

The detection queue is already integrated into the GUI (`code/gui.py`). You can use it from the "Object Detection" tab.

## API Reference

### DetectionQueue

**Methods:**
- `add_task(image_path, task_id, metadata)`: Add a single task
- `add_batch_tasks(image_paths, metadata_list)`: Add multiple tasks
- `start_processing()`: Start queue processing
- `stop_processing(wait)`: Stop processing
- `get_task_status(task_id)`: Get task information
- `get_queue_info()`: Get queue statistics
- `clear_completed_tasks()`: Remove completed tasks from memory
- `save_results(output_file)`: Save results to JSON

### DetectionInference

**Methods:**
- `detect(image_path, conf_threshold, return_timing)`: Run detection on single image
- `detect_batch(image_paths, conf_threshold)`: Run detection on multiple images
- `visualize_detections(image_path, detections, class_names, save_path, show)`: Visualize results
- `set_confidence_threshold(threshold)`: Update confidence threshold
- `get_model_info()`: Get model information

## Custom Models

Load your own trained models:

```python
detector = DetectionInference(
    model_name='faster_rcnn_r50_fpn',
    num_classes=5,  # Your number of classes
    model_path='saved_models/my_model/best_model.pth',
    device='cuda'
)
```

## Error Handling

The queue system handles errors gracefully:

```python
def on_task_error(task, error):
    print(f"Task {task.task_id} failed: {error}")
    # Log error, retry, or take other action

queue.on_task_error = on_task_error
```

## Thread Safety

All queue operations are thread-safe. You can add tasks from multiple threads:

```python
import threading

def add_images(image_list):
    for img in image_list:
        queue.add_task(img)

# Start multiple threads adding tasks
threads = [
    threading.Thread(target=add_images, args=(images1,)),
    threading.Thread(target=add_images, args=(images2,))
]

for t in threads:
    t.start()
```

## License

Part of the SP project.
