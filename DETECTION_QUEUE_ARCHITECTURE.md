# Object Detection Queue - System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION QUEUE SYSTEM                        │
└─────────────────────────────────────────────────────────────────┘

┌───────────────┐     ┌──────────────────┐     ┌─────────────────┐
│               │     │                  │     │                 │
│  Image Input  │────▶│ Detection Queue  │────▶│  Detector       │
│               │     │                  │     │  (Inference)    │
└───────────────┘     └──────────────────┘     └─────────────────┘
                              │                         │
                              │                         │
                              ▼                         ▼
                      ┌──────────────────┐     ┌─────────────────┐
                      │                  │     │                 │
                      │  Task Manager    │     │  Results        │
                      │  (Thread-safe)   │     │  (Bounding boxes)│
                      │                  │     │                 │
                      └──────────────────┘     └─────────────────┘
                              │                         │
                              │                         │
                              ▼                         ▼
                      ┌──────────────────┐     ┌─────────────────┐
                      │                  │     │                 │
                      │  Statistics      │     │  Visualization  │
                      │  Progress        │     │  JSON Export    │
                      │  Callbacks       │     │                 │
                      └──────────────────┘     └─────────────────┘
```

## Component Details

### 1. Input Layer
```
┌─────────────────────────────────────────────────────┐
│ IMAGE SOURCES                                       │
├─────────────────────────────────────────────────────┤
│ • Single images (add_task)                         │
│ • Batch images (add_batch_tasks)                   │
│ • Camera streams                                    │
│ • File directories                                  │
│ • Priority-based input (PriorityDetectionQueue)    │
└─────────────────────────────────────────────────────┘
```

### 2. Queue Layer
```
┌─────────────────────────────────────────────────────┐
│ QUEUE MANAGEMENT                                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Task 1  │→ │  Task 2  │→ │  Task 3  │  ...   │
│  └──────────┘  └──────────┘  └──────────┘        │
│                                                     │
│  Status: pending → processing → completed/failed   │
│                                                     │
│  Thread-safe operations:                           │
│  • Add tasks                                       │
│  • Remove tasks                                    │
│  • Query status                                    │
│  • Update statistics                               │
└─────────────────────────────────────────────────────┘
```

### 3. Processing Layer
```
┌─────────────────────────────────────────────────────┐
│ DETECTION INFERENCE                                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────┐         │
│  │  Load Image                          │         │
│  └──────────┬───────────────────────────┘         │
│             ▼                                      │
│  ┌──────────────────────────────────────┐         │
│  │  Transform (Normalize, Resize, etc) │         │
│  └──────────┬───────────────────────────┘         │
│             ▼                                      │
│  ┌──────────────────────────────────────┐         │
│  │  Model Inference                     │         │
│  │  (Faster R-CNN, SSD, EfficientDet)  │         │
│  └──────────┬───────────────────────────┘         │
│             ▼                                      │
│  ┌──────────────────────────────────────┐         │
│  │  Post-processing                     │         │
│  │  (NMS, Confidence Filtering)        │         │
│  └──────────┬───────────────────────────┘         │
│             ▼                                      │
│  ┌──────────────────────────────────────┐         │
│  │  Return Results                      │         │
│  │  {boxes, scores, labels}            │         │
│  └──────────────────────────────────────┘         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 4. Output Layer
```
┌─────────────────────────────────────────────────────┐
│ RESULTS & MONITORING                                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Detection Results:                                │
│  • Bounding boxes [x1, y1, x2, y2]                │
│  • Confidence scores                               │
│  • Class labels                                    │
│  • Number of detections                            │
│                                                     │
│  Visualization:                                    │
│  • Draw boxes on images                            │
│  • Add class labels                                │
│  • Save annotated images                           │
│                                                     │
│  Export:                                           │
│  • JSON format                                     │
│  • Task metadata                                   │
│  • Processing times                                │
│                                                     │
│  Callbacks:                                        │
│  • on_task_start                                   │
│  • on_task_complete                                │
│  • on_task_error                                   │
│  • on_queue_empty                                  │
└─────────────────────────────────────────────────────┘
```

## Data Flow

### Standard Queue Flow
```
Image → Queue.add_task() → Task created → Added to queue
                                              ↓
                                      Worker thread picks task
                                              ↓
                                      Task status: processing
                                              ↓
                                      Detector.detect()
                                              ↓
                                      Results returned
                                              ↓
                                      Task status: completed
                                              ↓
                                      Callback fired
                                              ↓
                                      Statistics updated
```

### Priority Queue Flow
```
Images with priorities → PriorityQueue.add_task(priority=X)
                                              ↓
                              Tasks sorted by priority
                              (0 = highest, 9 = lowest)
                                              ↓
                              High priority tasks processed first
                                              ↓
                              Standard detection flow
```

### Batch Processing Flow
```
Multiple images → Queue.add_batch_tasks()
                              ↓
                  Create multiple tasks in parallel
                              ↓
                  All tasks added to queue
                              ↓
                  Process sequentially (or parallel batches)
                              ↓
                  Monitor progress with get_queue_info()
```

## Threading Model

```
┌─────────────────────────────────────────────────────┐
│ MAIN THREAD                                         │
├─────────────────────────────────────────────────────┤
│ • Add tasks to queue                                │
│ • Query status                                      │
│ • Get statistics                                    │
│ • Start/stop processing                             │
└─────────────────────────────────────────────────────┘
                      ↓ ↑
                Thread-safe operations (locks)
                      ↓ ↑
┌─────────────────────────────────────────────────────┐
│ WORKER THREAD (Background)                          │
├─────────────────────────────────────────────────────┤
│ • Get task from queue                               │
│ • Update task status                                │
│ • Run detection                                     │
│ • Fire callbacks                                    │
│ • Update statistics                                 │
│ • Loop until stopped or queue empty                 │
└─────────────────────────────────────────────────────┘
```

## Task Lifecycle

```
┌──────────┐
│ CREATED  │
└────┬─────┘
     │ add_task()
     ▼
┌──────────┐
│ PENDING  │ ◀──────────────┐
└────┬─────┘                │
     │ dequeue              │ Priority reordering
     ▼                      │
┌──────────┐                │
│PROCESSING│                │
└────┬─────┴────────────────┘
     │ detect()
     ▼
┌──────────┬──────────┐
│COMPLETED │  FAILED  │
└──────────┴──────────┘
     │          │
     │          │ on_task_error
     │          ▼
     │    ┌──────────┐
     │    │  LOGGED  │
     │    └──────────┘
     │
     │ on_task_complete
     ▼
┌──────────┐
│ EXPORTED │ (JSON)
└──────────┘
```

## Class Hierarchy

```
DetectionTask
├── Properties: task_id, image_path, metadata, status, result
└── Methods: to_dict()

DetectionQueue (Base)
├── Properties: queue, tasks, is_running, detector
├── Methods: add_task, start_processing, stop_processing
├── Callbacks: on_task_start, on_task_complete, on_task_error
└── Statistics: total_processed, processing_times

PriorityDetectionQueue (extends DetectionQueue)
└── Uses priority queue instead of FIFO

DetectionInference
├── Properties: model, device, confidence_threshold
├── Methods: detect, detect_batch, visualize_detections
└── Support: Multiple model architectures

BatchDetectionInference (extends DetectionInference)
└── Optimized for batch processing
```

## Integration Points

```
┌─────────────────────────────────────────────────────┐
│ EXISTING SP PROJECT                                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  code/gui.py          ─────────────┐               │
│  (GUI Interface)                   │               │
│                                    ▼               │
│                          ┌──────────────────┐      │
│                          │ Detection Queue  │      │
│                          └──────────────────┘      │
│                                    │               │
│  code/train_detection.py           │               │
│  (Training)                        │               │
│                                    ▼               │
│                          ┌──────────────────┐      │
│  code/object_detection_  │ Detection        │      │
│  models.py ──────────────│ Inference        │      │
│  (Models)                └──────────────────┘      │
│                                    │               │
│                                    ▼               │
│  saved_models/           ┌──────────────────┐      │
│  (Trained weights) ──────│ Results          │      │
│                          └──────────────────┘      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Performance Characteristics

```
┌─────────────────────────────────────────────────────┐
│ PERFORMANCE PROFILE                                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Queue Operations:      O(1)   - Add/Remove         │
│ Priority Queue:        O(log n) - Insert          │
│ Status Query:          O(1)   - Lookup            │
│ Statistics Update:     O(1)   - Append            │
│                                                     │
│ Detection Speed:                                   │
│ • SSD300:             ~0.02s  per image           │
│ • Faster R-CNN:       ~0.05s  per image           │
│ • EfficientDet:       ~0.03s  per image           │
│                                                     │
│ Memory:                                            │
│ • Queue overhead:     ~1KB per task               │
│ • Model memory:       ~100-500MB                  │
│ • Image processing:   ~5-10MB per image           │
│                                                     │
│ Throughput:                                        │
│ • Single thread:      ~20-50 img/s                │
│ • Batch processing:   ~50-200 img/s               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Error Handling Flow

```
Task Processing
      │
      ▼
┌──────────┐
│ Try      │
│ Detect   │
└─────┬────┘
      │
      ├─ Success ──────────────┐
      │                        ▼
      │              ┌──────────────────┐
      │              │ task.status =    │
      │              │ "completed"      │
      │              │ task.result = {} │
      │              └──────────────────┘
      │                        │
      │                        ▼
      │              ┌──────────────────┐
      │              │ on_task_complete │
      │              └──────────────────┘
      │
      └─ Exception ────────────┐
                               ▼
                     ┌──────────────────┐
                     │ task.status =    │
                     │ "failed"         │
                     │ task.error = str │
                     └──────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ on_task_error    │
                     └──────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ Log error        │
                     │ Continue queue   │
                     └──────────────────┘
```

---

This architecture ensures:
- **Scalability**: Handle 1000+ images
- **Reliability**: Thread-safe, error handling
- **Flexibility**: Multiple models, priorities
- **Monitoring**: Real-time statistics, callbacks
- **Performance**: Optimized batch processing
