# GUI Structure - Object Detection Feature

## Updated GUI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Acne Image Grading Experiment Runner                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────┬──────────────────┬───────┬─────┐                          │
│  │ Clas │ Object Detection │ Queue │ Log │                          │
│  │sific │                  │       │     │                          │
│  │ation │                  │       │     │                          │
│  └──────┴──────────────────┴───────┴─────┘                          │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║ Object Detection Configuration                                ║  │
│  ╠═══════════════════════════════════════════════════════════════╣  │
│  ║                                                                ║  │
│  ║  Detection Model:  [faster_rcnn_r50_fpn ▼] [Model Info]       ║  │
│  ║                                                                ║  │
│  ║  Number of Classes:  [91          ]  (e.g., 91 for COCO)      ║  │
│  ║                                                                ║  │
│  ║  Batch Size:         [4           ]  (smaller for detection)  ║  │
│  ║                                                                ║  │
│  ║  Learning Rate:      [0.001       ]                            ║  │
│  ║                                                                ║  │
│  ║  Epochs:             [50          ]                            ║  │
│  ║                                                                ║  │
│  ║  Num Workers:        [4           ]                            ║  │
│  ║                                                                ║  │
│  ║  Dataset Path:       [./datasets/coco      ] [Browse...]       ║  │
│  ║                                                                ║  │
│  ║  ┌─────────────────────────────────────────────────────────┐  ║  │
│  ║  │ Information                                             │  ║  │
│  ║  ├─────────────────────────────────────────────────────────┤  ║  │
│  ║  │ Object Detection Models:                                │  ║  │
│  ║  │                                                          │  ║  │
│  ║  │ • One-Stage Detectors: SSD, EfficientDet, RetinaNet     │  ║  │
│  ║  │   - Faster inference, good for real-time applications   │  ║  │
│  ║  │                                                          │  ║  │
│  ║  │ • Two-Stage Detectors: Faster R-CNN, Cascade R-CNN      │  ║  │
│  ║  │   - Higher accuracy, slower inference                   │  ║  │
│  ║  │                                                          │  ║  │
│  ║  │ • Transformer-Based: DETR, Deformable DETR, RT-DETR     │  ║  │
│  ║  │   - Modern architecture, set-based predictions          │  ║  │
│  ║  │                                                          │  ║  │
│  ║  │ • Anchor-Free: CenterNet, FCOS                          │  ║  │
│  ║  │   - Simplified architecture without anchor boxes        │  ║  │
│  ║  │                                                          │  ║  │
│  ║  │ Note: Some models require additional packages           │  ║  │
│  ║  └─────────────────────────────────────────────────────────┘  ║  │
│  ║                                                                ║  │
│  ║           [Add Detection Training to Queue]                    ║  │
│  ║                                                                ║  │
│  ║           [Start Detection Training Now]                       ║  │
│  ║                                                                ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Status: Ready                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Model Selection Dropdown

```
┌──────────────────────────────────┐
│ Detection Model:                 │
├──────────────────────────────────┤
│ ▼ faster_rcnn_r50_fpn            │
│ ┌────────────────────────────┐   │
│ │ ssd300                     │   │ ← One-Stage (Fast)
│ │ efficientdet_d0            │   │
│ │ efficientdet_d1            │   │
│ │ efficientdet_d2            │   │
│ │ efficientdet_d3            │   │
│ │ retinanet_r50_fpn          │   │
│ ├────────────────────────────┤   │
│ │ faster_rcnn_r50_fpn        │   │ ← Two-Stage (Accurate)
│ │ cascade_rcnn_r50_fpn       │   │
│ ├────────────────────────────┤   │
│ │ fcos_r50_fpn               │   │ ← Anchor-Free
│ │ atss_r50_fpn               │   │
│ │ centernet_hourglass104     │   │
│ ├────────────────────────────┤   │
│ │ detr_r50                   │   │ ← Transformer-Based
│ │ deformable_detr_r50        │   │
│ │ rt_detr_r50                │   │
│ └────────────────────────────┘   │
└──────────────────────────────────┘
```

## Model Info Dialog

```
┌─────────────────────────────────────────────┐
│ Model Information                      [×]  │
├─────────────────────────────────────────────┤
│                                             │
│ Model: faster_rcnn_r50_fpn                  │
│                                             │
│ Description: Faster R-CNN with ResNet50-FPN │
│ backbone                                    │
│                                             │
│ Framework: torchvision                      │
│                                             │
│ Requirements: torchvision>=0.14.0           │
│                                             │
│ Notes: Two-stage detector, good accuracy    │
│                                             │
│                    [OK]                     │
└─────────────────────────────────────────────┘
```

## Queue Tab with Mixed Experiments

```
┌─────────────────────────────────────────────────────────────────┐
│ Experiment Queue                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Model: resnet50, Folds: 0,1,2                          │ │ ← Classification
│  │ 2. [DETECTION] Model: faster_rcnn_r50_fpn, Classes: 91    │ │ ← Detection
│  │ 3. Model: efficientnet_v2_s, Folds: 0                    │ │ ← Classification
│  │ 4. [DETECTION] Model: ssd300, Classes: 20                 │ │ ← Detection
│  │                                                            │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  [Start Queue]  [Stop After Current]  [Clear Queue]             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Tab Structure Comparison

### Before (3 Tabs)
```
┌──────────────┬───────┬─────┐
│Configuration │ Queue │ Log │
└──────────────┴───────┴─────┘
```

### After (4 Tabs)
```
┌──────────────┬──────────────────┬───────┬─────┐
│Classification│ Object Detection │ Queue │ Log │
└──────────────┴──────────────────┴───────┴─────┘
```

## Workflow Diagrams

### Classification Workflow (Existing)
```
┌─────────────────┐
│ Classification  │
│      Tab        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Configure:      │
│ - Model         │
│ - Folds         │
│ - Batch Size    │
│ - Learning Rate │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────┐
│ Add to Queue    │─────▶│  Queue   │
└─────────────────┘      │   Tab    │
                         └────┬─────┘
                              │
                              ▼
                         ┌──────────┐
                         │   Log    │
                         │   Tab    │
                         └──────────┘
```

### Detection Workflow (New)
```
┌─────────────────┐
│ Object Detection│
│      Tab        │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Configure:          │
│ - Detection Model   │◄─── 14 Models!
│ - Num Classes       │
│ - Batch Size        │
│ - Learning Rate     │
│ - Epochs            │
└────────┬────────────┘
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
┌──────────────────┐   ┌──────────────┐
│ Add to Queue     │   │ Start Now    │
└────────┬─────────┘   └──────┬───────┘
         │                    │
         ▼                    │
    ┌──────────┐              │
    │  Queue   │              │
    │   Tab    │◄─────────────┘
    └────┬─────┘
         │
         ▼
    ┌──────────┐
    │   Log    │
    │   Tab    │
    └──────────┘
```

## Key UI Features

### 1. Model Selection
- ✅ Dropdown with 14 models
- ✅ Organized by category
- ✅ "Model Info" button for details

### 2. Configuration
- ✅ Detection-specific parameters
- ✅ Helpful tooltips and hints
- ✅ File browser for dataset path

### 3. Information Panel
- ✅ Model categories explained
- ✅ Performance characteristics
- ✅ Dependency notes

### 4. Training Options
- ✅ Queue for batch processing
- ✅ Immediate training option
- ✅ Real-time logging

### 5. Queue Integration
- ✅ Mixed experiment types
- ✅ Clear labeling ([DETECTION])
- ✅ Sequential execution

## User Interactions

### Scenario 1: Quick Training
```
User Action                     System Response
───────────────────────────────────────────────────
1. Click "Object Detection"  → Opens detection tab
2. Select model from dropdown → Updates configuration
3. Click "Model Info"         → Shows model details
4. Adjust parameters          → Validates inputs
5. Click "Start Now"          → Begins training
6. Switch to "Log" tab        → Shows real-time output
```

### Scenario 2: Batch Experiments
```
User Action                     System Response
───────────────────────────────────────────────────
1. Configure Detection #1     → Ready to add
2. Click "Add to Queue"       → Added to queue list
3. Configure Detection #2     → Ready to add
4. Click "Add to Queue"       → Added to queue list
5. Switch to "Queue" tab      → Shows 2 experiments
6. Click "Start Queue"        → Begins sequential training
7. Monitor in "Log" tab       → Shows progress
```

## Accessibility

- Clear labels and tooltips
- Logical tab order
- Consistent button placement
- Informative error messages
- Progress indication

## Responsive Design

- Adapts to window resizing
- Scrollable content areas
- Fixed-size controls
- Flexible information panels

---

**Design Philosophy**: Keep it simple, intuitive, and powerful.
**Target Users**: Researchers, students, practitioners.
**Usability**: Beginner-friendly with advanced options.
