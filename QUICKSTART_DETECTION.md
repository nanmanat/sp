# Quick Start Guide - Object Detection

This guide will help you get started with the object detection features in just a few minutes.

## Step 1: Installation

Make sure you have the basic requirements installed:

```bash
# Basic requirements (required)
pip install torch torchvision

# Optional: For EfficientDet models
pip install effdet timm

# Optional: For DETR models  
pip install transformers
```

## Step 2: Launch the GUI

```bash
python run_gui.py
```

## Step 3: Configure Object Detection

1. Click on the **"Object Detection"** tab
2. Select a model from the dropdown (e.g., `faster_rcnn_r50_fpn`)
3. Configure parameters:
   - **Number of Classes**: 91 (for COCO), 21 (for VOC), or your custom number
   - **Batch Size**: Start with 4 (reduce if you get memory errors)
   - **Learning Rate**: 0.001 is a good starting point
   - **Epochs**: 50 for initial experiments
   - **Dataset Path**: Point to your dataset

4. Choose an option:
   - **Add to Queue**: Queue the experiment to run with others
   - **Start Now**: Begin training immediately

## Step 4: Monitor Training

Switch to the **"Log"** tab to see real-time training progress.

## Quick Model Recommendations

### For Speed (Real-time Applications)
```
Model: ssd300 or efficientdet_d0
Batch Size: 8
```

### For Accuracy
```
Model: cascade_rcnn_r50_fpn or efficientdet_d3
Batch Size: 4
```

### For Balance
```
Model: faster_rcnn_r50_fpn or efficientdet_d1
Batch Size: 4
```

## Example: Training on Custom Dataset

Let's say you have a dataset with 5 object classes:

1. **Number of Classes**: Set to 6 (5 classes + 1 background)
2. **Dataset Path**: Point to your annotations folder
3. **Model**: Start with `faster_rcnn_r50_fpn`
4. **Batch Size**: 4
5. **Learning Rate**: 0.001
6. **Epochs**: 50

Click "Start Detection Training Now" and monitor in the Log tab!

## Troubleshooting

### Out of Memory Error
✅ **Solution**: Reduce batch size to 2 or even 1

### Model Import Error
✅ **Solution**: Install required packages
```bash
pip install effdet timm transformers
```

### Training is Slow
✅ **Solution**: 
- Use a smaller model (e.g., `ssd300` instead of `cascade_rcnn_r50_fpn`)
- Increase `num_workers` to 8 or 12

## Next Steps

- Read the full guide: [OBJECT_DETECTION_GUIDE.md](OBJECT_DETECTION_GUIDE.md)
- Check model examples: `python run_detection_example.py`
- Experiment with different models in the GUI

## Need Help?

Check these resources:
- **Full documentation**: See OBJECT_DETECTION_GUIDE.md
- **Model listing**: Run `python run_detection_example.py`
- **Code examples**: See `examples/detection_models_example.py`

Happy Training! 🚀
