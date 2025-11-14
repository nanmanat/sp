"""
Example script demonstrating object detection models usage
"""

from code.object_detection_models import (
    list_available_models, 
    get_model_info,
    create_detection_model
)

def main():
    print("=" * 70)
    print("OBJECT DETECTION MODELS - AVAILABLE MODELS")
    print("=" * 70)
    
    # List all available models
    models = list_available_models()
    
    print("\nAvailable Object Detection Models:\n")
    
    for i, model_name in enumerate(models, 1):
        info = get_model_info(model_name)
        print(f"{i:2d}. {model_name}")
        print(f"    Description: {info['description']}")
        print(f"    Framework:   {info['framework']}")
        print(f"    Notes:       {info['notes']}")
        print()
    
    print("=" * 70)
    print("\nExample Usage:")
    print("=" * 70)
    
    print("""
# To create a model:
from code.object_detection_models import create_detection_model

# For COCO dataset (91 classes including background)
model = create_detection_model('faster_rcnn_r50_fpn', num_classes=91)

# For Pascal VOC (21 classes including background)
model = create_detection_model('ssd300', num_classes=21)

# For custom dataset (e.g., 10 classes + background)
model = create_detection_model('retinanet_r50_fpn', num_classes=11)

# To train a model using the GUI:
# 1. Run: python run_gui.py
# 2. Navigate to the "Object Detection" tab
# 3. Select your model and configure parameters
# 4. Click "Add Detection Training to Queue" or "Start Detection Training Now"
    """)
    
    print("\n" + "=" * 70)
    print("MODEL CATEGORIES")
    print("=" * 70)
    
    categories = {
        "One-Stage Detectors (Fast)": [
            'ssd300',
            'efficientdet_d0',
            'efficientdet_d1', 
            'efficientdet_d2',
            'efficientdet_d3',
            'retinanet_r50_fpn',
            'fcos_r50_fpn',
            'atss_r50_fpn'
        ],
        "Two-Stage Detectors (Accurate)": [
            'faster_rcnn_r50_fpn',
            'cascade_rcnn_r50_fpn'
        ],
        "Transformer-Based (Modern)": [
            'detr_r50',
            'deformable_detr_r50',
            'rt_detr_r50'
        ],
        "Anchor-Free (Simplified)": [
            'centernet_hourglass104',
            'fcos_r50_fpn'
        ]
    }
    
    for category, model_list in categories.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"  • {model}")
    
    print("\n" + "=" * 70)
    print("REQUIREMENTS")
    print("=" * 70)
    
    print("""
Basic Requirements (always needed):
  • torch >= 1.10.0
  • torchvision >= 0.14.0

Optional Requirements (for specific models):
  • effdet          - For EfficientDet models (D0-D3)
  • timm            - For EfficientDet and some backbones
  • transformers    - For DETR-based models
  • mmcv-full       - For advanced models (CenterNet, etc.)
  • mmdet           - For MMDetection models

Installation:
  pip install torch torchvision
  pip install effdet timm
  pip install transformers
  pip install mmcv-full mmdet  # Optional, for advanced models
    """)
    
    print("=" * 70)

if __name__ == "__main__":
    main()
