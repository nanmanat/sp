"""
Test script to verify custom number of classes works with detection models
"""

import torch
from code.object_detection_models import create_detection_model

def test_model_with_custom_classes(model_name, num_classes):
    """Test if a model can be created with custom number of classes"""
    print(f"\nTesting {model_name} with {num_classes} classes...")
    try:
        model = create_detection_model(model_name, num_classes=num_classes, pretrained=True)
        print(f"✓ Successfully created {model_name} with {num_classes} classes")
        
        # Test forward pass with dummy input
        model.eval()
        dummy_input = [torch.rand(3, 300, 300)]
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    # Test with 3 classes (2 object classes + background)
    num_classes = 3
    
    print("="*60)
    print(f"Testing Object Detection Models with {num_classes} classes")
    print("="*60)
    
    # Test torchvision models that needed fixes
    models_to_test = [
        'ssd300',
        'retinanet_r50_fpn',
        'faster_rcnn_r50_fpn',
        'fcos_r50_fpn',
        'atss_r50_fpn',
    ]
    
    results = {}
    for model_name in models_to_test:
        results[model_name] = test_model_with_custom_classes(model_name, num_classes)
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{model_name}: {status}")
