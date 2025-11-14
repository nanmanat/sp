"""
Object Detection Models Module
Supports various object detection architectures
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    ssd300_vgg16,
    retinanet_resnet50_fpn_v2,
    fasterrcnn_resnet50_fpn_v2,
    fcos_resnet50_fpn,
)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Some models will not be available.")

try:
    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
    from effdet.efficientdet import HeadNet
    EFFDET_AVAILABLE = True
except ImportError:
    EFFDET_AVAILABLE = False
    print("Warning: effdet not available. EfficientDet models will not be available.")


def create_detection_model(model_name, num_classes=91, pretrained=True):
    """
    Create and configure the specified object detection model
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of classes (including background for some models)
        pretrained: Whether to use pretrained weights (note: most pretrained weights expect 91 classes)
        
    Returns:
        model: The configured detection model
    """
    
    # Warn if using pretrained with custom num_classes for torchvision models
    torchvision_models = ['ssd300', 'retinanet_r50_fpn', 'faster_rcnn_r50_fpn', 
                          'cascade_rcnn_r50_fpn', 'fcos_r50_fpn', 'atss_r50_fpn']
    if model_name in torchvision_models and pretrained and num_classes != 91:
        print(f"⚠ Warning: Pretrained weights are for 91 classes (COCO).")
        print(f"⚠ Training from scratch with {num_classes} classes instead.")
        pretrained = False
    
    if model_name == 'ssd300':
        # SSD300 with VGG16 backbone
        model = ssd300_vgg16(num_classes=num_classes)
        
    elif model_name.startswith('efficientdet_d'):
        # EfficientDet models (D0-D3)
        if not EFFDET_AVAILABLE:
            raise ValueError("effdet is required for EfficientDet models. Please install: pip install effdet")
        
        # Extract variant number (0-3)
        variant = int(model_name.split('_d')[-1])
        config_name = f'efficientdet_d{variant}'
        
        config = get_efficientdet_config(config_name)
        config.num_classes = num_classes
        config.image_size = [512, 512]  # Standard training size
        
        # Create model
        net = EfficientDet(config, pretrained_backbone=pretrained)
        model = DetBenchTrain(net, config)
        
    elif model_name == 'retinanet_r50_fpn':
        # RetinaNet with ResNet50-FPN backbone
        model = retinanet_resnet50_fpn_v2(num_classes=num_classes)
        
    elif model_name == 'faster_rcnn_r50_fpn':
        # Faster R-CNN with ResNet50-FPN backbone
        model = fasterrcnn_resnet50_fpn_v2(num_classes=num_classes)
        
    elif model_name == 'cascade_rcnn_r50_fpn':
        # Cascade R-CNN (using Faster R-CNN as base with modifications)
        # Note: PyTorch doesn't have native Cascade R-CNN, this is a simplified version
        model = fasterrcnn_resnet50_fpn_v2(num_classes=num_classes)
        # In production, you'd want to use mmdetection for true Cascade R-CNN
        print("Note: Using Faster R-CNN as base. For true Cascade R-CNN, use mmdetection library.")
        
    elif model_name == 'fcos_r50_fpn':
        # FCOS with ResNet50-FPN backbone
        model = fcos_resnet50_fpn(num_classes=num_classes)
        
    elif model_name == 'atss_r50_fpn':
        # ATSS (similar to FCOS but with adaptive training sample selection)
        # Using FCOS as base since PyTorch doesn't have native ATSS
        model = fcos_resnet50_fpn(num_classes=num_classes)
        print("Note: Using FCOS as base. For true ATSS, use mmdetection library.")
        
    elif model_name == 'centernet_hourglass104':
        # CenterNet with Hourglass-104 backbone
        # This requires external libraries like mmdetection
        raise NotImplementedError(
            "CenterNet Hourglass-104 requires mmdetection. "
            "Install with: pip install mmcv-full mmdet"
        )
        
    elif model_name == 'detr_r50':
        # DETR (DEtection TRansformer) with ResNet50 backbone
        try:
            from transformers import DetrForObjectDetection
            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        except ImportError:
            raise ValueError(
                "transformers is required for DETR models. "
                "Please install: pip install transformers"
            )
            
    elif model_name == 'deformable_detr_r50':
        # Deformable DETR with ResNet50 backbone
        try:
            from transformers import DeformableDetrForObjectDetection
            model = DeformableDetrForObjectDetection.from_pretrained(
                "SenseTime/deformable-detr",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        except ImportError:
            raise ValueError(
                "transformers is required for Deformable DETR models. "
                "Please install: pip install transformers"
            )
            
    elif model_name == 'rt_detr_r50':
        # RT-DETR (Real-Time DETR) with ResNet50 backbone
        # This is a newer model that may require specific implementation
        try:
            from transformers import RTDetrForObjectDetection
            model = RTDetrForObjectDetection.from_pretrained(
                "PekingU/rtdetr_r50vd",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        except (ImportError, OSError):
            # Fallback to DETR if RT-DETR not available
            print("RT-DETR not available, using standard DETR as fallback")
            from transformers import DetrForObjectDetection
            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def get_model_info(model_name):
    """
    Get information about a specific model
    
    Returns:
        dict: Model information including description and requirements
    """
    
    model_info = {
        'ssd300': {
            'description': 'Single Shot MultiBox Detector with VGG16 backbone',
            'framework': 'torchvision',
            'requirements': ['torchvision>=0.14.0'],
            'notes': 'Fast one-stage detector'
        },
        'efficientdet_d0': {
            'description': 'EfficientDet-D0 (smallest, fastest)',
            'framework': 'effdet',
            'requirements': ['effdet', 'timm'],
            'notes': 'Efficient architecture with compound scaling'
        },
        'efficientdet_d1': {
            'description': 'EfficientDet-D1',
            'framework': 'effdet',
            'requirements': ['effdet', 'timm'],
            'notes': 'Balance between speed and accuracy'
        },
        'efficientdet_d2': {
            'description': 'EfficientDet-D2',
            'framework': 'effdet',
            'requirements': ['effdet', 'timm'],
            'notes': 'Good accuracy with reasonable speed'
        },
        'efficientdet_d3': {
            'description': 'EfficientDet-D3',
            'framework': 'effdet',
            'requirements': ['effdet', 'timm'],
            'notes': 'Higher accuracy, slower inference'
        },
        'retinanet_r50_fpn': {
            'description': 'RetinaNet with ResNet50-FPN backbone',
            'framework': 'torchvision',
            'requirements': ['torchvision>=0.14.0'],
            'notes': 'One-stage detector with focal loss'
        },
        'faster_rcnn_r50_fpn': {
            'description': 'Faster R-CNN with ResNet50-FPN backbone',
            'framework': 'torchvision',
            'requirements': ['torchvision>=0.14.0'],
            'notes': 'Two-stage detector, good accuracy'
        },
        'cascade_rcnn_r50_fpn': {
            'description': 'Cascade R-CNN with ResNet50-FPN backbone',
            'framework': 'torchvision (simplified)',
            'requirements': ['torchvision>=0.14.0'],
            'notes': 'Improved Faster R-CNN with cascaded refinement'
        },
        'fcos_r50_fpn': {
            'description': 'FCOS with ResNet50-FPN backbone',
            'framework': 'torchvision',
            'requirements': ['torchvision>=0.14.0'],
            'notes': 'Anchor-free one-stage detector'
        },
        'atss_r50_fpn': {
            'description': 'ATSS with ResNet50-FPN backbone',
            'framework': 'torchvision (using FCOS base)',
            'requirements': ['torchvision>=0.14.0'],
            'notes': 'Adaptive training sample selection'
        },
        'centernet_hourglass104': {
            'description': 'CenterNet with Hourglass-104 backbone',
            'framework': 'mmdetection',
            'requirements': ['mmcv-full', 'mmdet'],
            'notes': 'Keypoint-based detector'
        },
        'detr_r50': {
            'description': 'DETR with ResNet50 backbone',
            'framework': 'transformers',
            'requirements': ['transformers', 'torch>=1.10'],
            'notes': 'Transformer-based detector'
        },
        'deformable_detr_r50': {
            'description': 'Deformable DETR with ResNet50 backbone',
            'framework': 'transformers',
            'requirements': ['transformers', 'torch>=1.10'],
            'notes': 'DETR with deformable attention'
        },
        'rt_detr_r50': {
            'description': 'RT-DETR with ResNet50 backbone',
            'framework': 'transformers',
            'requirements': ['transformers', 'torch>=1.10'],
            'notes': 'Real-time DETR variant'
        }
    }
    
    return model_info.get(model_name, {'description': 'Unknown model'})


def list_available_models():
    """List all available object detection models"""
    return [
        'ssd300',
        'efficientdet_d0',
        'efficientdet_d1',
        'efficientdet_d2',
        'efficientdet_d3',
        'retinanet_r50_fpn',
        'faster_rcnn_r50_fpn',
        'cascade_rcnn_r50_fpn',
        'fcos_r50_fpn',
        'atss_r50_fpn',
        'centernet_hourglass104',
        'detr_r50',
        'deformable_detr_r50',
        'rt_detr_r50'
    ]
