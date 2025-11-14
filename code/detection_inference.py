"""
Object Detection Inference Module
Handles loading models and running inference for object detection
"""

import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.object_detection_models import create_detection_model


class DetectionInference:
    """
    Handles object detection inference on images
    """
    
    def __init__(self, 
                 model_name: str = 'faster_rcnn_r50_fpn',
                 num_classes: int = 91,
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 confidence_threshold: float = 0.5):
        """
        Initialize the detection inference engine
        
        Args:
            model_name: Name of the detection model
            num_classes: Number of classes (including background if applicable)
            model_path: Path to saved model weights (optional)
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        print(f"Initializing detection inference with {model_name}")
        print(f"Device: {self.device}")
        
        # Load model
        self.model = create_detection_model(model_name, num_classes=num_classes, pretrained=(model_path is None))
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        
        print("Inference engine ready!")
        
    def detect(self, 
              image_path: Union[str, Image.Image],
              conf_threshold: Optional[float] = None,
              return_timing: bool = False) -> Dict:
        """
        Run object detection on an image
        
        Args:
            image_path: Path to image or PIL Image object
            conf_threshold: Override confidence threshold for this detection
            return_timing: If True, include timing information
            
        Returns:
            Dictionary with detection results:
            {
                'boxes': List of [x1, y1, x2, y2] bounding boxes,
                'scores': List of confidence scores,
                'labels': List of class labels,
                'num_detections': Number of detections,
                'timing': Processing time (if return_timing=True)
            }
        """
        threshold = conf_threshold if conf_threshold is not None else self.confidence_threshold
        
        start_time = time.time()
        
        # Load image
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
            
        original_size = image.size
        
        # Transform image
        image_tensor = self.transform(image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model([image_tensor])
            
        # Extract results (handle different model output formats)
        if isinstance(prediction, list):
            prediction = prediction[0]
            
        # Filter by confidence threshold
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        # Apply confidence filtering
        keep_indices = scores >= threshold
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        processing_time = time.time() - start_time
        
        result = {
            'boxes': boxes.tolist(),
            'scores': scores.tolist(),
            'labels': labels.tolist(),
            'num_detections': len(boxes),
            'image_size': original_size
        }
        
        if return_timing:
            result['timing'] = processing_time
            
        return result
        
    def detect_batch(self, 
                    image_paths: List[Union[str, Image.Image]],
                    conf_threshold: Optional[float] = None) -> List[Dict]:
        """
        Run detection on a batch of images
        
        Args:
            image_paths: List of image paths or PIL Images
            conf_threshold: Override confidence threshold
            
        Returns:
            List of detection result dictionaries
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.detect(img_path, conf_threshold=conf_threshold)
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({'error': str(e)})
                
        return results
        
    def visualize_detections(self,
                           image_path: Union[str, Image.Image],
                           detections: Optional[Dict] = None,
                           class_names: Optional[List[str]] = None,
                           save_path: Optional[str] = None,
                           show: bool = True) -> Image.Image:
        """
        Visualize detection results on an image
        
        Args:
            image_path: Path to image or PIL Image
            detections: Detection results (if None, will run detection)
            class_names: List of class names for labels
            save_path: Path to save visualization
            show: If True, display the image
            
        Returns:
            PIL Image with visualizations
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
            
        # Get detections if not provided
        if detections is None:
            detections = self.detect(image)
            
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw bounding boxes
        boxes = detections['boxes']
        scores = detections['scores']
        labels = detections['labels']
        
        colors = plt.cm.hsv(np.linspace(0, 1, self.num_classes)).tolist()
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Get color for this class
            color = colors[int(label) % len(colors)]
            
            # Draw rectangle
            rect = patches.Rectangle((x1, y1), width, height,
                                    linewidth=2, edgecolor=color,
                                    facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            if class_names and int(label) < len(class_names):
                class_name = class_names[int(label)]
            else:
                class_name = f"Class {int(label)}"
                
            label_text = f"{class_name}: {score:.2f}"
            ax.text(x1, y1 - 5, label_text,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                   fontsize=10, color='white')
        
        ax.axis('off')
        plt.title(f"Detections: {len(boxes)} objects found")
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to: {save_path}")
            
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
            
        # Convert to PIL Image
        fig.canvas.draw()
        vis_image = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                    fig.canvas.tostring_rgb())
        plt.close(fig)
        
        return vis_image
        
    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        self.confidence_threshold = threshold
        
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }


class BatchDetectionInference(DetectionInference):
    """
    Optimized detection inference for processing batches of images efficiently
    """
    
    def __init__(self, batch_size: int = 8, **kwargs):
        """
        Initialize batch inference engine
        
        Args:
            batch_size: Number of images to process at once
            **kwargs: Arguments for DetectionInference
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        
    def detect_batch_optimized(self,
                               image_paths: List[Union[str, Image.Image]],
                               conf_threshold: Optional[float] = None) -> List[Dict]:
        """
        Run detection on a batch of images with batched inference
        
        Args:
            image_paths: List of image paths or PIL Images
            conf_threshold: Override confidence threshold
            
        Returns:
            List of detection result dictionaries
        """
        threshold = conf_threshold if conf_threshold is not None else self.confidence_threshold
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            
            # Load and transform batch
            batch_images = []
            valid_indices = []
            
            for idx, img_path in enumerate(batch_paths):
                try:
                    if isinstance(img_path, str):
                        image = Image.open(img_path).convert('RGB')
                    else:
                        image = img_path.convert('RGB')
                    
                    image_tensor = self.transform(image).to(self.device)
                    batch_images.append(image_tensor)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    results.append({'error': str(e)})
            
            if not batch_images:
                continue
                
            # Run inference on batch
            with torch.no_grad():
                predictions = self.model(batch_images)
                
            # Process predictions
            for pred_idx, pred in enumerate(predictions):
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                
                # Filter by confidence
                keep_indices = scores >= threshold
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                labels = labels[keep_indices]
                
                result = {
                    'boxes': boxes.tolist(),
                    'scores': scores.tolist(),
                    'labels': labels.tolist(),
                    'num_detections': len(boxes)
                }
                results.append(result)
                
        return results


def load_class_names(class_file: str) -> List[str]:
    """
    Load class names from a file
    
    Args:
        class_file: Path to file with one class name per line
        
    Returns:
        List of class names
    """
    if not os.path.exists(class_file):
        raise FileNotFoundError(f"Class file not found: {class_file}")
        
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
        
    return class_names


def get_coco_class_names() -> List[str]:
    """Get standard COCO dataset class names"""
    return [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
