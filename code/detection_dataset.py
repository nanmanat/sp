"""
Object Detection Dataset Loaders
Supports COCO, Pascal VOC, and YOLO formats
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
import torchvision.transforms.functional as F


class COCODetectionDataset(Dataset):
    """
    COCO format dataset loader for object detection
    
    Expected structure:
    data_path/
        annotations/
            instances_train.json (or instances_train2017.json)
            instances_val.json (or instances_val2017.json)
        images/ (or train2017/, val2017/)
            image1.jpg
            image2.jpg
            ...
    """
    
    def __init__(self, data_path, split='train', transform=None):
        """
        Args:
            data_path: Root path to dataset
            split: 'train' or 'val'
            transform: Optional transforms to apply
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        
        # Try different COCO annotation file naming conventions
        annotation_files = [
            self.data_path / 'annotations' / f'instances_{split}.json',
            self.data_path / 'annotations' / f'instances_{split}2017.json',
        ]
        
        self.annotation_file = None
        for ann_file in annotation_files:
            if ann_file.exists():
                self.annotation_file = ann_file
                break
        
        if self.annotation_file is None:
            raise FileNotFoundError(
                f"Could not find COCO annotation file. Tried:\n" + 
                "\n".join(str(f) for f in annotation_files)
            )
        
        # Load annotations
        with open(self.annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Try different image directory naming conventions
        image_dirs = [
            self.data_path / 'images',
            self.data_path / f'{split}2017',
            self.data_path / split,
        ]
        
        self.image_dir = None
        for img_dir in image_dirs:
            if img_dir.exists():
                self.image_dir = img_dir
                break
        
        if self.image_dir is None:
            raise FileNotFoundError(
                f"Could not find image directory. Tried:\n" + 
                "\n".join(str(d) for d in image_dirs)
            )
        
        # Create image id to annotations mapping
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)
        
        # Get list of images
        self.images = self.coco_data['images']
        
        print(f"Loaded COCO dataset: {len(self.images)} images, split={split}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        image_id = img_info['id']
        img_path = self.image_dir / img_info['file_name']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(image_id, [])
        
        # Extract boxes and labels
        boxes = []
        labels = []
        for ann in annotations:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }
        
        # Apply custom transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert PIL Image to tensor
            image = F.to_tensor(image)
        
        return image, target


class VOCDetectionDataset(Dataset):
    """
    Pascal VOC format dataset loader for object detection
    
    Expected structure:
    data_path/
        Annotations/
            image1.xml
            image2.xml
            ...
        JPEGImages/
            image1.jpg
            image2.jpg
            ...
        ImageSets/
            Main/
                train.txt
                val.txt
    """
    
    def __init__(self, data_path, split='train', transform=None, class_mapping=None):
        """
        Args:
            data_path: Root path to dataset
            split: 'train' or 'val'
            transform: Optional transforms to apply
            class_mapping: Dict mapping class names to integers (e.g., {'dog': 1, 'cat': 2})
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.class_mapping = class_mapping or {}
        
        # Load image list
        split_file = self.data_path / 'ImageSets' / 'Main' / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        self.annotations_dir = self.data_path / 'Annotations'
        self.images_dir = self.data_path / 'JPEGImages'
        
        print(f"Loaded VOC dataset: {len(self.image_ids)} images, split={split}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        img_path = self.images_dir / f'{image_id}.jpg'
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        ann_path = self.annotations_dir / f'{image_id}.xml'
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            # Map class name to integer
            if self.class_mapping:
                if class_name not in self.class_mapping:
                    continue  # Skip unknown classes
                label = self.class_mapping[class_name]
            else:
                # Auto-assign labels if no mapping provided
                label = hash(class_name) % 1000  # Simple hash
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        # Apply custom transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert PIL Image to tensor
            image = F.to_tensor(image)
        
        return image, target


class YOLODetectionDataset(Dataset):
    """
    YOLO format dataset loader for object detection
    
    Expected structure:
    data_path/
        images/
            train/
                image1.jpg
                image2.jpg
                ...
            val/
                image1.jpg
                ...
        labels/
            train/
                image1.txt
                image2.txt
                ...
            val/
                image1.txt
                ...
    """
    
    def __init__(self, data_path, split='train', transform=None):
        """
        Args:
            data_path: Root path to dataset
            split: 'train' or 'val'
            transform: Optional transforms to apply
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        
        self.images_dir = self.data_path / 'images' / split
        self.labels_dir = self.data_path / 'labels' / split
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                                  list(self.images_dir.glob('*.png')))
        
        print(f"Loaded YOLO dataset: {len(self.image_files)} images, split={split}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size
        
        # Load corresponding label file
        label_path = self.labels_dir / f'{img_path.stem}.txt'
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    # YOLO format: center_x, center_y, width, height (normalized)
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # Convert to absolute coordinates [x1, y1, x2, y2]
                    x1 = (cx - w / 2) * img_width
                    y1 = (cy - h / 2) * img_height
                    x2 = (cx + w / 2) * img_width
                    y2 = (cy + h / 2) * img_height
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)  # Add 1 for background class
        
        # Convert to tensors
        if len(boxes) == 0:
            # If no annotations, create empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        # Apply custom transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert PIL Image to tensor
            image = F.to_tensor(image)
        
        return image, target


def collate_fn(batch):
    """
    Custom collate function for object detection
    Since each image may have different numbers of objects
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets


def collate_fn_efficientdet(batch, target_size=512):
    """
    Custom collate function for EfficientDet models
    Resizes all images to the same size and stacks them
    
    Args:
        batch: List of (image, target) tuples where target has 'boxes', 'labels'
        target_size: Target image size (default 512)
    
    Returns:
        images: Stacked tensor [B, C, H, W]
        targets: Dict of lists {'bbox': [tensor, ...], 'cls': [tensor, ...]}
    """
    import torch.nn.functional as F_torch
    
    images = []
    bbox_list = []
    cls_list = []
    
    for batch_idx, (image, target) in enumerate(batch):
        # Get original image size
        _, orig_h, orig_w = image.shape
        
        # Resize image to target size
        if orig_h != target_size or orig_w != target_size:
            image = F_torch.interpolate(
                image.unsqueeze(0),  # Add batch dim
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dim
        
        images.append(image)
        
        # Get boxes and labels
        boxes = target['boxes'].clone()  # [N, 4] in format [x1, y1, x2, y2]
        labels = target['labels']  # [N]
        
        # Scale boxes to match resized image
        if len(boxes) > 0 and (orig_h != target_size or orig_w != target_size):
            scale_x = target_size / orig_w
            scale_y = target_size / orig_h
            boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
            boxes[:, [1, 3]] *= scale_y  # Scale y coordinates
        
        # DetBenchTrain expects dict of lists
        bbox_list.append(boxes)
        cls_list.append(labels.float())
    
    # Stack images into batch tensor
    images = torch.stack(images)  # [B, C, H, W]
    
    # Return as dict of lists
    targets = {
        'bbox': bbox_list,  # List of [N, 4] tensors
        'cls': cls_list      # List of [N] tensors
    }
    
    return images, targets


def create_detection_dataset(data_path, dataset_format='coco', split='train', transform=None, **kwargs):
    """
    Factory function to create detection dataset based on format
    
    Args:
        data_path: Path to dataset
        dataset_format: 'coco', 'voc', or 'yolo'
        split: 'train' or 'val'
        transform: Optional transforms
        **kwargs: Additional arguments for specific dataset types
    
    Returns:
        Dataset object
    """
    dataset_format = dataset_format.lower()
    
    if dataset_format == 'coco':
        return COCODetectionDataset(data_path, split=split, transform=transform)
    elif dataset_format in ['voc', 'pascal_voc', 'pascalvoc']:
        return VOCDetectionDataset(data_path, split=split, transform=transform, **kwargs)
    elif dataset_format == 'yolo':
        return YOLODetectionDataset(data_path, split=split, transform=transform)
    else:
        raise ValueError(f"Unknown dataset format: {dataset_format}. Supported: 'coco', 'voc', 'yolo'")
