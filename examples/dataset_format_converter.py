"""
Dataset Format Conversion Utilities
Helpers for converting between COCO, Pascal VOC, and YOLO formats
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def coco_to_yolo(coco_box, img_width, img_height):
    """
    Convert COCO format [x, y, width, height] to YOLO format
    [center_x, center_y, width, height] (normalized 0-1)
    
    Args:
        coco_box: List/tuple of [x, y, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (center_x, center_y, width, height) normalized to 0-1
    """
    x, y, w, h = coco_box
    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    return center_x, center_y, norm_w, norm_h


def yolo_to_coco(yolo_box, img_width, img_height):
    """
    Convert YOLO format [center_x, center_y, width, height] (normalized)
    to COCO format [x, y, width, height] in pixels
    
    Args:
        yolo_box: List/tuple of normalized [center_x, center_y, width, height]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (x, y, width, height) in pixels
    """
    center_x, center_y, w, h = yolo_box
    x = (center_x - w / 2) * img_width
    y = (center_y - h / 2) * img_height
    width = w * img_width
    height = h * img_height
    return x, y, width, height


def voc_to_coco(voc_box):
    """
    Convert Pascal VOC format [xmin, ymin, xmax, ymax] to COCO format
    [x, y, width, height]
    
    Args:
        voc_box: List/tuple of [xmin, ymin, xmax, ymax]
        
    Returns:
        Tuple of (x, y, width, height)
    """
    xmin, ymin, xmax, ymax = voc_box
    x = xmin
    y = ymin
    width = xmax - xmin
    height = ymax - ymin
    return x, y, width, height


def coco_to_voc(coco_box):
    """
    Convert COCO format [x, y, width, height] to Pascal VOC format
    [xmin, ymin, xmax, ymax]
    
    Args:
        coco_box: List/tuple of [x, y, width, height]
        
    Returns:
        Tuple of (xmin, ymin, xmax, ymax)
    """
    x, y, w, h = coco_box
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return xmin, ymin, xmax, ymax


def yolo_to_voc(yolo_box, img_width, img_height):
    """
    Convert YOLO format to Pascal VOC format
    
    Args:
        yolo_box: List/tuple of normalized [center_x, center_y, width, height]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (xmin, ymin, xmax, ymax) in pixels
    """
    center_x, center_y, w, h = yolo_box
    xmin = (center_x - w / 2) * img_width
    ymin = (center_y - h / 2) * img_height
    xmax = (center_x + w / 2) * img_width
    ymax = (center_y + h / 2) * img_height
    return xmin, ymin, xmax, ymax


def voc_to_yolo(voc_box, img_width, img_height):
    """
    Convert Pascal VOC format to YOLO format
    
    Args:
        voc_box: List/tuple of [xmin, ymin, xmax, ymax]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of normalized (center_x, center_y, width, height)
    """
    xmin, ymin, xmax, ymax = voc_box
    center_x = ((xmin + xmax) / 2) / img_width
    center_y = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return center_x, center_y, width, height


def read_yolo_annotation(txt_path):
    """
    Read YOLO format annotation file
    
    Args:
        txt_path: Path to .txt annotation file
        
    Returns:
        List of annotations, each as [class_id, center_x, center_y, width, height]
    """
    annotations = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append([class_id, center_x, center_y, width, height])
    return annotations


def write_yolo_annotation(txt_path, annotations):
    """
    Write YOLO format annotation file
    
    Args:
        txt_path: Path to output .txt file
        annotations: List of annotations [class_id, center_x, center_y, width, height]
    """
    with open(txt_path, 'w') as f:
        for ann in annotations:
            class_id, cx, cy, w, h = ann
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def read_voc_annotation(xml_path):
    """
    Read Pascal VOC format annotation file
    
    Args:
        xml_path: Path to .xml annotation file
        
    Returns:
        Dict with image info and list of object annotations
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image info
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Get objects
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append({
            'class': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return {
        'width': width,
        'height': height,
        'objects': objects
    }


# Example usage
if __name__ == "__main__":
    print("Dataset Format Conversion Utilities")
    print("=" * 50)
    
    # Example: COCO to YOLO conversion
    print("\nExample 1: COCO to YOLO")
    coco_box = [100, 150, 200, 300]  # [x, y, width, height]
    img_width, img_height = 1920, 1080
    yolo_box = coco_to_yolo(coco_box, img_width, img_height)
    print(f"COCO: {coco_box}")
    print(f"YOLO: {yolo_box}")
    
    # Example: YOLO to COCO conversion
    print("\nExample 2: YOLO to COCO")
    yolo_box = [0.5, 0.5, 0.3, 0.4]  # normalized [cx, cy, w, h]
    coco_box = yolo_to_coco(yolo_box, img_width, img_height)
    print(f"YOLO: {yolo_box}")
    print(f"COCO: {coco_box}")
    
    # Example: VOC to YOLO conversion
    print("\nExample 3: VOC to YOLO")
    voc_box = [100, 200, 300, 400]  # [xmin, ymin, xmax, ymax]
    yolo_box = voc_to_yolo(voc_box, img_width, img_height)
    print(f"VOC: {voc_box}")
    print(f"YOLO: {yolo_box}")
    
    print("\n" + "=" * 50)
    print("Import this module to use conversion functions:")
    print("  from examples.dataset_format_converter import coco_to_yolo")
    print("  from examples.dataset_format_converter import yolo_to_coco")
    print("  from examples.dataset_format_converter import voc_to_yolo")
