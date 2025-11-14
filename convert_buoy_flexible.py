"""
Flexible YOLO to COCO converter that detects dataset structure
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
from datetime import datetime


def detect_yolo_structure(dataset_path):
    """
    Detect the structure of a YOLO dataset
    Returns possible image and label directories
    """
    dataset_path = Path(dataset_path)
    
    print(f"Detecting dataset structure in: {dataset_path}")
    print(f"Contents of {dataset_path}:")
    
    structures = []
    
    # Check standard YOLO structure: images/train/, labels/train/
    if (dataset_path / 'images' / 'train').exists():
        structures.append({
            'type': 'standard',
            'train_images': dataset_path / 'images' / 'train',
            'train_labels': dataset_path / 'labels' / 'train',
            'val_images': dataset_path / 'images' / 'val' if (dataset_path / 'images' / 'val').exists() else None,
            'val_labels': dataset_path / 'labels' / 'val' if (dataset_path / 'labels' / 'val').exists() else None
        })
    
    # Check alternative: train/, val/ directly
    if (dataset_path / 'train' / 'images').exists():
        structures.append({
            'type': 'alternative',
            'train_images': dataset_path / 'train' / 'images',
            'train_labels': dataset_path / 'train' / 'labels',
            'val_images': dataset_path / 'val' / 'images' if (dataset_path / 'val' / 'images').exists() else None,
            'val_labels': dataset_path / 'val' / 'labels' if (dataset_path / 'val' / 'labels').exists() else None
        })
    
    # Check if images and labels are directly in dataset_path
    image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
    label_files = list(dataset_path.glob('*.txt'))
    
    if image_files and label_files:
        structures.append({
            'type': 'flat',
            'train_images': dataset_path,
            'train_labels': dataset_path,
            'val_images': None,
            'val_labels': None
        })
    
    # Check for images/ and labels/ directly (no train/val split)
    if (dataset_path / 'images').exists() and (dataset_path / 'labels').exists():
        img_dir = dataset_path / 'images'
        lbl_dir = dataset_path / 'labels'
        
        # Check if they contain image/label files directly
        imgs = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        lbls = list(lbl_dir.glob('*.txt'))
        
        if imgs and lbls:
            structures.append({
                'type': 'no_split',
                'train_images': img_dir,
                'train_labels': lbl_dir,
                'val_images': None,
                'val_labels': None
            })
    
    return structures


def yolo_to_coco_flexible(
    yolo_dataset_path,
    output_path,
    class_names=None,
    dataset_name="Custom Dataset",
    train_val_split=0.8
):
    """
    Flexible YOLO to COCO converter that auto-detects structure
    
    Args:
        yolo_dataset_path: Path to YOLO dataset
        output_path: Output path for COCO format
        class_names: List of class names
        dataset_name: Name of dataset
        train_val_split: If no val set exists, split ratio (0.8 = 80% train, 20% val)
    """
    
    yolo_path = Path(yolo_dataset_path)
    output_path = Path(output_path)
    
    print("="*70)
    print("FLEXIBLE YOLO TO COCO CONVERTER")
    print("="*70)
    
    # Detect structure
    structures = detect_yolo_structure(yolo_path)
    
    if not structures:
        print(f"\n✗ Could not detect valid YOLO dataset structure in {yolo_path}")
        print("\nPlease ensure your dataset has one of these structures:")
        print("  1. images/train/, images/val/, labels/train/, labels/val/")
        print("  2. train/images/, train/labels/, val/images/, val/labels/")
        print("  3. images/, labels/ (will auto-split into train/val)")
        print("  4. *.jpg and *.txt directly in dataset folder")
        return
    
    # Use first detected structure
    structure = structures[0]
    print(f"\n✓ Detected structure type: {structure['type']}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'annotations').mkdir(exist_ok=True)
    
    # Process dataset
    for split_name, images_dir, labels_dir in [
        ('train', structure['train_images'], structure['train_labels']),
        ('val', structure['val_images'], structure['val_labels'])
    ]:
        if images_dir is None:
            if split_name == 'val' and structure['type'] in ['flat', 'no_split']:
                # Create val split from train data
                print(f"\n{split_name.upper()}: Creating from train split ({int((1-train_val_split)*100)}%)")
                # Will handle this below
                continue
            else:
                print(f"\n{split_name.upper()}: Skipping (not found)")
                continue
        
        print(f"\n{split_name.upper()}: Processing...")
        print(f"  Images: {images_dir}")
        print(f"  Labels: {labels_dir}")
        
        # Get all images
        image_files = sorted(list(images_dir.glob('*.jpg')) + 
                           list(images_dir.glob('*.png')) +
                           list(images_dir.glob('*.jpeg')))
        
        if not image_files:
            print(f"  Warning: No images found")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # If this is train and no val, split it
        if split_name == 'train' and structure['val_images'] is None:
            num_train = int(len(image_files) * train_val_split)
            train_files = image_files[:num_train]
            val_files = image_files[num_train:]
            
            # Process both splits
            for actual_split, files_list in [('train', train_files), ('val', val_files)]:
                convert_split_to_coco(
                    files_list, labels_dir, output_path, actual_split,
                    class_names, dataset_name
                )
        else:
            # Process single split
            convert_split_to_coco(
                image_files, labels_dir, output_path, split_name,
                class_names, dataset_name
            )
    
    print("\n" + "="*70)
    print("✓ CONVERSION COMPLETE!")
    print("="*70)
    print(f"\nCOCO dataset created at: {output_path}")
    print("\nYou can now train with:")
    print(f"  data_path='{output_path}'")
    print(f"  dataset_format='coco'")


def convert_split_to_coco(image_files, labels_dir, output_path, split_name, class_names, dataset_name):
    """Convert a split to COCO format"""
    
    output_images_dir = output_path / f'{split_name}2017'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO structure
    coco_format = {
        "info": {
            "description": dataset_name,
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Detect classes
    all_class_ids = set()
    for img_file in image_files:
        label_file = labels_dir / f'{img_file.stem}.txt'
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        all_class_ids.add(int(parts[0]))
    
    num_classes = max(all_class_ids) + 1 if all_class_ids else 1
    
    # Create class names if not provided
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    # Create categories
    for class_id in range(num_classes):
        coco_format['categories'].append({
            "id": class_id + 1,
            "name": class_names[class_id] if class_id < len(class_names) else f'class_{class_id}',
            "supercategory": "object"
        })
    
    print(f"  Classes: {num_classes} detected")
    
    # Process images
    annotation_id = 1
    for image_id, img_file in enumerate(image_files, 1):
        # Copy image
        output_img_path = output_images_dir / img_file.name
        shutil.copy2(img_file, output_img_path)
        
        # Get dimensions
        try:
            with Image.open(img_file) as img:
                img_width, img_height = img.size
        except:
            print(f"  Warning: Could not read {img_file.name}")
            continue
        
        # Add image
        coco_format['images'].append({
            "id": image_id,
            "file_name": img_file.name,
            "width": img_width,
            "height": img_height,
            "license": 1
        })
        
        # Read annotations
        label_file = labels_dir / f'{img_file.stem}.txt'
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # Convert YOLO to COCO
                    x = (cx - w / 2) * img_width
                    y = (cy - h / 2) * img_height
                    width = w * img_width
                    height = h * img_height
                    
                    coco_format['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,
                        "bbox": [x, y, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
    
    # Save JSON
    output_json = output_path / 'annotations' / f'instances_{split_name}2017.json'
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"  ✓ {len(coco_format['images'])} images")
    print(f"  ✓ {len(coco_format['annotations'])} annotations")
    print(f"  ✓ Saved: {output_json}")


if __name__ == "__main__":
    # Configuration
    YOLO_DATASET_PATH = "E:/Buoy/dataset"
    OUTPUT_PATH = "E:/Buoy/dataset_coco"
    
    # Your class names (edit this!)
    CLASS_NAMES = ['buoy', 'class_1']  # Edit with your actual class names
    # Or set to None for auto-generated names
    # CLASS_NAMES = None
    
    print("\n" + "="*70)
    print("YOLO TO COCO DATASET CONVERTER")
    print("="*70)
    print(f"\nSource: {YOLO_DATASET_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    
    response = input("\nStart conversion? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        try:
            yolo_to_coco_flexible(
                yolo_dataset_path=YOLO_DATASET_PATH,
                output_path=OUTPUT_PATH,
                class_names=CLASS_NAMES,
                dataset_name="Buoy Detection Dataset"
            )
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Cancelled.")
