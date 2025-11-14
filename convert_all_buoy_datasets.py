"""
Convert all YOLO format datasets in E:\dataset_sep_1_2_aug to COCO format
Converts all 3 datasets with 2 classes: drowning and swimming
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
from datetime import datetime


def yolo_to_coco_converter(
    yolo_dataset_path,
    output_path,
    class_names=None,
    dataset_name="Buoy Dataset"
):
    """
    Convert YOLO format dataset to COCO format
    
    Args:
        yolo_dataset_path: Path to YOLO format dataset
        output_path: Path where COCO format dataset will be created
        class_names: List of class names (e.g., ['drowning', 'swimming'])
                    If None, will use class_0, class_1, etc.
        dataset_name: Name of the dataset
    """
    
    yolo_path = Path(yolo_dataset_path)
    output_path = Path(output_path)
    
    print("="*70)
    print(f"YOLO TO COCO CONVERSION")
    print("="*70)
    print(f"Source: {yolo_path}")
    print(f"Output: {output_path}")
    print("="*70)
    
    # Check source exists
    if not yolo_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {yolo_path}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'annotations').mkdir(exist_ok=True)
    
    # Process train, val/valid, and test splits
    # Check which naming convention is used (val or valid)
    splits_to_check = ['train', 'val', 'valid', 'test']
    available_splits = []
    
    for split_name in splits_to_check:
        split_dir = yolo_path / split_name
        if split_dir.exists() and (split_dir / 'images').exists():
            available_splits.append(split_name)
    
    if not available_splits:
        print("  Warning: No valid splits found with images directory")
        return
    
    print(f"  Found splits: {', '.join(available_splits)}")
    
    for split in available_splits:
        # Map 'valid' to 'val' for COCO naming
        coco_split = 'val' if split == 'valid' else split
        
        print(f"\nProcessing {split} split...")
        
        # Paths - images and labels are at root/split/images and root/split/labels
        images_dir = yolo_path / split / 'images'
        labels_dir = yolo_path / split / 'labels'
        output_images_dir = output_path / f'{coco_split}2017'  # COCO naming convention
        
        if not images_dir.exists():
            print(f"  Warning: {split} images directory not found: {images_dir}")
            continue
        
        if not labels_dir.exists():
            print(f"  Warning: {split} labels directory not found: {labels_dir}")
            continue
        
        # Create output images directory
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize COCO format structure
        coco_format = {
            "info": {
                "description": dataset_name,
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "",
                "date_created": datetime.now().strftime("%Y/%m/%d")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Get all image files
        image_files = sorted(list(images_dir.glob('*.jpg')) + 
                           list(images_dir.glob('*.png')) +
                           list(images_dir.glob('*.jpeg')))
        
        if len(image_files) == 0:
            print(f"  Warning: No images found in {images_dir}")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Determine number of classes from labels
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
        
        # Create categories
        if class_names is None:
            class_names = [f'class_{i}' for i in range(num_classes)]
        elif len(class_names) < num_classes:
            # Extend with generic names if not enough provided
            class_names = list(class_names) + [f'class_{i}' for i in range(len(class_names), num_classes)]
        
        for class_id in range(num_classes):
            coco_format['categories'].append({
                "id": class_id + 1,  # COCO uses 1-based indexing
                "name": class_names[class_id],
                "supercategory": "object"
            })
        
        print(f"  Classes detected: {num_classes}")
        for i, name in enumerate(class_names[:num_classes]):
            print(f"    {i}: {name}")
        
        # Process each image
        annotation_id = 1
        image_id = 1
        
        for img_file in image_files:
            # Copy image to output directory
            output_img_path = output_images_dir / img_file.name
            shutil.copy2(img_file, output_img_path)
            
            # Get image dimensions
            try:
                with Image.open(img_file) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"  Warning: Could not open {img_file.name}: {e}")
                continue
            
            # Add image info
            coco_format['images'].append({
                "id": image_id,
                "file_name": img_file.name,
                "width": img_width,
                "height": img_height,
                "license": 1,
                "date_captured": ""
            })
            
            # Read YOLO annotations
            label_file = labels_dir / f'{img_file.stem}.txt'
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO to COCO format
                        # YOLO: normalized [cx, cy, w, h]
                        # COCO: absolute [x, y, w, h]
                        x = (center_x - width / 2) * img_width
                        y = (center_y - height / 2) * img_height
                        w = width * img_width
                        h = height * img_height
                        
                        # Add annotation
                        coco_format['annotations'].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + 1,  # COCO uses 1-based indexing
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                            "segmentation": []
                        })
                        
                        annotation_id += 1
            
            image_id += 1
        
        # Save COCO format JSON
        output_json = output_path / 'annotations' / f'instances_{coco_split}2017.json'
        with open(output_json, 'w') as f:
            json.dump(coco_format, f, indent=2)
        
        print(f"  ✓ Converted {len(coco_format['images'])} images")
        print(f"  ✓ Created {len(coco_format['annotations'])} annotations")
        print(f"  ✓ Saved to: {output_json}")
        print(f"  ✓ Images copied to: {output_images_dir}")
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print(f"\nYour COCO format dataset is at: {output_path}")


if __name__ == "__main__":
    # Configuration
    BASE_PATH = "E:\\dataset_sep_1_2_aug"
    CLASS_NAMES = ['drowning', 'swimming']
    
    # Define all 3 datasets to convert
    datasets = [
        {
            'name': 'dataset_sep_1_2_aug',
            'source': f"{BASE_PATH}\\dataset_sep_1_2_aug",
            'output': f"{BASE_PATH}\\dataset_sep_1_2_aug_coco",
            'description': "Combined Dataset (Sep 1 + 2 Augmented)"
        },
        {
            'name': 'dataset_sep_1_aug',
            'source': f"{BASE_PATH}\\dataset_sep_1_aug",
            'output': f"{BASE_PATH}\\dataset_sep_1_aug_coco",
            'description': "Dataset Sep 1 Augmented"
        },
        {
            'name': 'dataset_sep_2_aug',
            'source': f"{BASE_PATH}\\dataset_sep_2_aug",
            'output': f"{BASE_PATH}\\dataset_sep_2_aug_coco",
            'description': "Dataset Sep 2 Augmented"
        }
    ]
    
    print("\n" + "="*70)
    print("BATCH CONVERSION: YOLO TO COCO FORMAT")
    print("="*70)
    print(f"\nWill convert {len(datasets)} datasets:")
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['description']}")
        print(f"   Source: {ds['source']}")
        print(f"   Output: {ds['output']}")
    
    print(f"\nClasses: {CLASS_NAMES}")
    
    # Ask for confirmation
    response = input("\nProceed with conversion of all 3 datasets? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        success_count = 0
        failed_datasets = []
        
        for i, ds in enumerate(datasets, 1):
            print(f"\n\n{'='*70}")
            print(f"CONVERTING DATASET {i}/{len(datasets)}")
            print(f"{'='*70}\n")
            
            try:
                yolo_to_coco_converter(
                    yolo_dataset_path=ds['source'],
                    output_path=ds['output'],
                    class_names=CLASS_NAMES,
                    dataset_name=ds['description']
                )
                success_count += 1
                print(f"\n✓ SUCCESS: {ds['name']} converted!")
                
            except Exception as e:
                print(f"\n✗ ERROR converting {ds['name']}: {e}")
                failed_datasets.append(ds['name'])
                import traceback
                traceback.print_exc()
        
        # Final summary
        print("\n\n" + "="*70)
        print("BATCH CONVERSION SUMMARY")
        print("="*70)
        print(f"\nTotal datasets: {len(datasets)}")
        print(f"Successfully converted: {success_count}")
        print(f"Failed: {len(failed_datasets)}")
        
        if failed_datasets:
            print(f"\nFailed datasets:")
            for name in failed_datasets:
                print(f"  - {name}")
        
        if success_count == len(datasets):
            print("\n✓ ALL DATASETS CONVERTED SUCCESSFULLY!")
        elif success_count > 0:
            print(f"\n⚠ PARTIAL SUCCESS: {success_count}/{len(datasets)} datasets converted")
        else:
            print("\n✗ ALL CONVERSIONS FAILED")
            
        print("\n" + "="*70)
        
    else:
        print("\nConversion cancelled.")
