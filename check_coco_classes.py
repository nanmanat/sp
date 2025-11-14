"""
Check the number of classes in the converted COCO dataset
"""

import json

# Load the COCO annotation file
annotation_file = 'E:/Buoy/dataset_coco/annotations/instances_train2017.json'

print("="*60)
print("COCO Dataset Class Analysis")
print("="*60)

with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

print(f"\nDataset: {coco_data['info']['description']}")
print(f"Images: {len(coco_data['images'])}")
print(f"Annotations: {len(coco_data['annotations'])}")

print(f"\nCategories:")
for cat in coco_data['categories']:
    print(f"  ID {cat['id']}: {cat['name']}")

# Count annotations per category
category_counts = {}
for ann in coco_data['annotations']:
    cat_id = ann['category_id']
    category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

print(f"\nAnnotations per category:")
for cat_id, count in sorted(category_counts.items()):
    cat_name = next((c['name'] for c in coco_data['categories'] if c['id'] == cat_id), 'Unknown')
    print(f"  Category {cat_id} ({cat_name}): {count} annotations")

num_categories = len(coco_data['categories'])
max_category_id = max(cat['id'] for cat in coco_data['categories'])

print(f"\n{'='*60}")
print(f"Number of categories: {num_categories}")
print(f"Max category ID: {max_category_id}")
print(f"\nFor training, you should use:")
print(f"  num_classes = {max_category_id + 1}  (includes background at index 0)")
print(f"{'='*60}")
