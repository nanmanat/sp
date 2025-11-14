"""
Compare Detection Models Trained on Different Datasets

This script helps you compare the performance of the same model
trained on different datasets, or different models on the same dataset.
"""

import os
from pathlib import Path


def list_trained_models(save_dir='./saved_models'):
    """List all trained detection models"""
    
    save_path = Path(save_dir)
    
    if not save_path.exists():
        print(f"Directory not found: {save_dir}")
        return []
    
    models = []
    for model_dir in save_path.iterdir():
        if model_dir.is_dir():
            checkpoint = model_dir / 'best_model.pth'
            if checkpoint.exists():
                # Parse model name and dataset
                name_parts = model_dir.name.split('_', 1)
                if len(name_parts) == 2:
                    model_type = name_parts[0]
                    dataset = name_parts[1]
                else:
                    model_type = model_dir.name
                    dataset = 'unknown'
                
                models.append({
                    'full_name': model_dir.name,
                    'model_type': model_type,
                    'dataset': dataset,
                    'path': str(checkpoint),
                    'size_mb': checkpoint.stat().st_size / (1024 * 1024)
                })
    
    return models


def group_by_model_type(models):
    """Group models by their type"""
    grouped = {}
    for model in models:
        model_type = model['model_type']
        if model_type not in grouped:
            grouped[model_type] = []
        grouped[model_type].append(model)
    return grouped


def group_by_dataset(models):
    """Group models by dataset"""
    grouped = {}
    for model in models:
        dataset = model['dataset']
        if dataset not in grouped:
            grouped[dataset] = []
        grouped[dataset].append(model)
    return grouped


def print_models_summary():
    """Print a summary of all trained models"""
    
    print("\n" + "="*80)
    print("TRAINED DETECTION MODELS SUMMARY")
    print("="*80)
    
    models = list_trained_models()
    
    if not models:
        print("\nNo trained models found in ./saved_models/")
        print("Train some models first using train_detection_with_dataset_name.py")
        return
    
    print(f"\nTotal trained models: {len(models)}")
    print("\n" + "-"*80)
    print("ALL MODELS")
    print("-"*80)
    print(f"{'Full Name':<50} {'Size (MB)':<12} {'Path'}")
    print("-"*80)
    
    for model in sorted(models, key=lambda x: x['full_name']):
        print(f"{model['full_name']:<50} {model['size_mb']:>10.2f}  {model['path']}")
    
    # Group by model type
    print("\n" + "-"*80)
    print("GROUPED BY MODEL TYPE")
    print("-"*80)
    
    by_type = group_by_model_type(models)
    for model_type in sorted(by_type.keys()):
        print(f"\n{model_type.upper()}:")
        for model in sorted(by_type[model_type], key=lambda x: x['dataset']):
            print(f"  ├─ {model['dataset']:<40} ({model['size_mb']:.2f} MB)")
    
    # Group by dataset
    print("\n" + "-"*80)
    print("GROUPED BY DATASET")
    print("-"*80)
    
    by_dataset = group_by_dataset(models)
    for dataset in sorted(by_dataset.keys()):
        print(f"\n{dataset.upper()}:")
        for model in sorted(by_dataset[dataset], key=lambda x: x['model_type']):
            print(f"  ├─ {model['model_type']:<40} ({model['size_mb']:.2f} MB)")
    
    # Recommendations
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    if models:
        example_model = models[0]
        print(f"""
# Load a model for inference:
from code.detection_inference import DetectionInference

detector = DetectionInference(
    model_name='{example_model['model_type']}',
    model_path='{example_model['path']}',
    num_classes=3  # background + drowning + swimming
)

results = detector.predict('path/to/image.jpg')
print(results)
""")
    
    print("="*80 + "\n")


def compare_datasets_for_model(model_type='ssd300'):
    """Compare how a specific model performs on different datasets"""
    
    print("\n" + "="*80)
    print(f"COMPARE {model_type.upper()} ACROSS DATASETS")
    print("="*80)
    
    models = list_trained_models()
    model_variants = [m for m in models if m['model_type'] == model_type]
    
    if not model_variants:
        print(f"\nNo {model_type} models found.")
        print("Train some first:")
        print(f"  python train_detection_with_dataset_name.py")
        return
    
    print(f"\nFound {len(model_variants)} variants of {model_type}:")
    print("\n" + "-"*80)
    print(f"{'Dataset':<50} {'Model Path'}")
    print("-"*80)
    
    for model in sorted(model_variants, key=lambda x: x['dataset']):
        print(f"{model['dataset']:<50} {model['path']}")
    
    print("\n" + "="*80)
    print("TO COMPARE PERFORMANCE")
    print("="*80)
    print("""
Run inference on the same test images with each model:

from code.detection_inference import DetectionInference

test_images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
datasets = ['dataset_sep_1_2_aug_coco', 'dataset_sep_1_aug_coco', 'dataset_sep_2_aug_coco']

for dataset in datasets:
    print(f"\\nTesting {dataset}:")
    model_path = f'./saved_models/{model_type}_{dataset}/best_model.pth'
    detector = DetectionInference('{model_type}', model_path, num_classes=3)
    
    for img in test_images:
        results = detector.predict(img)
        print(f"  {img}: {len(results['boxes'])} detections")
""")
    print("="*80 + "\n")


def compare_models_for_dataset(dataset='dataset_sep_1_2_aug_coco'):
    """Compare different models trained on the same dataset"""
    
    print("\n" + "="*80)
    print(f"COMPARE MODELS ON {dataset.upper()}")
    print("="*80)
    
    models = list_trained_models()
    dataset_models = [m for m in models if m['dataset'] == dataset]
    
    if not dataset_models:
        print(f"\nNo models found trained on {dataset}.")
        print("Train some first:")
        print(f"  python train_detection_with_dataset_name.py")
        return
    
    print(f"\nFound {len(dataset_models)} models trained on {dataset}:")
    print("\n" + "-"*80)
    print(f"{'Model Type':<30} {'Size (MB)':<12} {'Path'}")
    print("-"*80)
    
    for model in sorted(dataset_models, key=lambda x: x['model_type']):
        print(f"{model['model_type']:<30} {model['size_mb']:>10.2f}  {model['path']}")
    
    print("\n" + "="*80)
    print("TO COMPARE PERFORMANCE")
    print("="*80)
    print(f"""
Run inference on test images with each model:

from code.detection_inference import DetectionInference

test_images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
models = {[m['model_type'] for m in dataset_models]}

for model_type in models:
    print(f"\\nTesting {{model_type}}:")
    model_path = f'./saved_models/{{model_type}}_{dataset}/best_model.pth'
    detector = DetectionInference(model_type, model_path, num_classes=3)
    
    for img in test_images:
        results = detector.predict(img)
        print(f"  {{img}}: {{len(results['boxes'])}} detections")
""")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--by-model':
            # Compare a specific model across datasets
            model_type = sys.argv[2] if len(sys.argv) > 2 else 'ssd300'
            compare_datasets_for_model(model_type)
        elif sys.argv[1] == '--by-dataset':
            # Compare different models on a dataset
            dataset = sys.argv[2] if len(sys.argv) > 2 else 'dataset_sep_1_2_aug_coco'
            compare_models_for_dataset(dataset)
        else:
            print("Usage:")
            print("  python compare_detection_models.py                    # Show all models")
            print("  python compare_detection_models.py --by-model ssd300  # Compare ssd300 across datasets")
            print("  python compare_detection_models.py --by-dataset dataset_sep_1_2_aug_coco  # Compare models on dataset")
    else:
        # Show summary
        print_models_summary()
