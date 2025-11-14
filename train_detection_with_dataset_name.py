"""
Example: Train Object Detection Models with Dataset-Specific Naming

This script demonstrates how to train detection models with dataset-specific model names.
The models will be saved as: {model_name}_{dataset_name}/best_model.pth

Example: ssd300_dataset_sep_1_2_aug_coco/best_model.pth
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from code.train_detection import run_detection_training


def train_on_buoy_datasets():
    """
    Train object detection models on the buoy datasets
    """
    
    # Dataset configurations
    datasets = [
        {
            'name': 'dataset_sep_1_2_aug_coco',
            'path': 'E:\\dataset_sep_1_2_aug\\dataset_sep_1_2_aug_coco',
            'description': 'Combined Dataset (Sep 1 + 2 Augmented)'
        },
        {
            'name': 'dataset_sep_1_aug_coco',
            'path': 'E:\\dataset_sep_1_2_aug\\dataset_sep_1_aug_coco',
            'description': 'Dataset Sep 1 Augmented'
        },
        {
            'name': 'dataset_sep_2_aug_coco',
            'path': 'E:\\dataset_sep_1_2_aug\\dataset_sep_2_aug_coco',
            'description': 'Dataset Sep 2 Augmented'
        }
    ]
    
    # Detection models to train
    models = [
        'ssd300',
        'faster_rcnn_r50_fpn',
        'retinanet_r50_fpn'
    ]
    
    # Training configuration
    num_classes = 3  # Background + drowning + swimming
    batch_size = 4
    epochs = 50
    lr = 0.001
    
    print("\n" + "="*70)
    print("DETECTION TRAINING WITH DATASET-SPECIFIC MODEL NAMING")
    print("="*70)
    print(f"\nDatasets to train on: {len(datasets)}")
    for ds in datasets:
        print(f"  - {ds['description']}: {ds['name']}")
    
    print(f"\nModels to train: {len(models)}")
    for model in models:
        print(f"  - {model}")
    
    print(f"\nConfiguration:")
    print(f"  Classes: {num_classes} (background, drowning, swimming)")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    
    print(f"\nModel Naming Examples:")
    for model in models:
        for ds in datasets:
            print(f"  - {model}_{ds['name']}/best_model.pth")
    
    print("\n" + "="*70)
    
    # Ask for confirmation
    response = input("\nProceed with training? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nTraining cancelled.")
        return
    
    # Training loop
    total_experiments = len(datasets) * len(models)
    current_experiment = 0
    
    for dataset in datasets:
        for model_name in models:
            current_experiment += 1
            
            print("\n" + "="*70)
            print(f"TRAINING {current_experiment}/{total_experiments}")
            print("="*70)
            print(f"Model: {model_name}")
            print(f"Dataset: {dataset['description']}")
            print(f"Save Name: {model_name}_{dataset['name']}")
            print("="*70 + "\n")
            
            try:
                run_detection_training(
                    model_name=model_name,
                    num_classes=num_classes,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr,
                    num_workers=4,
                    data_path=dataset['path'],
                    dataset_format='coco',
                    save_dir='./saved_models',
                    dataset_name=dataset['name']  # This creates model_name_dataset_name folder
                )
                
                print(f"\n✓ SUCCESS: {model_name} trained on {dataset['name']}")
                print(f"   Saved to: ./saved_models/{model_name}_{dataset['name']}/best_model.pth")
                
            except Exception as e:
                print(f"\n✗ ERROR: Failed to train {model_name} on {dataset['name']}")
                print(f"   Error: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Models saved in: ./saved_models/")
    print("\nTo use a trained model:")
    print("  from code.detection_inference import DetectionInference")
    print("  model_path = './saved_models/ssd300_dataset_sep_1_2_aug_coco/best_model.pth'")
    print("  detector = DetectionInference('ssd300', model_path, num_classes=3)")
    print("  results = detector.predict('path/to/image.jpg')")
    print("\n" + "="*70)


def train_single_model():
    """
    Train a single model on a single dataset
    """
    
    print("\n" + "="*70)
    print("SINGLE MODEL TRAINING")
    print("="*70)
    
    # Example: Train SSD300 on combined dataset
    run_detection_training(
        model_name='ssd300',
        num_classes=3,  # background + drowning + swimming
        batch_size=4,
        epochs=50,
        lr=0.001,
        num_workers=4,
        data_path='E:\\dataset_sep_1_2_aug\\dataset_sep_1_2_aug_coco',
        dataset_format='coco',
        save_dir='./saved_models',
        dataset_name='dataset_sep_1_2_aug_coco'  # Model will be saved as ssd300_dataset_sep_1_2_aug_coco
    )
    
    print("\n✓ Training completed!")
    print("Model saved to: ./saved_models/ssd300_dataset_sep_1_2_aug_coco/best_model.pth")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--single':
        # Train single model
        train_single_model()
    else:
        # Train all combinations
        train_on_buoy_datasets()
