"""
Object Detection Training Module
Handles training and evaluation of object detection models
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.object_detection_models import create_detection_model, get_model_info
from code.detection_dataset import create_detection_dataset, collate_fn, collate_fn_efficientdet


class DetectionTrainer:
    """Handles training of object detection models"""
    
    def __init__(self, model_name, num_classes=91, device='cuda', dataset_name=None):
        """
        Initialize the detection trainer
        
        Args:
            model_name: Name of the detection model
            num_classes: Number of object classes (including background)
            device: Device to train on ('cuda' or 'cpu')
            dataset_name: Name of the dataset (optional, used for model naming)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing {model_name} for object detection...")
        print(f"Device: {self.device}")
        
        # Create model
        self.model = create_detection_model(model_name, num_classes=num_classes)
        self.model.to(self.device)
        
        # Get model info
        self.model_info = get_model_info(model_name)
        print(f"Model: {self.model_info['description']}")
        
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, save_dir='./saved_models'):
        """
        Train the detection model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            save_dir: Directory to save model checkpoints
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {lr}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"{'='*60}\n")
        
        # Setup optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0001)
        
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Create save directory with dataset name if provided
        if self.dataset_name:
            model_dir_name = f"{self.model_name}_{self.dataset_name}"
        else:
            model_dir_name = self.model_name
        
        model_save_dir = os.path.join(save_dir, model_dir_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        print(f"Models will be saved to: {model_save_dir}")
        
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Train phase
            self.model.train()
            train_loss = self._train_epoch(train_loader, optimizer)
            
            # Validation phase (keep model in train mode to get losses)
            self.model.train()  # Keep in train mode for loss calculation
            val_loss = self._validate_epoch(val_loader)
            
            # Update learning rate
            lr_scheduler.step()
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(model_save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
    def _train_epoch(self, data_loader, optimizer):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        for images, targets in data_loader:
            # Debug: Print types on first batch
            if num_batches == 0:
                print(f"  Debug - Images type: {type(images)}, shape: {images.shape if hasattr(images, 'shape') else 'N/A'}")
                print(f"  Debug - Targets type: {type(targets)}")
                if isinstance(targets, list):
                    print(f"  Debug - Targets length: {len(targets)}")
                    if len(targets) > 0:
                        print(f"  Debug - First target type: {type(targets[0])}")
                        if isinstance(targets[0], dict):
                            print(f"  Debug - First target keys: {list(targets[0].keys())}")
                elif isinstance(targets, dict):
                    print(f"  Debug - Targets is dict with keys: {list(targets.keys())}")
                    for k, v in targets.items():
                        if torch.is_tensor(v):
                            print(f"  Debug -   {k}: shape={v.shape}, dtype={v.dtype}")
            
            # Move to device
            # Check if images is already a stacked tensor (from collate_fn_efficientdet) or list
            if isinstance(images, torch.Tensor):
                images = images.to(self.device)
            else:
                images = [img.to(self.device) for img in images]
            
            # Move targets to device
            if isinstance(targets, list):
                # List of dicts (torchvision format)
                targets = [{k: v.to(self.device) if torch.is_tensor(v) else v 
                           for k, v in t.items()} if isinstance(t, dict) else t 
                          for t in targets]
            elif isinstance(targets, dict):
                # Dict of lists (EfficientDet format) or single dict
                # Check if values are lists
                first_value = next(iter(targets.values()))
                if isinstance(first_value, list):
                    # EfficientDet format: dict of lists
                    targets = {k: [t.to(self.device) if torch.is_tensor(t) else t for t in v]
                              for k, v in targets.items()}
                else:
                    # Single dict (unexpected but handle it)
                    targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                              for k, v in targets.items()}
            else:
                raise ValueError(f"Unexpected targets type: {type(targets)}")
            
            # Debug on first batch
            if num_batches == 0:
                print(f"  Debug - After moving to device:")
                if isinstance(images, torch.Tensor):
                    print(f"  Debug - Images device: {images.device}")
                else:
                    print(f"  Debug - Images is list, first device: {images[0].device if len(images) > 0 else 'N/A'}")
                print(f"  Debug - Targets type: {type(targets)}")
                if isinstance(targets, list) and len(targets) > 0:
                    print(f"  Debug - Targets length: {len(targets)}")
                    print(f"  Debug - First target keys: {list(targets[0].keys()) if isinstance(targets[0], dict) else 'N/A'}")
                    if isinstance(targets[0], dict):
                        for k, v in targets[0].items():
                            if torch.is_tensor(v):
                                print(f"  Debug -   {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                elif isinstance(targets, dict):
                    print(f"  Debug - Targets keys: {list(targets.keys())}")
                    first_value = next(iter(targets.values()))
                    if isinstance(first_value, list):
                        print(f"  Debug - First value is list, length: {len(first_value)}")
            
            # Forward pass
            try:
                loss_dict = self.model(images, targets)
            except Exception as e:
                print(f"  Error in model forward pass!")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print(f"  Images shape: {images.shape}, device: {images.device}")
                print(f"  Targets type: {type(targets)}")
                if isinstance(targets, dict):
                    print(f"  Targets keys: {list(targets.keys())}")
                    for k, v in targets.items():
                        if torch.is_tensor(v):
                            print(f"    {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                            if k in ['bbox', 'cls'] and len(v) > 0:
                                print(f"      First few values: {v[:min(3, len(v))]}")
                elif isinstance(targets, list):
                    print(f"  Number of targets: {len(targets)}")
                    for i, t in enumerate(targets[:2]):  # Show first 2
                        print(f"  Target {i}: type={type(t)}, keys={list(t.keys()) if isinstance(t, dict) else 'N/A'}")
                        if isinstance(t, dict):
                            for k, v in t.items():
                                print(f"    {k}: type={type(v)}, shape={v.shape if torch.is_tensor(v) else 'N/A'}")
                import traceback
                traceback.print_exc()
                raise
            
            # Calculate total loss
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            else:
                losses = loss_dict
            
            # Handle case where losses might be a list or tuple
            if isinstance(losses, (list, tuple)):
                losses = sum(losses)
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # Accumulate loss (safely extract scalar)
            loss_value = losses.item() if hasattr(losses, 'item') else float(losses)
            total_loss += loss_value
            num_batches += 1
            
            # Print progress
            if num_batches % 10 == 0:
                print(f"  Batch {num_batches}: Loss = {loss_value:.4f}")
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _validate_epoch(self, data_loader):
        """Validate for one epoch"""
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in data_loader:
                # Move to device
                if isinstance(images, torch.Tensor):
                    images = images.to(self.device)
                else:
                    images = [img.to(self.device) for img in images]
                
                # Move targets to device
                if isinstance(targets, list):
                    # List of dicts (torchvision format)
                    targets = [{k: v.to(self.device) if torch.is_tensor(v) else v 
                               for k, v in t.items()} if isinstance(t, dict) else t 
                              for t in targets]
                elif isinstance(targets, dict):
                    # Dict of lists (EfficientDet format) or single dict
                    first_value = next(iter(targets.values()))
                    if isinstance(first_value, list):
                        # EfficientDet format: dict of lists
                        targets = {k: [t.to(self.device) if torch.is_tensor(t) else t for t in v]
                                  for k, v in targets.items()}
                    else:
                        # Single dict
                        targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                                  for k, v in targets.items()}
                else:
                    raise ValueError(f"Unexpected targets type: {type(targets)}")
                
                # Forward pass
                loss_dict = self.model(images, targets)
                
                # Calculate total loss
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                else:
                    losses = loss_dict
                
                # Handle case where losses might be a list or tuple
                if isinstance(losses, (list, tuple)):
                    losses = sum(losses)
                
                # Accumulate loss (safely extract scalar)
                loss_value = losses.item() if hasattr(losses, 'item') else float(losses)
                total_loss += loss_value
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0


def run_detection_training(
    model_name='faster_rcnn_r50_fpn',
    num_classes=91,
    batch_size=4,
    epochs=50,
    lr=0.001,
    num_workers=4,
    data_path=None,
    dataset_format='coco',
    save_dir='./saved_models',
    dataset_name=None
):
    """
    Run object detection training
    
    Args:
        model_name: Name of the detection model
        num_classes: Number of object classes (including background)
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        num_workers: Number of data loading workers
        data_path: Path to dataset
        dataset_format: Format of dataset ('coco', 'voc', or 'yolo')
        save_dir: Directory to save models
        dataset_name: Name of the dataset (for model naming, e.g., 'dataset_sep_1_2_aug_coco')
    """
    print("\n" + "="*60)
    print("OBJECT DETECTION TRAINING")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Classes: {num_classes}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    if data_path:
        print(f"Dataset Format: {dataset_format}")
        print(f"Dataset Path: {data_path}")
    if dataset_name:
        print(f"Dataset Name: {dataset_name}")
    print("="*60 + "\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, training will be slow on CPU")
    
    # Auto-extract dataset name from path if not provided
    if dataset_name is None and data_path:
        # Extract last directory name from path
        dataset_name = os.path.basename(os.path.normpath(data_path))
        print(f"Auto-detected dataset name: {dataset_name}")
    
    # Create trainer
    trainer = DetectionTrainer(model_name, num_classes=num_classes, dataset_name=dataset_name)
    
    # Check if dataset path is provided
    if data_path is None or not os.path.exists(data_path):
        print("\n⚠️  No valid dataset path provided")
        print("\nTo start training, you need to provide:")
        print("  1. data_path: Path to your dataset directory")
        print("  2. dataset_format: One of 'coco', 'voc', or 'yolo'")
        print("\nDataset format structures:")
        print("\n  COCO format:")
        print("    data_path/annotations/instances_train.json")
        print("    data_path/images/ (or train2017/)")
        print("\n  VOC format:")
        print("    data_path/Annotations/")
        print("    data_path/JPEGImages/")
        print("    data_path/ImageSets/Main/train.txt")
        print("\n  YOLO format:")
        print("    data_path/images/train/")
        print("    data_path/labels/train/")
        print("\nSee OBJECT_DETECTION_GUIDE.md for detailed dataset preparation.")
        print("\n✓ Model initialized successfully!")
        print("Provide dataset path to begin training.")
        return
    
    try:
        # Load datasets
        print(f"Loading {dataset_format.upper()} format dataset...")
        print(f"Loading training dataset from: {data_path}")
        train_dataset = create_detection_dataset(
            data_path,
            dataset_format=dataset_format,
            split='train',
            transform=None  # Add transforms if needed
        )
        
        print("Loading validation dataset...")
        val_dataset = create_detection_dataset(
            data_path,
            dataset_format=dataset_format,
            split='val',
            transform=None
        )
        
        # Create data loaders
        # Choose appropriate collate function based on model
        if model_name.startswith('efficientdet'):
            collate_function = collate_fn_efficientdet
            print("  Using EfficientDet-specific batch processing (resizing to 512x512)")
        else:
            collate_function = collate_fn
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_function
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_function
        )
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}\n")
        
        # Start training
        print("Starting training...")
        trainer.train(train_loader, val_loader, epochs=epochs, lr=lr, save_dir=save_dir)
        
    except FileNotFoundError as e:
        print(f"\n❌ Dataset Error: {e}")
        print("\nPlease check:")
        print("  1. Dataset path is correct")
        print(f"  2. Dataset format '{dataset_format}' matches your actual dataset")
        print("  3. Required files/folders exist (see error message above)")
        print("\nSupported formats: 'coco', 'voc', 'yolo'")
        print("See OBJECT_DETECTION_GUIDE.md for dataset format details.")
        return
    
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ Detection training completed!")


if __name__ == "__main__":
    # Example usage
    run_detection_training(
        model_name='faster_rcnn_r50_fpn',
        num_classes=91,
        batch_size=4,
        epochs=50,
        lr=0.001,
        data_path='./datasets/coco',
        dataset_format='coco',
        dataset_name='coco'  # Optional: will be auto-detected from data_path if not provided
    )

