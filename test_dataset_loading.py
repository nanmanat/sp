"""
Test dataset loading to ensure images are converted to tensors
"""

import torch
from code.detection_dataset import create_detection_dataset

# Test COCO dataset loading
print("Testing COCO dataset with tensor conversion...")
print("="*60)

try:
    dataset = create_detection_dataset(
        data_path='E:/Buoy/dataset_coco',
        dataset_format='coco',
        split='train'
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Load first sample
    image, target = dataset[0]
    
    print(f"\n✓ First sample loaded:")
    print(f"  Image type: {type(image)}")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Boxes shape: {target['boxes'].shape}")
    print(f"  Labels shape: {target['labels'].shape}")
    print(f"  Number of objects: {len(target['boxes'])}")
    
    # Check if image is a tensor
    if isinstance(image, torch.Tensor):
        print(f"\n✓ SUCCESS! Image is a PyTorch tensor")
        print(f"  Can call .to() method: {hasattr(image, 'to')}")
        
        # Test moving to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_gpu = image.to(device)
        print(f"  Can move to {device}: ✓")
    else:
        print(f"\n✗ ERROR! Image is {type(image)}, not a tensor")
    
    print("\n" + "="*60)
    print("✓ All checks passed! Dataset is ready for training.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
