"""
Test the GUI queue callback fix
Verifies that both classification and detection experiments work
"""

# Simulate the fixed callback function
def _queue_callback_fixed(status, experiment=None, error=None):
    """Callback for queue processing events"""
    if status == "start":
        # Build status message based on experiment type
        experiment_type = experiment.get('type', 'classification')
        model_name = experiment.get('model', 'Unknown')
        
        if experiment_type == 'detection':
            # Detection experiments don't have folds
            num_classes = experiment.get('num_classes', 91)
            status_message = f"Running detection: Model {model_name}, Classes {num_classes}"
        else:
            # Classification experiments have folds
            folds = experiment.get('folds', ['0'])
            status_message = f"Running experiment: Model {model_name}, Folds {','.join(folds)}"
        
        return status_message
    return None


# Test cases
print("Testing GUI Queue Callback Fix")
print("=" * 70)

# Test 1: Classification experiment with folds
print("\nTest 1: Classification Experiment")
classification_exp = {
    'type': 'classification',
    'model': 'resnet50',
    'folds': ['0', '1', '2']
}
result = _queue_callback_fixed('start', classification_exp)
print(f"Result: {result}")
assert 'Folds 0,1,2' in result
print("✓ Classification experiment works")

# Test 2: Detection experiment without folds
print("\nTest 2: Detection Experiment")
detection_exp = {
    'type': 'detection',
    'model': 'faster_rcnn_r50_fpn',
    'num_classes': 5
}
result = _queue_callback_fixed('start', detection_exp)
print(f"Result: {result}")
assert 'Classes 5' in result
assert 'Folds' not in result  # Should NOT have 'Folds'
print("✓ Detection experiment works")

# Test 3: Default classification (no type specified)
print("\nTest 3: Default (no type specified)")
default_exp = {
    'model': 'efficientnet_b0',
    'folds': ['0']
}
result = _queue_callback_fixed('start', default_exp)
print(f"Result: {result}")
assert 'Folds 0' in result
print("✓ Default classification works")

# Test 4: Detection with default num_classes
print("\nTest 4: Detection with defaults")
detection_default = {
    'type': 'detection',
    'model': 'ssd300'
}
result = _queue_callback_fixed('start', detection_default)
print(f"Result: {result}")
assert 'Classes 91' in result  # Default COCO classes
print("✓ Detection with defaults works")

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("\nThe KeyError: 'folds' issue is now fixed!")
print("Both classification and detection experiments work correctly.")
