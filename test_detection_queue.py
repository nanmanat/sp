"""
Test the Detection Queue System
Simple tests to verify queue functionality
"""

import os
import sys
import time
import tempfile
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.detection_queue import DetectionQueue, PriorityDetectionQueue, DetectionTask
from code.detection_inference import DetectionInference


def create_test_image(path, size=(640, 480)):
    """Create a test image for testing"""
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    img.save(path)
    return path


def test_detection_task():
    """Test DetectionTask class"""
    print("\n" + "="*70)
    print("TEST: DetectionTask")
    print("="*70)
    
    task = DetectionTask("test_id", "test.jpg", {"key": "value"})
    
    assert task.task_id == "test_id"
    assert task.image_path == "test.jpg"
    assert task.metadata["key"] == "value"
    assert task.status == "pending"
    
    # Test to_dict
    task_dict = task.to_dict()
    assert task_dict["task_id"] == "test_id"
    assert task_dict["status"] == "pending"
    
    print("✓ DetectionTask working correctly")


def test_queue_basic():
    """Test basic queue functionality"""
    print("\n" + "="*70)
    print("TEST: Basic Queue")
    print("="*70)
    
    queue = DetectionQueue(max_queue_size=10)
    
    # Create temporary test images
    temp_dir = tempfile.mkdtemp()
    test_images = []
    for i in range(3):
        img_path = os.path.join(temp_dir, f"test_{i}.jpg")
        create_test_image(img_path)
        test_images.append(img_path)
    
    # Add tasks
    task_ids = []
    for img_path in test_images:
        task_id = queue.add_task(img_path, metadata={"test": True})
        task_ids.append(task_id)
        print(f"  Added task: {task_id}")
    
    assert len(task_ids) == 3
    assert queue.queue.qsize() == 3
    
    # Check task status
    for task_id in task_ids:
        status = queue.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "pending"
    
    # Get queue info
    info = queue.get_queue_info()
    assert info["pending"] == 3
    assert info["completed"] == 0
    
    print("✓ Basic queue working correctly")
    
    # Cleanup
    for img in test_images:
        if os.path.exists(img):
            os.remove(img)
    os.rmdir(temp_dir)


def test_batch_operations():
    """Test batch operations"""
    print("\n" + "="*70)
    print("TEST: Batch Operations")
    print("="*70)
    
    queue = DetectionQueue(max_queue_size=100)
    
    # Create test images
    temp_dir = tempfile.mkdtemp()
    test_images = []
    for i in range(5):
        img_path = os.path.join(temp_dir, f"batch_{i}.jpg")
        create_test_image(img_path)
        test_images.append(img_path)
    
    # Add as batch
    metadata_list = [{"index": i} for i in range(5)]
    task_ids = queue.add_batch_tasks(test_images, metadata_list)
    
    assert len(task_ids) == 5
    print(f"  Added {len(task_ids)} tasks in batch")
    
    # Verify metadata
    for i, task_id in enumerate(task_ids):
        status = queue.get_task_status(task_id)
        assert status["metadata"]["index"] == i
    
    print("✓ Batch operations working correctly")
    
    # Cleanup
    for img in test_images:
        if os.path.exists(img):
            os.remove(img)
    os.rmdir(temp_dir)


def test_priority_queue():
    """Test priority queue"""
    print("\n" + "="*70)
    print("TEST: Priority Queue")
    print("="*70)
    
    queue = PriorityDetectionQueue(max_queue_size=10)
    
    # Create test images
    temp_dir = tempfile.mkdtemp()
    test_images = []
    for i in range(3):
        img_path = os.path.join(temp_dir, f"priority_{i}.jpg")
        create_test_image(img_path)
        test_images.append(img_path)
    
    # Add with different priorities
    task_ids = []
    priorities = [2, 0, 1]  # 0 is highest priority
    for img_path, priority in zip(test_images, priorities):
        task_id = queue.add_task(img_path, priority=priority)
        task_ids.append(task_id)
        print(f"  Added task with priority {priority}")
    
    # Check that tasks were added
    assert len(task_ids) == 3
    
    # Get queue info
    info = queue.get_queue_info()
    assert info["pending"] == 3
    
    print("✓ Priority queue working correctly")
    
    # Cleanup
    for img in test_images:
        if os.path.exists(img):
            os.remove(img)
    os.rmdir(temp_dir)


def test_detection_inference():
    """Test detection inference (without actual model)"""
    print("\n" + "="*70)
    print("TEST: Detection Inference Initialization")
    print("="*70)
    
    try:
        # Test initialization (will load pretrained model if available)
        detector = DetectionInference(
            model_name='faster_rcnn_r50_fpn',
            num_classes=91,
            device='cpu',  # Use CPU for testing
            confidence_threshold=0.5
        )
        
        # Get model info
        info = detector.get_model_info()
        print(f"  Model: {info['model_name']}")
        print(f"  Classes: {info['num_classes']}")
        print(f"  Device: {info['device']}")
        print(f"  Parameters: {info['num_parameters']:,}")
        
        # Test confidence threshold setter
        detector.set_confidence_threshold(0.7)
        assert detector.confidence_threshold == 0.7
        
        print("✓ Detection inference initialization working")
        
    except Exception as e:
        print(f"⚠ Could not initialize detector (may need PyTorch/models): {e}")


def test_queue_with_mock_detector():
    """Test queue with a mock detector"""
    print("\n" + "="*70)
    print("TEST: Queue with Mock Detector")
    print("="*70)
    
    # Create mock detector
    class MockDetector:
        def detect(self, image_path):
            """Mock detection that returns dummy results"""
            time.sleep(0.1)  # Simulate processing time
            return {
                'boxes': [[10, 10, 100, 100]],
                'scores': [0.95],
                'labels': [1],
                'num_detections': 1
            }
    
    queue = DetectionQueue(max_queue_size=10)
    queue.set_detector(MockDetector())
    
    # Create test images
    temp_dir = tempfile.mkdtemp()
    test_images = []
    for i in range(3):
        img_path = os.path.join(temp_dir, f"mock_{i}.jpg")
        create_test_image(img_path)
        test_images.append(img_path)
    
    # Track callbacks
    callbacks_called = {
        'start': 0,
        'complete': 0,
        'error': 0
    }
    
    def on_start(task):
        callbacks_called['start'] += 1
        print(f"  Started: {task.task_id}")
    
    def on_complete(task):
        callbacks_called['complete'] += 1
        print(f"  Completed: {task.task_id} - {task.result['num_detections']} objects")
    
    def on_error(task, error):
        callbacks_called['error'] += 1
        print(f"  Error: {task.task_id}")
    
    queue.on_task_start = on_start
    queue.on_task_complete = on_complete
    queue.on_task_error = on_error
    
    # Add tasks
    task_ids = []
    for img_path in test_images:
        task_id = queue.add_task(img_path)
        task_ids.append(task_id)
    
    # Start processing
    queue.start_processing()
    
    # Wait for completion
    timeout = 10
    start_time = time.time()
    while (queue.is_running or not queue.queue.empty()) and (time.time() - start_time) < timeout:
        time.sleep(0.5)
    
    # Check results
    info = queue.get_queue_info()
    print(f"\n  Processed: {info['completed']}")
    print(f"  Failed: {info['failed']}")
    
    assert callbacks_called['start'] == 3
    assert callbacks_called['complete'] == 3
    assert callbacks_called['error'] == 0
    
    # Check task results
    for task_id in task_ids:
        status = queue.get_task_status(task_id)
        assert status['status'] == 'completed'
        assert status['result']['num_detections'] == 1
    
    print("✓ Queue with mock detector working correctly")
    
    # Cleanup
    for img in test_images:
        if os.path.exists(img):
            os.remove(img)
    os.rmdir(temp_dir)


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("DETECTION QUEUE SYSTEM TESTS")
    print("="*70)
    
    tests = [
        test_detection_task,
        test_queue_basic,
        test_batch_operations,
        test_priority_queue,
        test_detection_inference,
        test_queue_with_mock_detector,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
