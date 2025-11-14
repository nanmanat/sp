"""
Simple test for Detection Queue (no PyTorch required)
Tests basic queue functionality without requiring torch
"""

import os
import sys
import time
import tempfile
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.detection_queue import DetectionQueue, PriorityDetectionQueue, DetectionTask


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
    
    print("✓ DetectionTask class working correctly")
    return True


def test_queue_basic():
    """Test basic queue functionality without detector"""
    print("\n" + "="*70)
    print("TEST: Basic Queue")
    print("="*70)
    
    queue = DetectionQueue(max_queue_size=10)
    
    # Create temporary test files (just paths, no actual images)
    test_images = ["test_1.jpg", "test_2.jpg", "test_3.jpg"]
    
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
        print(f"  Task {task_id}: {status['status']}")
    
    # Get queue info
    info = queue.get_queue_info()
    assert info["pending"] == 3
    assert info["completed"] == 0
    print(f"  Queue info: {info['pending']} pending, {info['completed']} completed")
    
    print("✓ Basic queue operations working correctly")
    return True


def test_batch_operations():
    """Test batch operations"""
    print("\n" + "="*70)
    print("TEST: Batch Operations")
    print("="*70)
    
    queue = DetectionQueue(max_queue_size=100)
    
    # Create test image paths
    test_images = [f"batch_{i}.jpg" for i in range(5)]
    
    # Add as batch
    metadata_list = [{"index": i, "batch": True} for i in range(5)]
    task_ids = queue.add_batch_tasks(test_images, metadata_list)
    
    assert len(task_ids) == 5
    print(f"  Added {len(task_ids)} tasks in batch")
    
    # Verify metadata
    for i, task_id in enumerate(task_ids):
        status = queue.get_task_status(task_id)
        assert status["metadata"]["index"] == i
        assert status["metadata"]["batch"] == True
    
    print("✓ Batch operations working correctly")
    return True


def test_priority_queue():
    """Test priority queue"""
    print("\n" + "="*70)
    print("TEST: Priority Queue")
    print("="*70)
    
    queue = PriorityDetectionQueue(max_queue_size=10)
    
    # Create test image paths
    test_images = [f"priority_{i}.jpg" for i in range(3)]
    
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
    print(f"  Queue has {info['pending']} pending tasks")
    
    print("✓ Priority queue working correctly")
    return True


def test_queue_callbacks():
    """Test callback system"""
    print("\n" + "="*70)
    print("TEST: Queue Callbacks")
    print("="*70)
    
    queue = DetectionQueue(max_queue_size=10)
    
    # Track callbacks
    callbacks_called = {
        'start': [],
        'complete': [],
        'error': []
    }
    
    def on_start(task):
        callbacks_called['start'].append(task.task_id)
    
    def on_complete(task):
        callbacks_called['complete'].append(task.task_id)
    
    def on_error(task, error):
        callbacks_called['error'].append(task.task_id)
    
    # Set callbacks
    queue.on_task_start = on_start
    queue.on_task_complete = on_complete
    queue.on_task_error = on_error
    
    print("  Callbacks set successfully")
    print("✓ Callback system initialized correctly")
    return True


def test_save_results():
    """Test saving results to JSON"""
    print("\n" + "="*70)
    print("TEST: Save Results")
    print("="*70)
    
    queue = DetectionQueue(max_queue_size=10)
    
    # Add some tasks
    for i in range(3):
        task_id = queue.add_task(f"image_{i}.jpg", metadata={"index": i})
        # Manually mark as completed for testing
        task = queue.tasks[task_id]
        task.status = "completed"
        task.result = {"num_detections": i + 1, "boxes": [], "scores": [], "labels": []}
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.close()
    
    try:
        queue.save_results(temp_file.name)
        
        # Read and verify
        with open(temp_file.name, 'r') as f:
            results = json.load(f)
        
        assert len(results) == 3
        print(f"  Saved {len(results)} task results")
        
        # Verify structure
        for result in results:
            assert 'task_id' in result
            assert 'status' in result
            assert 'result' in result
        
        print("✓ Save results working correctly")
        return True
    finally:
        # Cleanup
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def test_queue_info():
    """Test queue information retrieval"""
    print("\n" + "="*70)
    print("TEST: Queue Info")
    print("="*70)
    
    queue = DetectionQueue(max_queue_size=10)
    
    # Add tasks with different statuses
    task_ids = []
    for i in range(5):
        task_id = queue.add_task(f"info_test_{i}.jpg")
        task_ids.append(task_id)
    
    # Manually set some statuses for testing
    queue.tasks[task_ids[0]].status = "completed"
    queue.tasks[task_ids[1]].status = "completed"
    queue.tasks[task_ids[2]].status = "failed"
    
    # Get info
    info = queue.get_queue_info()
    
    print(f"  Pending: {info['pending']}")
    print(f"  Completed: {info['completed']}")
    print(f"  Failed: {info['failed']}")
    print(f"  Total processed: {info['total_processed']}")
    
    assert info['completed'] == 2
    assert info['failed'] == 1
    
    print("✓ Queue info retrieval working correctly")
    return True


def test_clear_tasks():
    """Test clearing completed tasks"""
    print("\n" + "="*70)
    print("TEST: Clear Completed Tasks")
    print("="*70)
    
    queue = DetectionQueue(max_queue_size=10)
    
    # Add and complete some tasks
    for i in range(5):
        task_id = queue.add_task(f"clear_test_{i}.jpg")
        if i < 3:
            queue.tasks[task_id].status = "completed"
    
    initial_count = len(queue.tasks)
    print(f"  Initial task count: {initial_count}")
    
    # Clear completed
    queue.clear_completed_tasks()
    
    final_count = len(queue.tasks)
    print(f"  Final task count: {final_count}")
    
    assert final_count == 2  # Only 2 pending tasks remain
    
    print("✓ Clear completed tasks working correctly")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("DETECTION QUEUE SYSTEM - BASIC TESTS")
    print("="*70)
    print("\nNote: These tests verify queue functionality without PyTorch")
    print("For full testing with models, ensure PyTorch is installed\n")
    
    tests = [
        ("Detection Task", test_detection_task),
        ("Basic Queue", test_queue_basic),
        ("Batch Operations", test_batch_operations),
        ("Priority Queue", test_priority_queue),
        ("Queue Callbacks", test_queue_callbacks),
        ("Save Results", test_save_results),
        ("Queue Info", test_queue_info),
        ("Clear Tasks", test_clear_tasks),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST RESULTS")
    print("="*70)
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
