"""
Object Detection Queue Examples
Demonstrates how to use the detection queue system
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.detection_queue import DetectionQueue, PriorityDetectionQueue
from code.detection_inference import DetectionInference, get_coco_class_names


def example_basic_queue():
    """
    Example 1: Basic queue usage
    Process a list of images sequentially
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Detection Queue")
    print("="*70)
    
    # Initialize detector
    detector = DetectionInference(
        model_name='faster_rcnn_r50_fpn',
        num_classes=91,  # COCO dataset
        device='cuda',
        confidence_threshold=0.5
    )
    
    # Initialize queue
    queue = DetectionQueue(max_queue_size=100)
    queue.set_detector(detector)
    
    # Define callbacks
    def on_task_start(task):
        print(f"  Starting: {task.task_id} - {os.path.basename(task.image_path)}")
        
    def on_task_complete(task):
        num_detections = task.result['num_detections']
        print(f"  Completed: {task.task_id} - Found {num_detections} objects")
        
    def on_task_error(task, error):
        print(f"  Error: {task.task_id} - {error}")
        
    def on_queue_empty():
        print("\n  Queue is empty!")
        
    # Set callbacks
    queue.on_task_start = on_task_start
    queue.on_task_complete = on_task_complete
    queue.on_task_error = on_task_error
    queue.on_queue_empty = on_queue_empty
    
    # Add some images to the queue
    # Replace these with your actual image paths
    image_dir = "path/to/your/images"
    
    if not os.path.exists(image_dir):
        print(f"\nPlease set a valid image directory in the example script.")
        print(f"Current path: {image_dir}")
        return
    
    image_files = list(Path(image_dir).glob('*.jpg'))[:10]  # Process first 10 images
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\nAdding {len(image_files)} images to queue...")
    task_ids = []
    for img_path in image_files:
        task_id = queue.add_task(str(img_path))
        task_ids.append(task_id)
    
    # Start processing
    print("\nStarting queue processing...")
    queue.start_processing()
    
    # Wait for completion
    while queue.is_running or not queue.queue.empty():
        time.sleep(1)
        info = queue.get_queue_info()
        print(f"  Queue status: {info['completed']} completed, "
              f"{info['processing']} processing, "
              f"{info['pending']} pending")
    
    # Print final statistics
    print("\n" + "-"*70)
    info = queue.get_queue_info()
    print(f"Total processed: {info['total_processed']}")
    print(f"Total failed: {info['total_failed']}")
    print(f"Average processing time: {info['avg_processing_time']:.3f}s")
    
    # Save results
    output_file = "detection_results.json"
    queue.save_results(output_file)
    print(f"\nResults saved to: {output_file}")


def example_priority_queue():
    """
    Example 2: Priority queue usage
    Process important images first
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Priority Detection Queue")
    print("="*70)
    
    # Initialize detector
    detector = DetectionInference(
        model_name='ssd300',
        num_classes=91,
        device='cuda',
        confidence_threshold=0.5
    )
    
    # Initialize priority queue
    queue = PriorityDetectionQueue(max_queue_size=100)
    queue.set_detector(detector)
    
    # Callbacks
    def on_task_start(task):
        priority = task.metadata.get('priority', 0)
        print(f"  [Priority {priority}] Processing: {os.path.basename(task.image_path)}")
        
    def on_task_complete(task):
        print(f"  [Done] {task.task_id}: {task.result['num_detections']} objects")
    
    queue.on_task_start = on_task_start
    queue.on_task_complete = on_task_complete
    
    # Add images with different priorities
    image_dir = "path/to/your/images"
    
    if not os.path.exists(image_dir):
        print(f"\nPlease set a valid image directory in the example script.")
        return
    
    image_files = list(Path(image_dir).glob('*.jpg'))[:15]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\nAdding {len(image_files)} images with priorities...")
    
    # Add images with different priorities
    for i, img_path in enumerate(image_files):
        # Every 3rd image gets high priority (0), others get normal priority (1)
        priority = 0 if i % 3 == 0 else 1
        metadata = {'source': 'example', 'index': i}
        task_id = queue.add_task(str(img_path), metadata=metadata, priority=priority)
        print(f"  Added: {os.path.basename(img_path)} with priority {priority}")
    
    # Start processing
    print("\nStarting priority queue processing...")
    queue.start_processing()
    
    # Wait for completion
    while queue.is_running or not queue.queue.empty():
        time.sleep(1)
    
    print("\nProcessing complete!")


def example_batch_processing():
    """
    Example 3: Batch processing
    Add multiple images at once
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Processing")
    print("="*70)
    
    # Initialize detector
    detector = DetectionInference(
        model_name='faster_rcnn_r50_fpn',
        num_classes=91,
        device='cuda',
        confidence_threshold=0.5
    )
    
    # Initialize queue
    queue = DetectionQueue(max_queue_size=1000)
    queue.set_detector(detector)
    
    # Find images
    image_dir = "path/to/your/images"
    
    if not os.path.exists(image_dir):
        print(f"\nPlease set a valid image directory in the example script.")
        return
    
    image_files = [str(p) for p in Path(image_dir).glob('*.jpg')]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create metadata for each image
    metadata_list = [{'filename': os.path.basename(p)} for p in image_files]
    
    # Add all images as a batch
    print("\nAdding batch of images to queue...")
    task_ids = queue.add_batch_tasks(image_files, metadata_list)
    print(f"Added {len(task_ids)} tasks to queue")
    
    # Start processing
    print("\nStarting batch processing...")
    queue.start_processing()
    
    # Monitor progress
    start_time = time.time()
    last_completed = 0
    
    while queue.is_running or not queue.queue.empty():
        time.sleep(2)
        info = queue.get_queue_info()
        
        # Calculate speed
        elapsed = time.time() - start_time
        completed = info['completed']
        if completed > last_completed:
            speed = completed / elapsed if elapsed > 0 else 0
            eta = (len(task_ids) - completed) / speed if speed > 0 else 0
            print(f"  Progress: {completed}/{len(task_ids)} "
                  f"({completed/len(task_ids)*100:.1f}%) "
                  f"Speed: {speed:.2f} img/s "
                  f"ETA: {eta:.1f}s")
            last_completed = completed
    
    # Final statistics
    elapsed = time.time() - start_time
    print(f"\nBatch processing complete!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average: {elapsed/len(task_ids):.3f}s per image")


def example_with_visualization():
    """
    Example 4: Queue with visualization
    Process images and save visualizations
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Queue with Visualization")
    print("="*70)
    
    # Initialize detector
    detector = DetectionInference(
        model_name='faster_rcnn_r50_fpn',
        num_classes=91,
        device='cuda',
        confidence_threshold=0.5
    )
    
    # Initialize queue
    queue = DetectionQueue(max_queue_size=100)
    queue.set_detector(detector)
    
    # Create output directory
    output_dir = "detection_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class names
    class_names = get_coco_class_names()
    
    # Callback to visualize detections
    def on_task_complete(task):
        print(f"  Visualizing: {os.path.basename(task.image_path)}")
        
        # Create visualization
        output_path = os.path.join(output_dir, f"{task.task_id}.jpg")
        detector.visualize_detections(
            task.image_path,
            detections=task.result,
            class_names=class_names,
            save_path=output_path,
            show=False
        )
        print(f"    Saved to: {output_path}")
    
    queue.on_task_complete = on_task_complete
    
    # Add images
    image_dir = "path/to/your/images"
    
    if not os.path.exists(image_dir):
        print(f"\nPlease set a valid image directory in the example script.")
        return
    
    image_files = list(Path(image_dir).glob('*.jpg'))[:5]  # Process first 5
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\nAdding {len(image_files)} images...")
    for img_path in image_files:
        queue.add_task(str(img_path))
    
    # Start processing
    print("\nStarting processing with visualization...")
    queue.start_processing()
    
    # Wait for completion
    while queue.is_running or not queue.queue.empty():
        time.sleep(1)
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("OBJECT DETECTION QUEUE EXAMPLES")
    print("="*70)
    
    print("\nAvailable examples:")
    print("1. Basic Queue")
    print("2. Priority Queue")
    print("3. Batch Processing")
    print("4. Queue with Visualization")
    print("5. Run all examples")
    
    choice = input("\nSelect example (1-5): ").strip()
    
    if choice == '1':
        example_basic_queue()
    elif choice == '2':
        example_priority_queue()
    elif choice == '3':
        example_batch_processing()
    elif choice == '4':
        example_with_visualization()
    elif choice == '5':
        example_basic_queue()
        example_priority_queue()
        example_batch_processing()
        example_with_visualization()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
