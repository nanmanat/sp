"""
Object Detection Queue System
Manages queuing and processing of object detection inference tasks
"""

import os
import queue
import threading
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

# Optional imports - these are only needed when using the queue with actual detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class DetectionTask:
    """Represents a single detection task"""
    
    def __init__(self, task_id: str, image_path: str, metadata: Optional[Dict] = None):
        """
        Initialize a detection task
        
        Args:
            task_id: Unique identifier for the task
            image_path: Path to the image to process
            metadata: Optional metadata associated with the task
        """
        self.task_id = task_id
        self.image_path = image_path
        self.metadata = metadata or {}
        self.status = "pending"  # pending, processing, completed, failed
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        
    def to_dict(self) -> Dict:
        """Convert task to dictionary"""
        return {
            'task_id': self.task_id,
            'image_path': self.image_path,
            'metadata': self.metadata,
            'status': self.status,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


class DetectionQueue:
    """
    Queue system for managing object detection tasks
    Supports both synchronous and asynchronous processing
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize the detection queue
        
        Args:
            max_queue_size: Maximum number of tasks in the queue
        """
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.tasks = {}  # task_id -> DetectionTask
        self.processing_thread = None
        self.is_running = False
        self.stop_requested = False
        self.current_task = None
        self.detector = None
        self.lock = threading.Lock()
        
        # Callbacks
        self.on_task_start: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        self.on_task_error: Optional[Callable] = None
        self.on_queue_empty: Optional[Callable] = None
        
        # Statistics
        self.total_processed = 0
        self.total_failed = 0
        self.processing_times = []
        
        # Task ID counter for uniqueness
        self._task_counter = 0
        self._counter_lock = threading.Lock()
        
    def set_detector(self, detector):
        """
        Set the detector instance to use for processing
        
        Args:
            detector: DetectionInference instance
        """
        self.detector = detector
        
    def add_task(self, image_path: str, task_id: Optional[str] = None, 
                 metadata: Optional[Dict] = None) -> str:
        """
        Add a detection task to the queue
        
        Args:
            image_path: Path to the image
            task_id: Optional unique ID (will be generated if not provided)
            metadata: Optional metadata for the task
            
        Returns:
            task_id: The ID of the added task
        """
        if task_id is None:
            task_id = self._generate_task_id()
            
        task = DetectionTask(task_id, image_path, metadata)
        
        with self.lock:
            self.tasks[task_id] = task
            
        try:
            self.queue.put(task, block=False)
            return task_id
        except queue.Full:
            with self.lock:
                del self.tasks[task_id]
            raise RuntimeError("Queue is full. Cannot add more tasks.")
            
    def add_batch_tasks(self, image_paths: List[str], 
                       metadata_list: Optional[List[Dict]] = None) -> List[str]:
        """
        Add multiple detection tasks to the queue
        
        Args:
            image_paths: List of image paths
            metadata_list: Optional list of metadata dicts (must match length of image_paths)
            
        Returns:
            List of task IDs
        """
        if metadata_list is None:
            metadata_list = [None] * len(image_paths)
            
        if len(image_paths) != len(metadata_list):
            raise ValueError("image_paths and metadata_list must have same length")
            
        task_ids = []
        for img_path, metadata in zip(image_paths, metadata_list):
            try:
                task_id = self.add_task(img_path, metadata=metadata)
                task_ids.append(task_id)
            except RuntimeError as e:
                print(f"Failed to add task for {img_path}: {e}")
                
        return task_ids
        
    def start_processing(self) -> bool:
        """
        Start processing the queue in a background thread
        
        Returns:
            True if started successfully, False if already running
        """
        if self.is_running:
            return False
            
        if self.detector is None:
            raise RuntimeError("No detector set. Call set_detector() first.")
            
        self.stop_requested = False
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        return True
        
    def stop_processing(self, wait: bool = True) -> bool:
        """
        Stop processing the queue
        
        Args:
            wait: If True, wait for current task to complete
            
        Returns:
            True if stopped successfully
        """
        self.stop_requested = True
        if wait and self.processing_thread:
            self.processing_thread.join(timeout=30)
        return True
        
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        Get the status of a task
        
        Args:
            task_id: The task ID
            
        Returns:
            Task information as dictionary, or None if not found
        """
        with self.lock:
            task = self.tasks.get(task_id)
            return task.to_dict() if task else None
            
    def get_queue_info(self) -> Dict:
        """
        Get information about the queue state
        
        Returns:
            Dictionary with queue information
        """
        with self.lock:
            pending = sum(1 for t in self.tasks.values() if t.status == "pending")
            processing = sum(1 for t in self.tasks.values() if t.status == "processing")
            completed = sum(1 for t in self.tasks.values() if t.status == "completed")
            failed = sum(1 for t in self.tasks.values() if t.status == "failed")
            
            avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            
        return {
            'is_running': self.is_running,
            'queue_size': self.queue.qsize(),
            'pending': pending,
            'processing': processing,
            'completed': completed,
            'failed': failed,
            'total_processed': self.total_processed,
            'total_failed': self.total_failed,
            'avg_processing_time': avg_time,
            'current_task': self.current_task.task_id if self.current_task else None
        }
        
    def clear_completed_tasks(self):
        """Remove completed and failed tasks from memory"""
        with self.lock:
            to_remove = [tid for tid, task in self.tasks.items() 
                        if task.status in ["completed", "failed"]]
            for tid in to_remove:
                del self.tasks[tid]
                
    def _generate_task_id(self) -> str:
        """Generate a unique task ID"""
        with self._counter_lock:
            self._task_counter += 1
            counter = self._task_counter
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"task_{timestamp}_{counter:06d}"
        
    def _process_queue(self):
        """Main processing loop (runs in background thread)"""
        print("Detection queue processing started")
        
        while not self.stop_requested:
            try:
                # Get task from queue with timeout
                try:
                    task = self.queue.get(timeout=1)
                except queue.Empty:
                    if self.queue.empty() and self.on_queue_empty:
                        self.on_queue_empty()
                    continue
                    
                # Process the task
                self.current_task = task
                task.status = "processing"
                task.started_at = datetime.now()
                
                if self.on_task_start:
                    self.on_task_start(task)
                    
                try:
                    # Run detection
                    start_time = time.time()
                    result = self.detector.detect(task.image_path)
                    processing_time = time.time() - start_time
                    
                    # Update task
                    task.result = result
                    task.status = "completed"
                    task.completed_at = datetime.now()
                    
                    # Update statistics
                    with self.lock:
                        self.total_processed += 1
                        self.processing_times.append(processing_time)
                        if len(self.processing_times) > 100:
                            self.processing_times.pop(0)
                    
                    if self.on_task_complete:
                        self.on_task_complete(task)
                        
                except Exception as e:
                    # Handle error
                    import traceback
                    task.status = "failed"
                    task.error = f"{str(e)}\n{traceback.format_exc()}"
                    task.completed_at = datetime.now()
                    
                    with self.lock:
                        self.total_failed += 1
                    
                    if self.on_task_error:
                        self.on_task_error(task, e)
                    
                    print(f"Error processing task {task.task_id}: {e}")
                    print(traceback.format_exc())
                    
                finally:
                    self.queue.task_done()
                    self.current_task = None
                    
            except Exception as e:
                print(f"Unexpected error in processing loop: {e}")
                
        self.is_running = False
        print("Detection queue processing stopped")
        
    def save_results(self, output_file: str):
        """
        Save all task results to a JSON file
        
        Args:
            output_file: Path to output JSON file
        """
        with self.lock:
            results = [task.to_dict() for task in self.tasks.values()]
            
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Saved {len(results)} task results to {output_file}")


class PriorityDetectionQueue(DetectionQueue):
    """
    Priority queue for object detection tasks
    Tasks with higher priority are processed first
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """Initialize priority queue"""
        super().__init__(max_queue_size)
        self.queue = queue.PriorityQueue(maxsize=max_queue_size)
        
    def add_task(self, image_path: str, task_id: Optional[str] = None,
                 metadata: Optional[Dict] = None, priority: int = 0) -> str:
        """
        Add a detection task with priority
        
        Args:
            image_path: Path to the image
            task_id: Optional unique ID
            metadata: Optional metadata
            priority: Priority level (lower number = higher priority)
            
        Returns:
            task_id: The ID of the added task
        """
        if task_id is None:
            task_id = self._generate_task_id()
            
        task = DetectionTask(task_id, image_path, metadata)
        task.metadata['priority'] = priority
        
        with self.lock:
            self.tasks[task_id] = task
            
        try:
            # PriorityQueue uses (priority, item) tuples
            self.queue.put((priority, task), block=False)
            return task_id
        except queue.Full:
            with self.lock:
                del self.tasks[task_id]
            raise RuntimeError("Queue is full. Cannot add more tasks.")
            
    def _process_queue(self):
        """Override to handle priority queue format"""
        print("Priority detection queue processing started")
        
        while not self.stop_requested:
            try:
                try:
                    priority, task = self.queue.get(timeout=1)
                except queue.Empty:
                    if self.queue.empty() and self.on_queue_empty:
                        self.on_queue_empty()
                    continue
                    
                # Process the task (same as parent class)
                self.current_task = task
                task.status = "processing"
                task.started_at = datetime.now()
                
                if self.on_task_start:
                    self.on_task_start(task)
                    
                try:
                    start_time = time.time()
                    result = self.detector.detect(task.image_path)
                    processing_time = time.time() - start_time
                    
                    task.result = result
                    task.status = "completed"
                    task.completed_at = datetime.now()
                    
                    with self.lock:
                        self.total_processed += 1
                        self.processing_times.append(processing_time)
                        if len(self.processing_times) > 100:
                            self.processing_times.pop(0)
                    
                    if self.on_task_complete:
                        self.on_task_complete(task)
                        
                except Exception as e:
                    import traceback
                    task.status = "failed"
                    task.error = f"{str(e)}\n{traceback.format_exc()}"
                    task.completed_at = datetime.now()
                    
                    with self.lock:
                        self.total_failed += 1
                    
                    if self.on_task_error:
                        self.on_task_error(task, e)
                    
                    print(f"Error processing task {task.task_id}: {e}")
                    print(traceback.format_exc())
                    
                finally:
                    self.queue.task_done()
                    self.current_task = None
                    
            except Exception as e:
                print(f"Unexpected error in processing loop: {e}")
                
        self.is_running = False
        print("Priority detection queue processing stopped")
