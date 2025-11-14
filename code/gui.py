import os
import sys
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import torch

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from code.utils.run_logger import CSVRunLogger
from code.train import run_training

class ExperimentQueue:
    """Manages a queue of experiments to be run sequentially"""

    def __init__(self):
        self.queue = queue.Queue()
        self.current_experiment = None
        self.is_running = False
        self.thread = None
        self.stop_requested = False

    def add_experiment(self, experiment_config):
        """Add an experiment to the queue"""
        self.queue.put(experiment_config)
        return self.queue.qsize()

    def start_processing(self, callback=None):
        """Start processing the queue in a separate thread"""
        if self.is_running:
            return False

        self.stop_requested = False
        self.is_running = True
        self.thread = threading.Thread(target=self._process_queue, args=(callback,))
        self.thread.daemon = True
        self.thread.start()
        return True

    def stop_processing(self):
        """Request to stop processing after the current experiment"""
        self.stop_requested = True
        return True

    def _process_queue(self, callback=None):
        """Process experiments from the queue until empty or stopped"""
        while not self.queue.empty() and not self.stop_requested:
            try:
                self.current_experiment = self.queue.get()
                if callback:
                    callback("start", self.current_experiment)

                # Check experiment type
                experiment_type = self.current_experiment.get('type', 'classification')
                
                if experiment_type == 'detection':
                    # Run object detection training
                    from code.train_detection import run_detection_training
                    
                    run_detection_training(
                        model_name=self.current_experiment.get('model', 'faster_rcnn_r50_fpn'),
                        num_classes=self.current_experiment.get('num_classes', 91),
                        batch_size=self.current_experiment.get('batch_size', 4),
                        epochs=self.current_experiment.get('epochs', 50),
                        lr=self.current_experiment.get('lr', 0.001),
                        num_workers=self.current_experiment.get('workers', 4),
                        data_path=self.current_experiment.get('data_path', None),
                        dataset_name=self.current_experiment.get('dataset_name', None)
                    )
                else:
                    # Run classification training (original behavior)
                    model_name = self.current_experiment.get('model', 'resnet50')
                    cross_val_lists = self.current_experiment.get('folds', ['0'])
                    batch_size = self.current_experiment.get('batch_size', None)
                    lr = self.current_experiment.get('lr', None)
                    num_workers = self.current_experiment.get('workers', None)
                    data_path = self.current_experiment.get('data_path', None)

                    # Call the training function with all parameters
                    early_stop_patience = self.current_experiment.get('early_stop_patience', None)
                    improvement_threshold = self.current_experiment.get('improvement_threshold', 0.015)
                    run_training(
                        model_name=model_name, 
                        cross_val_lists=cross_val_lists,
                        batch_size=batch_size,
                        lr=lr,
                        num_workers=num_workers,
                        data_path=data_path,
                        early_stop_patience=early_stop_patience,
                        improvement_threshold=improvement_threshold
                    )

                if callback:
                    callback("complete", self.current_experiment)

                self.queue.task_done()
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                print(f"Error in experiment: {error_msg}")
                if callback:
                    callback("error", self.current_experiment, error_msg)

        self.is_running = False
        self.current_experiment = None
        if callback:
            callback("queue_empty" if not self.stop_requested else "stopped")

class ExperimentGUI:
    """GUI for configuring and running experiments"""

    def __init__(self, root):
        self.root = root
        self.root.title("Acne Image Grading Experiment Runner")
        self.root.geometry("1000x800")

        # Create experiment queue
        self.experiment_queue = ExperimentQueue()

        # Create the main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create the notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.config_tab = ttk.Frame(self.notebook)
        self.detection_tab = ttk.Frame(self.notebook)
        self.queue_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.config_tab, text="Classification")
        self.notebook.add(self.detection_tab, text="Object Detection")
        self.notebook.add(self.queue_tab, text="Queue")
        self.notebook.add(self.log_tab, text="Log")

        # Setup each tab
        self._setup_config_tab()
        self._setup_detection_tab()
        self._setup_queue_tab()
        self._setup_log_tab()

        # Initialize log redirection
        self._setup_log_redirection()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_config_tab(self):
        """Setup the configuration tab"""
        # Create frames
        config_frame = ttk.LabelFrame(self.config_tab, text="Experiment Configuration", padding="10")
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Model selection
        ttk.Label(config_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="resnet50")
        model_options = [
            'vgg16_bn', 'resnet50', 'efficientnet_v2_s', 'convnext_tiny',
            'densenet121', 'regnet_y_8gf', 'mobilenet_v3_large',
            'vit_small_patch16_224', 'swin_tiny_patch4_window7_224', 'deit_small_patch16_224'
        ]
        model_dropdown = ttk.Combobox(config_frame, textvariable=self.model_var, values=model_options, state="readonly")
        model_dropdown.grid(row=0, column=1, sticky=tk.W, pady=5)

        # Cross-validation folds
        ttk.Label(config_frame, text="Folds:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.folds_frame = ttk.Frame(config_frame)
        self.folds_frame.grid(row=1, column=1, sticky=tk.W, pady=5)

        self.fold_vars = []
        for i in range(5):
            var = tk.BooleanVar(value=True if i == 0 else False)
            self.fold_vars.append(var)
            ttk.Checkbutton(self.folds_frame, text=str(i), variable=var).pack(side=tk.LEFT, padx=5)

        # Batch size
        ttk.Label(config_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(config_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)

        # Learning rate
        ttk.Label(config_frame, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(config_frame, textvariable=self.lr_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=5)

        # Number of workers
        ttk.Label(config_frame, text="Num Workers:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.workers_var = tk.StringVar(value="12")
        ttk.Entry(config_frame, textvariable=self.workers_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=5)

        # Data path
        ttk.Label(config_frame, text="Data Path:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.data_path_var = tk.StringVar(value="./code/Classification/JPEGImages")
        data_path_entry = ttk.Entry(config_frame, textvariable=self.data_path_var, width=40)
        data_path_entry.grid(row=5, column=1, sticky=tk.W, pady=5)
        ttk.Button(config_frame, text="Browse...", command=self._browse_data_path).grid(row=5, column=2, sticky=tk.W, pady=5)

        # Early stopping patience
        ttk.Label(config_frame, text="Early Stop Patience:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.early_stop_var = tk.StringVar(value="0")
        ttk.Entry(config_frame, textvariable=self.early_stop_var, width=10).grid(row=6, column=1, sticky=tk.W, pady=5)
        ttk.Label(config_frame, text="(0 to disable)").grid(row=6, column=2, sticky=tk.W, pady=5)

        # Improvement threshold
        ttk.Label(config_frame, text="Improvement Threshold:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.improvement_threshold_var = tk.StringVar(value="0.015")
        ttk.Entry(config_frame, textvariable=self.improvement_threshold_var, width=10).grid(row=7, column=1, sticky=tk.W, pady=5)
        ttk.Label(config_frame, text="(threshold for early stopping)").grid(row=7, column=2, sticky=tk.W, pady=5)

        # Add to queue button
        ttk.Button(config_frame, text="Add to Queue", command=self._add_to_queue).grid(row=8, column=0, columnspan=3, pady=20)

    def _setup_detection_tab(self):
        """Setup the object detection tab"""
        # Create frames
        detection_frame = ttk.LabelFrame(self.detection_tab, text="Object Detection Configuration", padding="10")
        detection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Model selection
        ttk.Label(detection_frame, text="Detection Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.detection_model_var = tk.StringVar(value="faster_rcnn_r50_fpn")
        
        detection_models = [
            'ssd300',
            'efficientdet_d0',
            'efficientdet_d1',
            'efficientdet_d2',
            'efficientdet_d3',
            'retinanet_r50_fpn',
            'faster_rcnn_r50_fpn',
            'cascade_rcnn_r50_fpn',
            'fcos_r50_fpn',
            'atss_r50_fpn',
            'centernet_hourglass104',
            'detr_r50',
            'deformable_detr_r50',
            'rt_detr_r50'
        ]
        
        detection_dropdown = ttk.Combobox(
            detection_frame, 
            textvariable=self.detection_model_var, 
            values=detection_models, 
            state="readonly",
            width=30
        )
        detection_dropdown.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Model info button
        ttk.Button(
            detection_frame, 
            text="Model Info", 
            command=self._show_detection_model_info
        ).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

        # Number of classes
        ttk.Label(detection_frame, text="Number of Classes:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.detection_num_classes_var = tk.StringVar(value="91")
        ttk.Entry(detection_frame, textvariable=self.detection_num_classes_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(detection_frame, text="(e.g., 91 for COCO, 20 for VOC)").grid(row=1, column=2, sticky=tk.W, pady=5)

        # Batch size
        ttk.Label(detection_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.detection_batch_size_var = tk.StringVar(value="4")
        ttk.Entry(detection_frame, textvariable=self.detection_batch_size_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Label(detection_frame, text="(smaller for detection models)").grid(row=2, column=2, sticky=tk.W, pady=5)

        # Learning rate
        ttk.Label(detection_frame, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.detection_lr_var = tk.StringVar(value="0.001")
        ttk.Entry(detection_frame, textvariable=self.detection_lr_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=5)

        # Epochs
        ttk.Label(detection_frame, text="Epochs:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.detection_epochs_var = tk.StringVar(value="50")
        ttk.Entry(detection_frame, textvariable=self.detection_epochs_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=5)

        # Number of workers
        ttk.Label(detection_frame, text="Num Workers:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.detection_workers_var = tk.StringVar(value="4")
        ttk.Entry(detection_frame, textvariable=self.detection_workers_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=5)

        # Data path
        ttk.Label(detection_frame, text="Dataset Path:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.detection_data_path_var = tk.StringVar(value="./datasets/coco")
        detection_data_entry = ttk.Entry(detection_frame, textvariable=self.detection_data_path_var, width=40)
        detection_data_entry.grid(row=6, column=1, sticky=tk.W, pady=5)
        ttk.Button(
            detection_frame, 
            text="Browse...", 
            command=self._browse_detection_data_path
        ).grid(row=6, column=2, sticky=tk.W, pady=5)

        # Information section
        info_frame = ttk.LabelFrame(detection_frame, text="Information", padding="10")
        info_frame.grid(row=7, column=0, columnspan=3, sticky=tk.EW, pady=15)
        
        info_text = (
            "Object Detection Models:\n\n"
            "• One-Stage Detectors: SSD, EfficientDet, RetinaNet, FCOS, ATSS\n"
            "  - Faster inference, good for real-time applications\n\n"
            "• Two-Stage Detectors: Faster R-CNN, Cascade R-CNN\n"
            "  - Higher accuracy, slower inference\n\n"
            "• Transformer-Based: DETR, Deformable DETR, RT-DETR\n"
            "  - Modern architecture, set-based predictions\n\n"
            "• Anchor-Free: CenterNet, FCOS\n"
            "  - Simplified architecture without anchor boxes\n\n"
            "Note: Some models require additional packages (effdet, transformers, mmdet)"
        )
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack()

        # Add to queue button
        ttk.Button(
            detection_frame, 
            text="Add Detection Training to Queue", 
            command=self._add_detection_to_queue
        ).grid(row=8, column=0, columnspan=3, pady=20)

        # Quick start button
        ttk.Button(
            detection_frame, 
            text="Start Detection Training Now", 
            command=self._start_detection_training
        ).grid(row=9, column=0, columnspan=3, pady=10)

    def _setup_queue_tab(self):
        """Setup the queue tab"""
        # Create frames
        queue_frame = ttk.LabelFrame(self.queue_tab, text="Experiment Queue", padding="10")
        queue_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Queue list
        self.queue_listbox = tk.Listbox(queue_frame, height=15)
        self.queue_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        # Control buttons
        button_frame = ttk.Frame(queue_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.start_button = ttk.Button(button_frame, text="Start Queue", command=self._start_queue)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop After Current", command=self._stop_queue, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = ttk.Button(button_frame, text="Clear Queue", command=self._clear_queue)
        self.clear_button.pack(side=tk.LEFT, padx=5)

    def _setup_log_tab(self):
        """Setup the log tab"""
        # Create frames
        log_frame = ttk.LabelFrame(self.log_tab, text="Training Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text.config(state=tk.DISABLED)

        # Control buttons
        button_frame = ttk.Frame(log_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Log", command=self._save_log).pack(side=tk.LEFT, padx=5)

    def _setup_log_redirection(self):
        """Redirect stdout to the log text widget"""
        class TextRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
                self.buffer = ""

            def write(self, string):
                self.buffer += string
                if '\n' in self.buffer:
                    lines = self.buffer.split('\n')
                    self.buffer = lines[-1]  # Keep the last incomplete line

                    # Write complete lines to the text widget
                    for line in lines[:-1]:
                        self.text_widget.config(state=tk.NORMAL)
                        self.text_widget.insert(tk.END, line + '\n')
                        self.text_widget.see(tk.END)
                        self.text_widget.config(state=tk.DISABLED)

            def flush(self):
                if self.buffer:
                    self.text_widget.config(state=tk.NORMAL)
                    self.text_widget.insert(tk.END, self.buffer)
                    self.text_widget.see(tk.END)
                    self.text_widget.config(state=tk.DISABLED)
                    self.buffer = ""

        # Redirect stdout to the log text widget
        sys.stdout = TextRedirector(self.log_text)

    def _browse_data_path(self):
        """Open a file dialog to select the data path"""
        path = filedialog.askdirectory(initialdir=os.path.dirname(self.data_path_var.get()))
        if path:
            self.data_path_var.set(path)

    def _browse_detection_data_path(self):
        """Open a file dialog to select the detection dataset path"""
        initial_dir = os.path.dirname(self.detection_data_path_var.get()) if os.path.exists(self.detection_data_path_var.get()) else "./"
        path = filedialog.askdirectory(initialdir=initial_dir)
        if path:
            self.detection_data_path_var.set(path)

    def _show_detection_model_info(self):
        """Show information about the selected detection model"""
        try:
            from code.object_detection_models import get_model_info
            model_name = self.detection_model_var.get()
            info = get_model_info(model_name)
            
            info_message = (
                f"Model: {model_name}\n\n"
                f"Description: {info.get('description', 'N/A')}\n\n"
                f"Framework: {info.get('framework', 'N/A')}\n\n"
                f"Requirements: {', '.join(info.get('requirements', []))}\n\n"
                f"Notes: {info.get('notes', 'N/A')}"
            )
            
            messagebox.showinfo("Model Information", info_message)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model info: {str(e)}")

    def _add_detection_to_queue(self):
        """Add detection training configuration to the queue"""
        try:
            config = {
                'type': 'detection',
                'model': self.detection_model_var.get(),
                'num_classes': int(self.detection_num_classes_var.get()),
                'batch_size': int(self.detection_batch_size_var.get()),
                'lr': float(self.detection_lr_var.get()),
                'epochs': int(self.detection_epochs_var.get()),
                'workers': int(self.detection_workers_var.get()),
                'data_path': self.detection_data_path_var.get()
            }

            # Add to queue
            position = self.experiment_queue.add_experiment(config)

            # Update queue listbox
            self.queue_listbox.insert(
                tk.END, 
                f"{position}. [DETECTION] Model: {config['model']}, Classes: {config['num_classes']}"
            )

            # Update status
            self.status_var.set(f"Added detection experiment to queue. Queue size: {position}")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input values: {str(e)}")

    def _start_detection_training(self):
        """Start detection training immediately (not queued)"""
        try:
            from code.train_detection import run_detection_training
            
            # Show confirmation dialog
            result = messagebox.askyesno(
                "Start Detection Training",
                f"Start training {self.detection_model_var.get()} now?\n\n"
                "This will run in the background and log output to the Log tab."
            )
            
            if result:
                # Run in a separate thread
                def train_thread():
                    try:
                        run_detection_training(
                            model_name=self.detection_model_var.get(),
                            num_classes=int(self.detection_num_classes_var.get()),
                            batch_size=int(self.detection_batch_size_var.get()),
                            epochs=int(self.detection_epochs_var.get()),
                            lr=float(self.detection_lr_var.get()),
                            num_workers=int(self.detection_workers_var.get()),
                            data_path=self.detection_data_path_var.get(),
                            dataset_name=None  # Auto-detect from data_path
                        )
                        print("\n✓ Detection training completed!")
                    except Exception as e:
                        print(f"\n✗ Detection training error: {str(e)}")
                
                thread = threading.Thread(target=train_thread)
                thread.daemon = True
                thread.start()
                
                self.status_var.set(f"Detection training started: {self.detection_model_var.get()}")
        except ImportError as e:
            messagebox.showerror(
                "Import Error", 
                f"Could not import detection training module:\n{str(e)}\n\n"
                "Make sure train_detection.py is available."
            )
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input values: {str(e)}")

    def _add_to_queue(self):
        """Add the current configuration to the queue"""
        # Get selected folds
        selected_folds = [str(i) for i, var in enumerate(self.fold_vars) if var.get()]
        if not selected_folds:
            messagebox.showerror("Error", "Please select at least one fold")
            return

        # Create experiment config
        early_stop = int(self.early_stop_var.get())
        config = {
            'model': self.model_var.get(),
            'folds': selected_folds,
            'batch_size': int(self.batch_size_var.get()),
            'lr': float(self.lr_var.get()),
            'workers': int(self.workers_var.get()),
            'data_path': self.data_path_var.get(),
            'early_stop_patience': early_stop if early_stop > 0 else None,
            'improvement_threshold': float(self.improvement_threshold_var.get())
        }

        # Add to queue
        position = self.experiment_queue.add_experiment(config)

        # Update queue listbox
        self.queue_listbox.insert(tk.END, f"{position}. Model: {config['model']}, Folds: {','.join(config['folds'])}")

        # Update status
        self.status_var.set(f"Added experiment to queue. Queue size: {position}")

    def _start_queue(self):
        """Start processing the queue"""
        if self.experiment_queue.start_processing(callback=self._queue_callback):
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Queue processing started")

    def _stop_queue(self):
        """Stop processing the queue after the current experiment"""
        if self.experiment_queue.stop_processing():
            self.status_var.set("Queue will stop after current experiment")

    def _clear_queue(self):
        """Clear the queue"""
        if not self.experiment_queue.is_running:
            # Create a new queue
            self.experiment_queue = ExperimentQueue()
            self.queue_listbox.delete(0, tk.END)
            self.status_var.set("Queue cleared")
        else:
            messagebox.showerror("Error", "Cannot clear queue while it's running")

    def _clear_log(self):
        """Clear the log text"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _save_log(self):
        """Save the log text to a file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.status_var.set(f"Log saved to {file_path}")

    def _queue_callback(self, status, experiment=None, error=None):
        """Callback for queue processing events"""
        if status == "start":
            # Build status message based on experiment type
            experiment_type = experiment.get('type', 'classification')
            model_name = experiment.get('model', 'Unknown')
            
            if experiment_type == 'detection':
                # Detection experiments don't have folds
                num_classes = experiment.get('num_classes', 91)
                self.status_var.set(f"Running detection: Model {model_name}, Classes {num_classes}")
            else:
                # Classification experiments have folds
                folds = experiment.get('folds', ['0'])
                self.status_var.set(f"Running experiment: Model {model_name}, Folds {','.join(folds)}")
            
            # Highlight the current experiment in the queue
            for i in range(self.queue_listbox.size()):
                if self.queue_listbox.get(i).startswith("1."):  # First item
                    self.queue_listbox.itemconfig(i, {'bg': 'light green'})

        elif status == "complete":
            # Remove the completed experiment from the queue listbox
            if self.queue_listbox.size() > 0:
                self.queue_listbox.delete(0)
                # Renumber remaining items
                for i in range(self.queue_listbox.size()):
                    item = self.queue_listbox.get(i)
                    new_item = f"{i+1}.{item[item.find('.'):]}"
                    self.queue_listbox.delete(i)
                    self.queue_listbox.insert(i, new_item)

        elif status == "error":
            self.status_var.set(f"Error in experiment: {error}")
            messagebox.showerror("Experiment Error", f"Error running experiment: {error}")

        elif status == "queue_empty":
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Queue processing completed")

        elif status == "stopped":
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Queue processing stopped")

def main():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("Warning: CUDA is not available. Training will be slow.")

    # Create the GUI
    root = tk.Tk()
    app = ExperimentGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
