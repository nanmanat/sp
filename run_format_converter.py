"""
Run the dataset format converter utility
This script demonstrates format conversion between COCO, Pascal VOC, and YOLO
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the converter examples
from examples.dataset_format_converter import *

if __name__ == "__main__":
    print("=" * 70)
    print("DATASET FORMAT CONVERTER - Examples and Usage")
    print("=" * 70)
    print()
    
    # Run the examples from the module
    import examples.dataset_format_converter as converter
    
    # Get the __main__ block from the module
    print("Running conversion examples...\n")
    exec(open('examples/dataset_format_converter.py').read())
