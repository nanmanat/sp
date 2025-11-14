"""
Run the object detection models example
This script demonstrates the available detection models and their configurations
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the example
from examples.detection_models_example import main

if __name__ == "__main__":
    main()
