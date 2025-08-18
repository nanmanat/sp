#!/usr/bin/env python
"""
Launcher script for the Acne Image Grading Experiment Runner GUI.
"""

import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the GUI
from code.gui import main

if __name__ == "__main__":
    main()