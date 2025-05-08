#!/usr/bin/env python

"""
Simple test script to verify FLORIS import from local installation.
"""

import os
import sys

print("Python executable:", sys.executable)
print("Current working directory:", os.getcwd())
print("Script directory:", os.path.dirname(os.path.abspath(__file__)))
print("Python path:", sys.path)

# Try different import approaches
print("\nAttempt 1: Standard import")
try:
    import floris
    print(f"Success! FLORIS imported from {os.path.dirname(floris.__file__)}")
except ImportError as e:
    print(f"Failed: {e}")

print("\nAttempt 2: Add parent directory to path")
try:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import floris
    print(f"Success! FLORIS imported from {os.path.dirname(floris.__file__)}")
except ImportError as e:
    print(f"Failed: {e}")
