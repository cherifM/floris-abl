#!/usr/bin/env python

"""
Test script to verify FLORIS import and basic functionality.
"""

import os
import sys
import importlib.util

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Python path:", sys.path)

# Try to import FLORIS package
try:
    import floris
    print("\nSuccessfully imported floris package")
    print("FLORIS version:", floris.__version__)
    print("FLORIS path:", floris.__path__)
    
    # Try to import floris.tools
    try:
        import floris.tools as wfct
        print("\nSuccessfully imported floris.tools")
        print("Tools path:", wfct.__file__)
    except ImportError as e:
        print("\nCouldn't import floris.tools:", e)
        
        # Check if tools module exists in the package
        print("\nChecking available modules in floris package:")
        print(dir(floris))
except ImportError as e:
    print("\nCouldn't import floris:", e)
    print("\nLooking for floris module in sys.path...")
    
    for path in sys.path:
        if not os.path.isdir(path):
            continue
        if 'floris' in os.listdir(path):
            print(f"Found floris in {path}")
            
            # Check what's in the floris directory
            floris_path = os.path.join(path, 'floris')
            if os.path.isdir(floris_path):
                print(f"Content of {floris_path}:")
                print(os.listdir(floris_path))
