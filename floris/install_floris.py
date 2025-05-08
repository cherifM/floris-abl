#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Installation script for FLORIS with blockage models.

This script provides a simple way to install FLORIS in development mode,
allowing for immediate testing and validation of the blockage models.

MIT License

Copyright (c) 2025 Cherif Mihoubi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def install_floris(develop=True, force=False, clean=False):
    """
    Install FLORIS in development mode for immediate testing and validation.
    
    Args:
        develop (bool): Install in development mode if True, regular install otherwise
        force (bool): Force reinstallation even if already installed
        clean (bool): Clean build artifacts before installation
    
    Returns:
        bool: True if installation successful, False otherwise
    """
    # Get the root FLORIS directory (parent of the directory containing this script)
    script_dir = Path(__file__).resolve().parent
    floris_root = script_dir.parent
    
    print(f"FLORIS root directory: {floris_root}")
    
    if clean:
        print("Cleaning build artifacts...")
        clean_dirs = [
            floris_root / "build",
            floris_root / "dist",
            floris_root / "*.egg-info"
        ]
        
        for clean_dir in clean_dirs:
            if "*" in str(clean_dir):
                # Handle glob patterns
                import glob
                for path in glob.glob(str(clean_dir)):
                    if os.path.exists(path):
                        subprocess.run(["rm", "-rf", path])
            else:
                if clean_dir.exists():
                    subprocess.run(["rm", "-rf", str(clean_dir)])
    
    # Check if FLORIS is already installed
    try:
        import floris
        if not force:
            print(f"FLORIS is already installed at {floris.__file__}")
            print("Use --force to reinstall or --clean to clean and reinstall")
            return True
        else:
            print("Force reinstallation requested")
    except ImportError:
        print("FLORIS not found in current environment, proceeding with installation")
    
    # Install FLORIS
    os.chdir(str(floris_root))
    
    if develop:
        print("Installing FLORIS in development mode...")
        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    else:
        print("Installing FLORIS in regular mode...")
        cmd = [sys.executable, "-m", "pip", "install", "."]
    
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print("✅ FLORIS installation successful!")
            
            # Test import to verify installation
            try:
                import floris
                print(f"Installed FLORIS version: {floris.__version__}")
                print(f"Installation location: {floris.__file__}")
                return True
            except ImportError:
                print("❌ FLORIS was installed but cannot be imported.")
                return False
        else:
            print(f"❌ FLORIS installation failed with code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ FLORIS installation failed: {e}")
        return False

def main():
    """Main entry point for the installation script."""
    parser = argparse.ArgumentParser(description="Install FLORIS with blockage models")
    parser.add_argument("--regular", action="store_true", 
                       help="Install in regular mode instead of development mode")
    parser.add_argument("--force", action="store_true", 
                       help="Force reinstallation even if already installed")
    parser.add_argument("--clean", action="store_true", 
                       help="Clean build artifacts before installation")
    
    args = parser.parse_args()
    
    success = install_floris(develop=not args.regular, force=args.force, clean=args.clean)
    
    if success:
        print("\nInstallation complete. You can now run the validation scripts:")
        print("python floris/docs/blockage_models/run_validations.py")
    else:
        print("\nInstallation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
