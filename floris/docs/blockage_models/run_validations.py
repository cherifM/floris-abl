#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run all blockage model validation cases and generate plots.

This script executes all validation scripts for the FLORIS blockage models,
comparing them against reference data from the literature and generating
comprehensive validation plots.

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
import time
import argparse
import subprocess
from pathlib import Path


def run_validation_script(script_path, verbose=True):
    """
    Run a single validation script and capture its output.
    
    Args:
        script_path (Path): Path to the validation script
        verbose (bool): Whether to print script output
        
    Returns:
        bool: True if script ran successfully, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running validation: {script_path.name}")
    print(f"{'='*80}")
    
    # Run the script
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(result.stdout)
            if result.stderr:
                print("Errors/Warnings:")
                print(result.stderr)
        
        print(f"✅ Validation completed successfully in {elapsed_time:.2f} seconds.")
        return True
    
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Validation failed after {elapsed_time:.2f} seconds.")
        print(f"Return code: {e.returncode}")
        print("Output:")
        print(e.stdout)
        print("Error:")
        print(e.stderr)
        return False


def run_all_validations(verbose=True):
    """
    Run all blockage model validation scripts.
    
    Args:
        verbose (bool): Whether to print script output
        
    Returns:
        dict: Summary of results
    """
    # Get the directory containing this script
    script_dir = Path(__file__).resolve().parent
    
    # Create validation_images directory if it doesn't exist
    validation_images_dir = script_dir / "validation_images"
    validation_images_dir.mkdir(exist_ok=True)
    
    # Define all validation scripts
    validation_scripts = [
        script_dir / "validate_centerline_deficit.py",
        script_dir / "validate_lateral_profiles.py", 
        script_dir / "validate_ground_effect.py",
        script_dir / "validate_stability_effects.py"
    ]
    
    # Run each validation script
    results = {}
    success_count = 0
    
    for script in validation_scripts:
        if not script.exists():
            print(f"Warning: Validation script {script.name} not found.")
            results[script.name] = False
            continue
            
        success = run_validation_script(script, verbose=verbose)
        results[script.name] = success
        
        if success:
            success_count += 1
    
    # Print summary
    print("\n" + "="*40)
    print(f"Validation Summary: {success_count}/{len(validation_scripts)} successful")
    print("="*40)
    
    for script_name, success in results.items():
        status = "✅ Passed" if success else "❌ Failed"
        print(f"{script_name:40s} {status}")
    
    print("\nValidation plots saved to:")
    print(f"  {validation_images_dir}")
    
    return results


def main():
    """Main entry point for running validation scripts."""
    parser = argparse.ArgumentParser(description="Run blockage model validations")
    parser.add_argument("--quiet", action="store_true", 
                       help="Suppress detailed output from validation scripts")
    parser.add_argument("--script", type=str, 
                       help="Run specific validation script instead of all")
    
    args = parser.parse_args()
    
    if args.script:
        script_dir = Path(__file__).resolve().parent
        script_path = script_dir / args.script
        
        if not script_path.exists():
            print(f"Error: Validation script {args.script} not found.")
            return 1
            
        success = run_validation_script(script_path, verbose=not args.quiet)
        return 0 if success else 1
    else:
        results = run_all_validations(verbose=not args.quiet)
        return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
