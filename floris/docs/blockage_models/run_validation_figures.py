#!/usr/bin/env python

"""
Simple script to run the figure generation functions from generate_validation_figures.py
"""

import os
import sys
import time

# Add repository root to Python path if running the script directly
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

# Import the figure generation functions
from floris.docs.blockage_models.generate_validation_figures import (
    OUTPUT_DIR,
    generate_single_turbine_centerline,
    generate_single_turbine_horizontal_cut,
    generate_three_turbine_horizontal_cut,
    generate_small_farm_blockage,
    generate_large_farm_blockage,
    generate_hub_height_influence
)

def main():
    """Run all figure generation functions."""
    print(f"Generating validation figures to {OUTPUT_DIR}")
    
    # Figure 1: Single turbine centerline comparison
    print("Generating Figure 1: Single turbine centerline comparison...")
    save_path = os.path.join(OUTPUT_DIR, "fig1_single_turbine_centerline.png")
    generate_single_turbine_centerline(save_path)
    
    # Figure 2: Single turbine horizontal cut
    print("Generating Figure 2: Single turbine horizontal cut...")
    save_path = os.path.join(OUTPUT_DIR, "fig2_single_turbine_horizontal.png")
    generate_single_turbine_horizontal_cut(save_path)
    
    # Figure 3: Three turbine row
    print("Generating Figure 3: Three turbine row horizontal cut...")
    save_path = os.path.join(OUTPUT_DIR, "fig3_three_turbine_horizontal.png")
    generate_three_turbine_horizontal_cut(save_path)
    
    # Figure 4: Small farm (3x3) blockage
    print("Generating Figure 4: Small wind farm (3x3) blockage...")
    save_path = os.path.join(OUTPUT_DIR, "fig4_small_farm_blockage.png")
    generate_small_farm_blockage(save_path)
    
    # Figure 5: Large farm (10x10) blockage
    print("Generating Figure 5: Large wind farm (10x10) blockage...")
    save_path = os.path.join(OUTPUT_DIR, "fig5_large_farm_blockage.png")
    generate_large_farm_blockage(save_path)
    
    # Figure 6: Hub height influence
    print("Generating Figure 6: Hub height influence...")
    save_path = os.path.join(OUTPUT_DIR, "fig6_hub_height_influence.png")
    generate_hub_height_influence(save_path)
    
    print("All validation figures generated successfully.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
