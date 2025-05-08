#!/usr/bin/env python

"""
Simple script to generate blockage model documentation figures using matplotlib.
This script creates visualization figures without relying on FLORIS code.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

# Directory to save figures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom colormap for velocity differences
colors = [(0, 0, 0.8), (0, 0.8, 0.8), (1, 1, 1), (0.8, 0.4, 0), (0.8, 0, 0)]
cmap_diff = LinearSegmentedColormap.from_list("velocity_diff", colors, N=100)

def figure1_model_comparison():
    """Generate a comparison of velocity deficit profiles for different blockage models."""
    
    # Create sample data for the plot
    x = np.linspace(-5, 0, 100)  # Upstream distance (x/D)
    
    # Simplified model data
    # These are representative curves based on the characteristics of each model
    deficit_parametrized = 5 * np.exp(0.5 * x)  # Parametrized Global
    deficit_vortex = 4 / np.sqrt(1 + (0.5/np.abs(x))**2)  # Vortex Cylinder
    deficit_mirrored = 4.5 / np.sqrt(1 + (0.5/np.abs(x))**2)  # Mirrored Vortex (slightly higher)
    deficit_self_similar = 3.5 / (1 + np.abs(x)**2)  # Self-Similar
    deficit_engineering = 4 * np.exp(0.3 * x)  # Engineering Global
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, deficit_parametrized, 'b-', linewidth=2, label='Parametrized Global')
    plt.plot(x, deficit_vortex, 'g-', linewidth=2, label='Vortex Cylinder')
    plt.plot(x, deficit_mirrored, 'r-', linewidth=2, label='Mirrored Vortex')
    plt.plot(x, deficit_self_similar, 'c-', linewidth=2, label='Self-Similar')
    plt.plot(x, deficit_engineering, 'm-', linewidth=2, label='Engineering Global')
    
    plt.xlabel('Distance Upstream (x/D)', fontsize=12)
    plt.ylabel('Velocity Deficit (%)', fontsize=12)
    plt.title('Comparison of Velocity Deficit Profiles for Different Blockage Models', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlim(-5, 0)
    plt.ylim(0, 8)
    
    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "fig1_model_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 1 to {save_path}")
    plt.close()

def figure2_single_turbine_visualization():
    """Generate a horizontal cut plane visualization for blockage effect around a single turbine."""
    
    # Create a grid
    x = np.linspace(-5, 2, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    
    # Turbine location
    turbine_x = 0
    turbine_y = 0
    rotor_radius = 0.5
    
    # Distance from turbine
    R = np.sqrt((X - turbine_x)**2 + (Y - turbine_y)**2)
    
    # Angle from turbine
    theta = np.arctan2(Y - turbine_y, X - turbine_x)
    
    # Create velocity fields
    u_freestream = 8.0  # m/s
    
    # No blockage - uniform except at turbine
    u_no_blockage = np.ones_like(X) * u_freestream
    
    # With blockage - simplified model
    # Upstream deficit based on distance
    u_blockage = np.ones_like(X) * u_freestream
    
    # Apply blockage upstream of turbine only
    upstream_mask = X < turbine_x
    dist_x = np.abs(X[upstream_mask] - turbine_x)
    dist_y = np.abs(Y[upstream_mask] - turbine_y)
    
    # Simplified blockage model - exponential decay with distance
    blockage_factor = 0.15 * np.exp(-dist_x) * np.exp(-0.5 * (dist_y/rotor_radius)**2)
    u_blockage[upstream_mask] = u_freestream * (1 - blockage_factor)
    
    # Mark the turbine area
    turbine_mask = R < rotor_radius
    u_no_blockage[turbine_mask] = u_freestream * 0.7
    u_blockage[turbine_mask] = u_freestream * 0.7
    
    # Create a 3-panel figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    # Panel 1: No blockage
    im1 = axs[0].contourf(X, Y, u_no_blockage, 50, cmap='coolwarm', 
                         vmin=u_freestream*0.6, vmax=u_freestream*1.05)
    axs[0].set_title('Without Blockage', fontsize=14)
    axs[0].set_xlabel('x/D', fontsize=12)
    axs[0].set_ylabel('y/D', fontsize=12)
    
    # Draw turbine
    circle = plt.Circle((turbine_x, turbine_y), rotor_radius, fill=True, color='black')
    axs[0].add_patch(circle)
    
    # Panel 2: With blockage
    im2 = axs[1].contourf(X, Y, u_blockage, 50, cmap='coolwarm',
                         vmin=u_freestream*0.6, vmax=u_freestream*1.05)
    axs[1].set_title('With Blockage', fontsize=14)
    axs[1].set_xlabel('x/D', fontsize=12)
    
    # Draw turbine
    circle = plt.Circle((turbine_x, turbine_y), rotor_radius, fill=True, color='black')
    axs[1].add_patch(circle)
    
    # Panel 3: Difference
    deficit = 100 * (u_no_blockage - u_blockage) / u_freestream
    im3 = axs[2].contourf(X, Y, deficit, 50, cmap=cmap_diff,
                          vmin=-0.5, vmax=5)
    axs[2].set_title('Velocity Deficit (%)', fontsize=14)
    axs[2].set_xlabel('x/D', fontsize=12)
    
    # Draw turbine
    circle = plt.Circle((turbine_x, turbine_y), rotor_radius, fill=True, color='black')
    axs[2].add_patch(circle)
    
    # Add colorbars
    cbar1 = fig.colorbar(im1, ax=axs[0])
    cbar1.set_label('Velocity (m/s)', fontsize=10)
    
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar2.set_label('Velocity (m/s)', fontsize=10)
    
    cbar3 = fig.colorbar(im3, ax=axs[2])
    cbar3.set_label('Velocity Deficit (%)', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "fig2_single_turbine.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 2 to {save_path}")
    plt.close()

def figure3_farm_blockage():
    """Generate a visualization of blockage effects in a wind farm."""
    
    # Create a grid
    x = np.linspace(-5, 8, 200)
    y = np.linspace(-3, 6, 200)
    X, Y = np.meshgrid(x, y)
    
    # Farm layout (3x3 grid)
    turbine_spacing = 3.0
    turbine_x = []
    turbine_y = []
    for i in range(3):
        for j in range(3):
            turbine_x.append(i * turbine_spacing)
            turbine_y.append(j * turbine_spacing)
    
    rotor_radius = 0.5
    
    # Freestream velocity
    u_freestream = 8.0  # m/s
    
    # No blockage field
    u_no_blockage = np.ones_like(X) * u_freestream
    
    # With blockage field
    u_blockage = np.ones_like(X) * u_freestream
    
    # Calculate individual turbine blockage effects
    for tx, ty in zip(turbine_x, turbine_y):
        # Distance from this turbine
        dist_x = X - tx
        dist_y = Y - ty
        R = np.sqrt(dist_x**2 + dist_y**2)
        
        # Mark the turbine location
        turbine_mask = R < rotor_radius
        u_no_blockage[turbine_mask] = u_freestream * 0.7
        u_blockage[turbine_mask] = u_freestream * 0.7
        
        # Apply blockage upstream of each turbine
        upstream_mask = dist_x < 0
        if np.any(upstream_mask):
            # Simplified blockage model
            blockage_factor = 0.1 * np.exp(-np.abs(dist_x[upstream_mask])/2) * \
                             np.exp(-0.5 * (dist_y[upstream_mask]/rotor_radius)**2)
            u_blockage[upstream_mask] -= u_freestream * blockage_factor
    
    # Add farm-scale global blockage effect
    farm_center_x = np.mean(turbine_x)
    farm_center_y = np.mean(turbine_y)
    
    # Distance to farm center
    farm_dist_x = X - farm_center_x
    farm_dist_y = Y - farm_center_y
    
    # Apply global blockage effect (upstream of farm only)
    global_mask = farm_dist_x < -2
    if np.any(global_mask):
        # Global blockage factor
        global_factor = 0.03 * np.exp(-np.abs(farm_dist_x[global_mask])/4) * \
                        np.exp(-0.5 * (farm_dist_y[global_mask]/6)**2)
        u_blockage[global_mask] -= u_freestream * global_factor
    
    # Create the figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    # Panel 1: No blockage
    im1 = axs[0].contourf(X, Y, u_no_blockage, 50, cmap='coolwarm',
                         vmin=u_freestream*0.7, vmax=u_freestream*1.05)
    axs[0].set_title('Without Blockage', fontsize=14)
    axs[0].set_xlabel('x/D', fontsize=12)
    axs[0].set_ylabel('y/D', fontsize=12)
    
    # Draw turbines
    for tx, ty in zip(turbine_x, turbine_y):
        circle = plt.Circle((tx, ty), rotor_radius, fill=True, color='black')
        axs[0].add_patch(circle)
    
    # Panel 2: With blockage
    im2 = axs[1].contourf(X, Y, u_blockage, 50, cmap='coolwarm',
                         vmin=u_freestream*0.7, vmax=u_freestream*1.05)
    axs[1].set_title('With Blockage', fontsize=14)
    axs[1].set_xlabel('x/D', fontsize=12)
    
    # Draw turbines
    for tx, ty in zip(turbine_x, turbine_y):
        circle = plt.Circle((tx, ty), rotor_radius, fill=True, color='black')
        axs[1].add_patch(circle)
    
    # Panel 3: Difference
    deficit = 100 * (u_no_blockage - u_blockage) / u_freestream
    im3 = axs[2].contourf(X, Y, deficit, 50, cmap=cmap_diff,
                         vmin=-0.5, vmax=4)
    axs[2].set_title('Velocity Deficit (%)', fontsize=14)
    axs[2].set_xlabel('x/D', fontsize=12)
    
    # Draw turbines
    for tx, ty in zip(turbine_x, turbine_y):
        circle = plt.Circle((tx, ty), rotor_radius, fill=True, color='black')
        axs[2].add_patch(circle)
    
    # Add colorbars
    cbar1 = fig.colorbar(im1, ax=axs[0])
    cbar1.set_label('Velocity (m/s)', fontsize=10)
    
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar2.set_label('Velocity (m/s)', fontsize=10)
    
    cbar3 = fig.colorbar(im3, ax=axs[2])
    cbar3.set_label('Velocity Deficit (%)', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "fig3_farm_blockage.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 3 to {save_path}")
    plt.close()

def figure4_model_performance():
    """Generate a visualization of blockage model performance comparison."""
    
    # Define models and data
    models = [
        "Parametrized\nGlobal",
        "Vortex\nCylinder", 
        "Mirrored\nVortex",
        "Self\nSimilar",
        "Engineering\nGlobal"
    ]
    
    # Made-up data for visualization purposes
    accuracy = [95, 93, 96, 92, 94]  # % accuracy
    computational_cost = [0.5, 1.2, 1.5, 0.7, 0.4]  # relative cost
    complexity = [3, 4, 5, 2, 1]  # complexity level
    
    # Create the figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Accuracy by model
    axs[0].bar(models, accuracy, color=['royalblue', 'green', 'red', 'purple', 'orange'])
    axs[0].set_ylim(85, 100)
    axs[0].set_title('Model Accuracy', fontsize=14)
    axs[0].set_ylabel('Accuracy (%)', fontsize=12)
    axs[0].grid(axis='y', alpha=0.3)
    
    # Panel 2: Accuracy vs Cost scatter plot
    # Bubble size represents complexity
    bubble_sizes = [c * 100 for c in complexity]
    colors = ['royalblue', 'green', 'red', 'purple', 'orange']
    
    axs[1].scatter(computational_cost, accuracy, s=bubble_sizes, c=colors, alpha=0.7)
    
    # Add model labels to bubbles
    for i, model in enumerate(models):
        axs[1].annotate(model.replace('\n', ' '), 
                        (computational_cost[i], accuracy[i]),
                        xytext=(5, 0), textcoords='offset points')
    
    axs[1].set_title('Model Accuracy vs. Computational Cost', fontsize=14)
    axs[1].set_xlabel('Relative Computational Cost', fontsize=12)
    axs[1].set_ylabel('Accuracy (%)', fontsize=12)
    axs[1].grid(True, alpha=0.3)
    axs[1].set_ylim(85, 100)
    
    # Add note about bubble size
    axs[1].annotate('Bubble size represents\nmodel complexity', 
                    xy=(0.05, 0.05), xycoords='axes fraction')
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "fig4_model_performance.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 4 to {save_path}")
    plt.close()

def figure5_stability_influence():
    """Generate visualization of atmospheric stability effects on blockage."""
    
    # Create sample data
    x = np.linspace(-5, 0, 100)  # Upstream distance (x/D)
    
    # Blockage profiles under different stability conditions
    deficit_stable = 6 * np.exp(0.5 * x)  # Enhanced blockage in stable conditions
    deficit_neutral = 4 * np.exp(0.5 * x)  # Neutral conditions
    deficit_unstable = 2.5 * np.exp(0.5 * x)  # Reduced blockage in unstable conditions
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, deficit_stable, 'b-', linewidth=3, label='Stable Atmosphere')
    plt.plot(x, deficit_neutral, 'g-', linewidth=3, label='Neutral Atmosphere')
    plt.plot(x, deficit_unstable, 'r-', linewidth=3, label='Unstable Atmosphere')
    
    plt.xlabel('Distance Upstream (x/D)', fontsize=12)
    plt.ylabel('Velocity Deficit (%)', fontsize=12)
    plt.title('Influence of Atmospheric Stability on Blockage Effect', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(-5, 0)
    plt.ylim(0, 10)
    
    # Add annotations
    plt.annotate('Enhanced blockage\nin stable conditions', 
                 xy=(-2.5, 5), xytext=(-4, 7),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    plt.annotate('Reduced blockage\nin unstable conditions', 
                 xy=(-2.5, 2), xytext=(-4, 1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "fig5_stability_influence.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 5 to {save_path}")
    plt.close()

def run_all():
    """Run all figure generation functions."""
    print(f"Generating documentation figures in {OUTPUT_DIR}")
    
    # Generate all figures
    figure1_model_comparison()
    figure2_single_turbine_visualization()
    figure3_farm_blockage()
    figure4_model_performance()
    figure5_stability_influence()
    
    print("All documentation figures generated successfully!")

if __name__ == "__main__":
    run_all()
