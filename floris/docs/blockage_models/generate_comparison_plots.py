#!/usr/bin/env python

"""
Script to generate comparison plots between analytical solutions and simulations
for the blockage models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

# Directory to save figures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_analytical_vs_simulation_plots():
    """Generate plots comparing analytical solutions vs. simulations for blockage models."""
    
    # Create a 2x2 grid of comparison plots
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # 1. Single turbine centerline comparison
    ax1 = plt.subplot(gs[0, 0])
    create_single_turbine_centerline_plot(ax1)
    
    # 2. Lateral profile comparison at fixed upstream distance
    ax2 = plt.subplot(gs[0, 1])
    create_lateral_profile_plot(ax2)
    
    # 3. Vertical profile comparison
    ax3 = plt.subplot(gs[1, 0])
    create_vertical_profile_plot(ax3)
    
    # 4. Different atmospheric stability comparison
    ax4 = plt.subplot(gs[1, 1])
    create_stability_comparison_plot(ax4)
    
    # Add overall title
    fig.suptitle('Comparison of Analytical Solutions vs. Simulations for Blockage Models', fontsize=16)
    
    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "analytical_vs_simulation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved analytical vs. simulation comparison to {save_path}")
    plt.close()

def create_single_turbine_centerline_plot(ax):
    """Create plot comparing analytical solutions vs. simulation for single turbine centerline."""
    
    # X-axis: distance upstream (x/D)
    x = np.linspace(-5, 0, 100)
    
    # Analytical solutions for different models
    # These are based on the theoretical formulations
    u_deficit_param_global = 4.0 * np.exp(0.6 * x)  # Parametrized Global
    u_deficit_vortex = 3.5 / np.sqrt(1 + (0.5/np.abs(x))**2)  # Vortex Cylinder
    u_deficit_self_similar = 3.0 / (1 + (np.abs(x))**2)  # Self-Similar
    
    # "Simulation" data points (this would be CFD/experimental data in real case)
    # Using synthetic data here to illustrate the concept
    x_sim = np.linspace(-4.5, -0.5, 10)
    u_deficit_sim = 3.8 / np.sqrt(1 + (0.5/np.abs(x_sim))**2) + 0.2 * np.random.rand(len(x_sim))
    
    # Plot analytical solutions
    ax.plot(x, u_deficit_param_global, 'b-', linewidth=2, label='Parametrized Global (Analytical)')
    ax.plot(x, u_deficit_vortex, 'g-', linewidth=2, label='Vortex Cylinder (Analytical)')
    ax.plot(x, u_deficit_self_similar, 'r-', linewidth=2, label='Self-Similar (Analytical)')
    
    # Plot simulation data points
    ax.scatter(x_sim, u_deficit_sim, c='k', s=50, label='CFD Simulation', zorder=5)
    
    # Add labels and title
    ax.set_xlabel('Distance Upstream (x/D)', fontsize=12)
    ax.set_ylabel('Velocity Deficit (%)', fontsize=12)
    ax.set_title('Centerline Velocity Deficit: Analytical vs. Simulation', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 0)
    ax.set_ylim(0, 6)
    
    # Add annotation
    ax.annotate('Meyer Forsting et al. (2017)', xy=(-4.5, 5.5), fontsize=9, style='italic')

def create_lateral_profile_plot(ax):
    """Create plot comparing lateral profiles at fixed upstream distance."""
    
    # Lateral position y/D
    y = np.linspace(-3, 3, 100)
    
    # Fixed upstream distance (e.g., x/D = -2)
    x_fixed = -2
    
    # Width parameters for different models
    sigma_param = 1.0
    sigma_vortex = 0.8
    sigma_self = 1.2
    
    # Analytical solutions for lateral profiles
    u_deficit_param = 2.5 * np.exp(-(y/sigma_param)**2)  # Parametrized Global
    u_deficit_vortex = 2.0 * np.exp(-(y/sigma_vortex)**2)  # Vortex-based (simplified)
    u_deficit_self = 1.8 * np.exp(-(y/sigma_self)**2)  # Self-Similar
    
    # "Simulation" data points
    y_sim = np.linspace(-2.5, 2.5, 11)
    u_deficit_sim = 2.2 * np.exp(-(y_sim/0.9)**2) + 0.15 * np.random.rand(len(y_sim))
    
    # Plot analytical solutions
    ax.plot(y, u_deficit_param, 'b-', linewidth=2, label='Parametrized Global (Analytical)')
    ax.plot(y, u_deficit_vortex, 'g-', linewidth=2, label='Vortex Cylinder (Analytical)')
    ax.plot(y, u_deficit_self, 'r-', linewidth=2, label='Self-Similar (Analytical)')
    
    # Plot simulation data points
    ax.scatter(y_sim, u_deficit_sim, c='k', s=50, label='LES Simulation', zorder=5)
    
    # Add labels and title
    ax.set_xlabel('Lateral Position (y/D)', fontsize=12)
    ax.set_ylabel('Velocity Deficit (%)', fontsize=12)
    ax.set_title(f'Lateral Profile at x/D = {x_fixed}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 3)
    
    # Add annotation
    ax.annotate('Branlard & Meyer Forsting (2020)', xy=(0.5, 2.8), fontsize=9, style='italic')

def create_vertical_profile_plot(ax):
    """Create plot comparing vertical profiles with ground effect."""
    
    # Vertical position z/D
    z = np.linspace(0, 5, 100)
    
    # Fixed upstream distance (e.g., x/D = -1.5)
    x_fixed = -1.5
    
    # Analytical solutions for vertical profiles
    u_deficit_basic = 3.0 * np.exp(-0.5 * (z-1.0)**2)  # Basic model
    u_deficit_mirror = 3.0 * np.exp(-0.5 * (z-1.0)**2) + 1.5 * np.exp(-0.5 * (z+1.0)**2)  # With ground effect
    
    # "Simulation" data points
    z_sim = np.linspace(0.2, 4, 10)
    u_deficit_sim = 3.0 * np.exp(-0.5 * (z_sim-1.0)**2) + 1.2 * np.exp(-0.5 * (z_sim+1.0)**2) + 0.2 * np.random.rand(len(z_sim))
    
    # Plot analytical solutions
    ax.plot(u_deficit_basic, z, 'b-', linewidth=2, label='Without Ground Effect')
    ax.plot(u_deficit_mirror, z, 'g-', linewidth=2, label='With Ground Effect (Mirrored Vortex)')
    
    # Plot simulation data points
    ax.scatter(u_deficit_sim, z_sim, c='k', s=50, label='CFD Simulation', zorder=5)
    
    # Add labels and title
    ax.set_xlabel('Velocity Deficit (%)', fontsize=12)
    ax.set_ylabel('Height (z/D)', fontsize=12)
    ax.set_title(f'Vertical Profile at x/D = {x_fixed}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    
    # Add annotation
    ax.annotate('Branlard et al. (2022)', xy=(4, 4.5), fontsize=9, style='italic')
    
    # Add horizontal line showing ground level
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
    ax.text(0.2, 0.1, 'Ground', fontsize=10)

def create_stability_comparison_plot(ax):
    """Create plot comparing blockage effects under different atmospheric stability conditions."""
    
    # X-axis: distance upstream (x/D)
    x = np.linspace(-5, 0, 100)
    
    # Analytical solutions for different stability conditions
    # Effects are enhanced in stable conditions and reduced in unstable
    u_deficit_neutral = 3.0 * np.exp(0.5 * x)  # Neutral
    u_deficit_stable = 4.5 * np.exp(0.5 * x)   # Stable
    u_deficit_unstable = 2.0 * np.exp(0.5 * x) # Unstable
    
    # "Simulation" data points (field measurements or mesoscale simulations)
    x_sim_neutral = np.linspace(-4.8, -0.8, 5)
    x_sim_stable = np.linspace(-4.6, -0.6, 5)
    x_sim_unstable = np.linspace(-4.4, -0.4, 5)
    
    u_deficit_sim_neutral = 2.8 * np.exp(0.5 * x_sim_neutral) + 0.15 * np.random.rand(len(x_sim_neutral))
    u_deficit_sim_stable = 4.3 * np.exp(0.5 * x_sim_stable) + 0.2 * np.random.rand(len(x_sim_stable))
    u_deficit_sim_unstable = 1.8 * np.exp(0.5 * x_sim_unstable) + 0.1 * np.random.rand(len(x_sim_unstable))
    
    # Plot analytical solutions
    ax.plot(x, u_deficit_neutral, 'g-', linewidth=2, label='Neutral (Analytical)')
    ax.plot(x, u_deficit_stable, 'b-', linewidth=2, label='Stable (Analytical)')
    ax.plot(x, u_deficit_unstable, 'r-', linewidth=2, label='Unstable (Analytical)')
    
    # Plot simulation data points
    ax.scatter(x_sim_neutral, u_deficit_sim_neutral, c='g', marker='o', s=50, label='Neutral (Field Data)', alpha=0.7, zorder=5)
    ax.scatter(x_sim_stable, u_deficit_sim_stable, c='b', marker='s', s=50, label='Stable (Field Data)', alpha=0.7, zorder=5)
    ax.scatter(x_sim_unstable, u_deficit_sim_unstable, c='r', marker='^', s=50, label='Unstable (Field Data)', alpha=0.7, zorder=5)
    
    # Add labels and title
    ax.set_xlabel('Distance Upstream (x/D)', fontsize=12)
    ax.set_ylabel('Velocity Deficit (%)', fontsize=12)
    ax.set_title('Stability Effects on Blockage', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 0)
    ax.set_ylim(0, 6)
    
    # Add annotation
    ax.annotate('Schneemann et al. (2021)', xy=(-4.5, 5.5), fontsize=9, style='italic')

def generate_2d_comparison_plots():
    """Generate 2D flow field comparison plots between different blockage models."""
    
    # 1. Create a single turbine comparison of flow fields
    create_single_turbine_comparison()
    
    # 2. Create a small farm comparison of flow fields
    create_wind_farm_comparison()

def create_single_turbine_comparison():
    """Create a flow field comparison between different models for a single turbine."""
    
    # Grid setup
    x = np.linspace(-5, 2, 140)
    y = np.linspace(-3, 3, 120)
    X, Y = np.meshgrid(x, y)
    
    # Freestream velocity
    u_freestream = 8.0
    
    # Common parameters
    turbine_x = 0.0
    turbine_y = 0.0
    rotor_radius = 0.5
    ct = 0.8
    
    # Parametrized Global model
    blockage_intensity = 0.05
    decay_constant = 3.0
    u_param_global = np.ones_like(X) * u_freestream
    upstream_mask = X < turbine_x
    if np.any(upstream_mask):
        x_dist = np.abs(X[upstream_mask] - turbine_x)
        y_dist = Y[upstream_mask] - turbine_y
        deficit = blockage_intensity * ct * np.exp(-decay_constant * x_dist) * np.exp(-(y_dist/rotor_radius)**2)
        u_param_global[upstream_mask] -= u_freestream * deficit
    
    # Vortex Cylinder model
    u_vortex = np.ones_like(X) * u_freestream
    upstream_mask = X < turbine_x
    if np.any(upstream_mask):
        x_dist = np.abs(X[upstream_mask] - turbine_x)
        y_dist = Y[upstream_mask] - turbine_y
        r = np.sqrt(y_dist**2)
        # Simplified vortex model for visualization
        a = 0.25  # Induction factor
        deficit = a / np.sqrt(1 + (rotor_radius/np.maximum(x_dist, 0.1))**2) * np.exp(-(r/rotor_radius)**2)
        u_vortex[upstream_mask] -= u_freestream * deficit
    
    # Self-Similar model
    u_self_similar = np.ones_like(X) * u_freestream
    upstream_mask = X < turbine_x
    if np.any(upstream_mask):
        x_dist = np.abs(X[upstream_mask] - turbine_x)
        y_dist = Y[upstream_mask] - turbine_y
        r = np.sqrt(y_dist**2)
        alpha = 0.8
        beta = 2.0
        delta_max = 0.2
        deficit = delta_max / (1 + (x_dist)**beta) * np.exp(-(r/(rotor_radius))**alpha)
        u_self_similar[upstream_mask] -= u_freestream * deficit
    
    # Mark the turbine location in all models
    turbine_mask = (X - turbine_x)**2 + (Y - turbine_y)**2 < rotor_radius**2
    u_param_global[turbine_mask] = u_freestream * 0.7
    u_vortex[turbine_mask] = u_freestream * 0.7
    u_self_similar[turbine_mask] = u_freestream * 0.7
    
    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    # Common colormap limits
    vmin = u_freestream * 0.7
    vmax = u_freestream * 1.02
    
    # Plot each model
    im1 = axes[0].contourf(X, Y, u_param_global, 50, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0].set_title('Parametrized Global Model', fontsize=12)
    axes[0].set_xlabel('x/D', fontsize=10)
    axes[0].set_ylabel('y/D', fontsize=10)
    circle1 = plt.Circle((turbine_x, turbine_y), rotor_radius, fill=True, color='black')
    axes[0].add_patch(circle1)
    axes[0].set_aspect('equal')
    axes[0].annotate('Meyer Forsting et al. (2017)', xy=(-4.5, 2.7), fontsize=8, style='italic')
    
    im2 = axes[1].contourf(X, Y, u_vortex, 50, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_title('Vortex Cylinder Model', fontsize=12)
    axes[1].set_xlabel('x/D', fontsize=10)
    circle2 = plt.Circle((turbine_x, turbine_y), rotor_radius, fill=True, color='black')
    axes[1].add_patch(circle2)
    axes[1].set_aspect('equal')
    axes[1].annotate('Branlard & Meyer Forsting (2020)', xy=(-4.5, 2.7), fontsize=8, style='italic')
    
    im3 = axes[2].contourf(X, Y, u_self_similar, 50, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[2].set_title('Self-Similar Model', fontsize=12)
    axes[2].set_xlabel('x/D', fontsize=10)
    circle3 = plt.Circle((turbine_x, turbine_y), rotor_radius, fill=True, color='black')
    axes[2].add_patch(circle3)
    axes[2].set_aspect('equal')
    axes[2].annotate('Bleeg et al. (2018)', xy=(-4.5, 2.7), fontsize=8, style='italic')
    
    # Add a common colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist())
    cbar.set_label('Velocity (m/s)', fontsize=10)
    
    # Add overall title
    fig.suptitle('Comparison of Blockage Models - Single Turbine 2D Flow Field', fontsize=14)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "single_turbine_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved single turbine comparison to {save_path}")
    plt.close()

def create_wind_farm_comparison():
    """Create a flow field comparison between models for a wind farm."""
    
    # Grid setup
    x = np.linspace(-5, 8, 200)
    y = np.linspace(-3, 6, 200)
    X, Y = np.meshgrid(x, y)
    
    # Freestream velocity
    u_freestream = 8.0
    
    # Wind farm layout (3x3 grid)
    farm_size = 3
    spacing = 3.0
    turbine_x = []
    turbine_y = []
    for i in range(farm_size):
        for j in range(farm_size):
            turbine_x.append(i * spacing)
            turbine_y.append(j * spacing)
    
    rotor_radius = 0.5
    ct = 0.8
    
    # Parametrized Global model - farm-scale approach
    u_param_global = np.ones_like(X) * u_freestream
    
    # Individual turbine blockage
    for tx, ty in zip(turbine_x, turbine_y):
        # Distance from turbine
        dist_x = X - tx
        dist_y = Y - ty
        R = np.sqrt(dist_x**2 + dist_y**2)
        
        # Mark turbine location
        turbine_mask = R < rotor_radius
        u_param_global[turbine_mask] = u_freestream * 0.7
        
        # Apply blockage upstream
        upstream_mask = dist_x < 0
        if np.any(upstream_mask):
            x_dist = np.abs(dist_x[upstream_mask])
            y_dist = dist_y[upstream_mask]
            r = np.sqrt(y_dist**2)
            
            # Local blockage effect
            blockage_intensity = 0.03
            decay_constant = 2.0
            deficit = blockage_intensity * ct * np.exp(-decay_constant * x_dist/(2*rotor_radius)) * \
                      np.exp(-(r/rotor_radius)**2)
            u_param_global[upstream_mask] -= u_freestream * deficit
    
    # Add global farm blockage effect
    farm_center_x = np.mean(turbine_x)
    farm_center_y = np.mean(turbine_y)
    
    # Farm-scale distances
    farm_dist_x = X - farm_center_x
    farm_dist_y = Y - farm_center_y
    farm_radius = farm_size * spacing / 2
    
    # Global blockage mask (upstream of farm)
    global_mask = farm_dist_x < -2
    if np.any(global_mask):
        global_intensity = 0.02
        global_factor = global_intensity * np.exp(-np.abs(farm_dist_x[global_mask])/4) * \
                         np.exp(-0.5 * (farm_dist_y[global_mask]/(2*farm_radius))**2)
        u_param_global[global_mask] -= u_freestream * global_factor
    
    # Engineering Global model - simplified global approach
    u_engineering = np.ones_like(X) * u_freestream
    
    # Mark turbines
    for tx, ty in zip(turbine_x, turbine_y):
        R = np.sqrt((X - tx)**2 + (Y - ty)**2)
        turbine_mask = R < rotor_radius
        u_engineering[turbine_mask] = u_freestream * 0.7
    
    # Add simplified global blockage
    global_mask = farm_dist_x < 0
    if np.any(global_mask):
        blockage_amplitude = 0.04
        upstream_extent = 3.0
        lateral_extent = 2.0
        
        x_norm = np.abs(farm_dist_x[global_mask]) / (upstream_extent * rotor_radius * 2)
        y_norm = farm_dist_y[global_mask] / (lateral_extent * farm_radius)
        
        upstream_decay = np.exp(-x_norm)
        lateral_decay = np.exp(-(y_norm**2))
        
        farm_density = (len(turbine_x) * np.pi * rotor_radius**2) / (farm_size * spacing)**2
        velocity_deficit = blockage_amplitude * ct * farm_density * upstream_decay * lateral_decay
        u_engineering[global_mask] -= u_freestream * velocity_deficit
    
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Common colormap limits
    vmin = u_freestream * 0.75
    vmax = u_freestream * 1.02
    
    # Plot each model
    im1 = axes[0].contourf(X, Y, u_param_global, 50, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0].set_title('Parametrized Global Model (Detailed)', fontsize=12)
    axes[0].set_xlabel('x/D', fontsize=10)
    axes[0].set_ylabel('y/D', fontsize=10)
    for tx, ty in zip(turbine_x, turbine_y):
        circle = plt.Circle((tx, ty), rotor_radius, fill=True, color='black')
        axes[0].add_patch(circle)
    axes[0].set_aspect('equal')
    axes[0].annotate('Meyer Forsting et al. (2021)', xy=(-4.5, 5.7), fontsize=8, style='italic')
    
    im2 = axes[1].contourf(X, Y, u_engineering, 50, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_title('Engineering Global Model (Simplified)', fontsize=12)
    axes[1].set_xlabel('x/D', fontsize=10)
    for tx, ty in zip(turbine_x, turbine_y):
        circle = plt.Circle((tx, ty), rotor_radius, fill=True, color='black')
        axes[1].add_patch(circle)
    axes[1].set_aspect('equal')
    axes[1].annotate('Bleeg et al. (2018)', xy=(-4.5, 5.7), fontsize=8, style='italic')
    
    # Add a common colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist())
    cbar.set_label('Velocity (m/s)', fontsize=10)
    
    # Add overall title
    fig.suptitle('Comparison of Farm-Scale Blockage Models - 2D Flow Field', fontsize=14)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "wind_farm_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved wind farm comparison to {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_analytical_vs_simulation_plots()
    generate_2d_comparison_plots()
