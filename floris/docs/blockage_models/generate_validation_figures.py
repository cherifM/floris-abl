#!/usr/bin/env python
# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""
This script generates validation figures for the blockage models implemented in FLORIS.
It creates a set of comparison figures for different scenarios to validate
the blockage model implementations.
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
import time

import floris.tools as wfct


# Directory to save figures
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom colormap for velocity differences
colors = [(0, 0, 0.8), (0, 0.8, 0.8), (1, 1, 1), (0.8, 0.4, 0), (0.8, 0, 0)]
cmap_diff = LinearSegmentedColormap.from_list("velocity_diff", colors, N=100)


def setup_base_floris():
    """Set up a base FLORIS model with default settings."""
    input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(OUTPUT_DIR))), 
                            "example_input_abl.yaml")
    fi = wfct.floris_interface.FlorisInterface(input_path)
    return fi


def setup_blockage_models(fi_base, model_configs=None):
    """
    Set up FLORIS instances with different blockage models.
    
    Args:
        fi_base: Base FLORIS interface
        model_configs: Dictionary of model configurations
        
    Returns:
        Dictionary of FLORIS interfaces with different blockage models
    """
    if model_configs is None:
        # Default model configurations
        model_configs = {
            "none": {
                "blockage_model": "none",
                "parameters": {}
            },
            "parametrized_global": {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.05,
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            },
            "vortex_cylinder": {
                "blockage_model": "vortex_cylinder",
                "parameters": {
                    "include_ground_effect": False,
                    "finite_length": False,
                    "wake_length": 10.0
                }
            },
            "mirrored_vortex": {
                "blockage_model": "mirrored_vortex",
                "parameters": {
                    "finite_length": False,
                    "wake_length": 10.0,
                    "mirror_weight": 1.0
                }
            },
            "self_similar": {
                "blockage_model": "self_similar",
                "parameters": {
                    "alpha": 0.8,
                    "beta": 2.0,
                    "delta_max": 0.2
                }
            },
            "engineering_global": {
                "blockage_model": "engineering_global",
                "parameters": {
                    "blockage_amplitude": 0.1,
                    "lateral_extent": 2.5,
                    "upstream_extent": 3.0,
                    "vertical_extent": 2.0
                }
            }
        }
    
    fi_models = {}
    
    for model_name, config in model_configs.items():
        fi = copy.deepcopy(fi_base)
        
        wake_config = {
            "model_strings": {
                "velocity_model": "gauss", 
                "deflection_model": "gauss", 
                "combination_model": "sosfs",
                "turbulence_model": "crespo_hernandez",
                "blockage_model": config["blockage_model"]
            },
            "enable_blockage": True
        }
        
        if config["parameters"]:
            wake_config["wake_blockage_parameters"] = {
                config["blockage_model"]: config["parameters"]
            }
        
        fi.set_wake_model(wake=wake_config)
        fi_models[model_name] = fi
    
    return fi_models


def setup_single_turbine(fi_base):
    """
    Set up a single turbine scenario for blockage validation.
    
    Args:
        fi_base: Base FLORIS interface
        
    Returns:
        Updated FLORIS interface with single turbine
    """
    fi = copy.deepcopy(fi_base)
    
    # Set a single turbine layout
    fi.reinitialize(layout_x=[0.0], layout_y=[0.0])
    
    # Set flow conditions
    fi.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    
    return fi


def generate_single_turbine_centerline(save_path=None):
    """
    Generate comparison of velocity deficit along the centerline upstream 
    of a single turbine for all blockage models.
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up single turbine scenario
    fi_base = setup_single_turbine(fi_base)
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # Set up all blockage models
    fi_models = setup_blockage_models(fi_base)
    
    # Calculate wake for each model
    for model_name, fi in fi_models.items():
        fi.calculate_wake()
    
    # Set up evaluation points along centerline upstream
    x_D = np.linspace(-5, 0, 100)  # x/D (negative is upstream)
    x = x_D * D  # Convert to meters
    y = np.zeros_like(x)
    z = np.ones_like(x) * fi_base.floris.farm.hub_heights[0][0]
    
    # Calculate velocities for each model
    velocities = {}
    for model_name, fi in fi_models.items():
        velocities[model_name] = fi.calculate_flow_field(x, y, z)
    
    # Calculate deficit as percentage
    freestream = fi_base.floris.flow_field.wind_speeds[0]
    deficits = {}
    for model_name, vel in velocities.items():
        deficits[model_name] = 100 * (freestream - vel) / freestream
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot each model
    for model_name, deficit in deficits.items():
        if model_name == "none":
            continue  # Skip the none model which should have zero deficit
        plt.plot(x_D, deficit, label=model_name.replace("_", " ").title(), linewidth=2)
    
    # Add reference line at x=0 (turbine location)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Add labels and legend
    plt.xlabel("Distance Upstream (x/D)")
    plt.ylabel("Velocity Deficit (%)")
    plt.title("Blockage Effect: Velocity Deficit Along Centerline Upstream of Single Turbine")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis limits for better visualization
    plt.ylim(0, 3.5)
    
    # Invert x-axis to show upstream direction from right to left
    plt.xlim(0, -5)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def setup_three_turbine_row(fi_base):
    """
    Set up a three-turbine row perpendicular to the wind direction.
    
    Args:
        fi_base: Base FLORIS interface
        
    Returns:
        Updated FLORIS interface with three turbines in a row
    """
    fi = copy.deepcopy(fi_base)
    
    # Get rotor diameter
    D = fi.floris.farm.rotor_diameters[0][0]
    
    # Set layout with three turbines in a row perpendicular to wind
    # (wind from the west at 270°, so the row is north-south)
    fi.reinitialize(layout_x=[0.0, 0.0, 0.0], layout_y=[-7*D, 0.0, 7*D])
    
    # Set flow conditions
    fi.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    
    return fi


def generate_three_turbine_horizontal_cut(save_path=None):
    """
    Generate horizontal cut planes showing blockage effects for three turbines in a row.
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up three-turbine scenario
    fi_base = setup_three_turbine_row(fi_base)
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # Set up selected blockage models
    model_selection = ["none", "vortex_cylinder", "parametrized_global", "self_similar"]
    
    model_configs = {}
    for model in model_selection:
        if model == "none":
            model_configs[model] = {
                "blockage_model": "none",
                "parameters": {}
            }
        elif model == "vortex_cylinder":
            model_configs[model] = {
                "blockage_model": "vortex_cylinder",
                "parameters": {
                    "include_ground_effect": False,
                    "finite_length": False
                }
            }
        elif model == "parametrized_global":
            model_configs[model] = {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.07,  # Increased for better visualization
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            }
        elif model == "self_similar":
            model_configs[model] = {
                "blockage_model": "self_similar",
                "parameters": {
                    "alpha": 0.8,
                    "beta": 2.0,
                    "delta_max": 0.2
                }
            }
    
    fi_models = setup_blockage_models(fi_base, model_configs)
    
    # Calculate wake for each model
    for fi in fi_models.values():
        fi.calculate_wake()
    
    # Set up horizontal cut parameters
    x_range = (-5*D, D)
    y_range = (-10*D, 10*D)  # Wide enough to capture all three turbines
    z_plane = fi_base.floris.farm.hub_heights[0][0]
    
    # Generate cut planes for each model
    horizontal_planes = {}
    for model_name, fi in fi_models.items():
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=100,
            y_resolution=100,
            x_min=x_range[0],
            x_max=x_range[1],
            y_min=y_range[0],
            y_max=y_range[1],
            z=z_plane
        )
        horizontal_plane.calculate_wake(fi)
        horizontal_planes[model_name] = horizontal_plane
    
    # Create subplots for comparison
    fig, axarr = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    axarr = axarr.flatten()
    
    # Calculate common velocity range for consistent colormaps
    vmin = min([hp.df.u.min() for hp in horizontal_planes.values()])
    vmax = max([hp.df.u.max() for hp in horizontal_planes.values()])
    
    # Plot each model
    for i, (model_name, hp) in enumerate(horizontal_planes.items()):
        ax = axarr[i]
        
        # Reshape data for contour plot
        x_mesh = hp.df.x.values.reshape(hp.resolution[1], hp.resolution[0])
        y_mesh = hp.df.y.values.reshape(hp.resolution[1], hp.resolution[0])
        u_mesh = hp.df.u.values.reshape(hp.resolution[1], hp.resolution[0])
        
        # Create the contour plot
        im = ax.contourf(
            x_mesh / D,  # Normalize by diameter
            y_mesh / D,
            u_mesh,
            levels=50,
            cmap=cm.coolwarm,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add turbine markers
        for x, y in zip(fi_base.layout_x, fi_base.layout_y):
            ax.plot(x/D, y/D, 'ko', markersize=8)
            circle = plt.Circle(
                (x/D, y/D),
                0.5,  # Radius is 0.5D
                fill=False,
                color='k'
            )
            ax.add_patch(circle)
        
        # Add title and labels
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add velocity deficit contour lines
        freestream = fi_base.floris.flow_field.wind_speeds[0]
        deficit_levels = [0.5, 1.0, 2.0, 3.0, 4.0]  # Percentage levels
        deficit_contours = ax.contour(
            x_mesh / D,
            y_mesh / D,
            100 * (freestream - u_mesh) / freestream,
            levels=deficit_levels,
            colors='k',
            linewidths=0.8,
            alpha=0.7
        )
        ax.clabel(deficit_contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.tolist())
    cbar.set_label('Wind Speed (m/s)')
    
    # Add overall title
    fig.suptitle("Blockage Effect: Horizontal Cut Plane for Three Turbines in a Row", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def setup_small_wind_farm(fi_base, layout="3x3"):
    """
    Set up a small wind farm layout.
    
    Args:
        fi_base: Base FLORIS interface
        layout: Layout type, either "3x3" or "5x5"
        
    Returns:
        Updated FLORIS interface with wind farm layout
    """
    fi = copy.deepcopy(fi_base)
    D = fi.floris.farm.rotor_diameters[0][0]
    
    # Create the layout based on the specified type
    if layout == "3x3":
        n = 3
    elif layout == "5x5":
        n = 5
    else:
        raise ValueError(f"Unknown layout type: {layout}")
    
    # Create a grid layout with 7D spacing
    spacing = 7 * D
    layout_x = []
    layout_y = []
    
    for i in range(n):
        for j in range(n):
            layout_x.append(i * spacing)
            layout_y.append(j * spacing)
    
    # Set the layout
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
    
    # Set flow conditions
    fi.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    
    return fi


def generate_small_farm_blockage(save_path=None):
    """
    Generate horizontal cut planes showing blockage effects for a small wind farm (3x3).
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up small wind farm scenario (3x3)
    fi_base = setup_small_wind_farm(fi_base, layout="3x3")
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # Set up selected blockage models
    model_selection = ["none", "vortex_cylinder", "parametrized_global", "engineering_global"]
    
    model_configs = {}
    for model in model_selection:
        if model == "none":
            model_configs[model] = {
                "blockage_model": "none",
                "parameters": {}
            }
        elif model == "vortex_cylinder":
            model_configs[model] = {
                "blockage_model": "vortex_cylinder",
                "parameters": {
                    "include_ground_effect": False,
                    "finite_length": False
                }
            }
        elif model == "parametrized_global":
            model_configs[model] = {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.07,
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            }
        elif model == "engineering_global":
            model_configs[model] = {
                "blockage_model": "engineering_global",
                "parameters": {
                    "blockage_amplitude": 0.1,
                    "lateral_extent": 2.5,
                    "upstream_extent": 3.0,
                    "vertical_extent": 2.0
                }
            }
    
    fi_models = setup_blockage_models(fi_base, model_configs)
    
    # Calculate wake for each model
    for fi in fi_models.values():
        fi.calculate_wake()
    
    # Set up horizontal cut parameters
    # Cover the entire farm plus upstream area
    farm_size = 2 * 7 * D  # 2 spacings of 7D each
    x_range = (-7*D, farm_size + D)
    y_range = (-2*D, farm_size + 2*D)
    z_plane = fi_base.floris.farm.hub_heights[0][0]
    
    # Generate cut planes for each model
    horizontal_planes = {}
    for model_name, fi in fi_models.items():
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=150,
            y_resolution=150,
            x_min=x_range[0],
            x_max=x_range[1],
            y_min=y_range[0],
            y_max=y_range[1],
            z=z_plane
        )
        horizontal_plane.calculate_wake(fi)
        horizontal_planes[model_name] = horizontal_plane
    
    # Create subplots for comparison
    fig, axarr = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    axarr = axarr.flatten()
    
    # Calculate common velocity range for consistent colormaps
    vmin = min([hp.df.u.min() for hp in horizontal_planes.values()])
    vmax = max([hp.df.u.max() for hp in horizontal_planes.values()])
    
    # Plot each model
    for i, (model_name, hp) in enumerate(horizontal_planes.items()):
        ax = axarr[i]
        
        # Reshape data for contour plot
        x_mesh = hp.df.x.values.reshape(hp.resolution[1], hp.resolution[0])
        y_mesh = hp.df.y.values.reshape(hp.resolution[1], hp.resolution[0])
        u_mesh = hp.df.u.values.reshape(hp.resolution[1], hp.resolution[0])
        
        # Create the contour plot
        im = ax.contourf(
            x_mesh / D,  # Normalize by diameter
            y_mesh / D,
            u_mesh,
            levels=50,
            cmap=cm.coolwarm,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add turbine markers
        for x, y in zip(fi_base.layout_x, fi_base.layout_y):
            ax.plot(x/D, y/D, 'ko', markersize=5)
            circle = plt.Circle(
                (x/D, y/D),
                0.5,  # Radius is 0.5D
                fill=False,
                color='k'
            )
            ax.add_patch(circle)
        
        # Add title and labels
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.grid(True, alpha=0.3)
        
        # Add velocity deficit contour lines for blockage visualization
        freestream = fi_base.floris.flow_field.wind_speeds[0]
        deficit_levels = [0.5, 1.0, 2.0, 3.0]  # Percentage levels
        deficit_contours = ax.contour(
            x_mesh / D,
            y_mesh / D,
            100 * (freestream - u_mesh) / freestream,
            levels=deficit_levels,
            colors='k',
            linewidths=0.8,
            alpha=0.7
        )
        ax.clabel(deficit_contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.tolist())
    cbar.set_label('Wind Speed (m/s)')
    
    # Add overall title
    fig.suptitle("Blockage Effect: Horizontal Cut Plane for 3×3 Wind Farm", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def setup_large_wind_farm(fi_base):
    """
    Set up a large wind farm layout (10x10).
    
    Args:
        fi_base: Base FLORIS interface
        
    Returns:
        Updated FLORIS interface with large wind farm layout
    """
    fi = copy.deepcopy(fi_base)
    D = fi.floris.farm.rotor_diameters[0][0]
    
    # Create a 10x10 grid layout with 7D spacing
    n = 10
    spacing = 7 * D
    layout_x = []
    layout_y = []
    
    for i in range(n):
        for j in range(n):
            layout_x.append(i * spacing)
            layout_y.append(j * spacing)
    
    # Set the layout
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
    
    # Set flow conditions
    fi.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    
    return fi


def generate_large_farm_blockage(save_path=None):
    """
    Generate horizontal cut planes showing blockage effects for a large wind farm (10x10).
    This example uses only the global blockage models due to computational efficiency.
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up large wind farm scenario (10x10)
    fi_base = setup_large_wind_farm(fi_base)
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # For large farms, focus on global models for computational efficiency
    model_selection = ["none", "parametrized_global", "engineering_global"]
    
    model_configs = {}
    for model in model_selection:
        if model == "none":
            model_configs[model] = {
                "blockage_model": "none",
                "parameters": {}
            }
        elif model == "parametrized_global":
            model_configs[model] = {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.08,
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            }
        elif model == "engineering_global":
            model_configs[model] = {
                "blockage_model": "engineering_global",
                "parameters": {
                    "blockage_amplitude": 0.12,
                    "lateral_extent": 3.0,
                    "upstream_extent": 3.5,
                    "vertical_extent": 2.0
                }
            }
    
    fi_models = setup_blockage_models(fi_base, model_configs)
    
    # Calculate wake for each model
    for fi in fi_models.values():
        fi.calculate_wake()
    
    # Set up horizontal cut parameters
    # Cover the entire farm plus upstream area
    farm_size = 9 * 7 * D  # 9 spacings of 7D each
    x_range = (-10*D, farm_size + 5*D)
    y_range = (-5*D, farm_size + 5*D)
    z_plane = fi_base.floris.farm.hub_heights[0][0]
    
    # Generate cut planes for each model
    horizontal_planes = {}
    for model_name, fi in fi_models.items():
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=200,
            y_resolution=200,
            x_min=x_range[0],
            x_max=x_range[1],
            y_min=y_range[0],
            y_max=y_range[1],
            z=z_plane
        )
        horizontal_plane.calculate_wake(fi)
        horizontal_planes[model_name] = horizontal_plane
    
    # Create a figure focusing on comparing the global models
    fig, axarr = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    # Calculate common velocity range for consistent colormaps
    vmin = min([hp.df.u.min() for hp in horizontal_planes.values()])
    vmax = max([hp.df.u.max() for hp in horizontal_planes.values()])
    
    # Plot each model
    for i, (model_name, hp) in enumerate(horizontal_planes.items()):
        ax = axarr[i]
        
        # Reshape data for contour plot
        x_mesh = hp.df.x.values.reshape(hp.resolution[1], hp.resolution[0])
        y_mesh = hp.df.y.values.reshape(hp.resolution[1], hp.resolution[0])
        u_mesh = hp.df.u.values.reshape(hp.resolution[1], hp.resolution[0])
        
        # Create the contour plot
        im = ax.contourf(
            x_mesh / D,  # Normalize by diameter
            y_mesh / D,
            u_mesh,
            levels=50,
            cmap=cm.coolwarm,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add a simplified representation of turbines (just dots for large farm)
        for x, y in zip(fi_base.layout_x, fi_base.layout_y):
            ax.plot(x/D, y/D, 'ko', markersize=2)
        
        # Add title and labels
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("x/D")
        if i == 0:
            ax.set_ylabel("y/D")
        ax.grid(True, alpha=0.3)
        
        # For the global models, add velocity deficit contour lines
        if model_name != "none":
            freestream = fi_base.floris.flow_field.wind_speeds[0]
            deficit_levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]  # Percentage levels
            deficit_contours = ax.contour(
                x_mesh / D,
                y_mesh / D,
                100 * (freestream - u_mesh) / freestream,
                levels=deficit_levels,
                colors='k',
                linewidths=0.8,
                alpha=0.7
            )
            ax.clabel(deficit_contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.tolist())
    cbar.set_label('Wind Speed (m/s)')
    
    # Add overall title
    fig.suptitle("Blockage Effect: Horizontal Cut Plane for 10×10 Wind Farm", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def generate_single_turbine_horizontal_cut(save_path=None):
    """
    Generate horizontal cut planes showing blockage effects for a single turbine.
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up single turbine scenario
    fi_base = setup_single_turbine(fi_base)
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # Set up selected blockage models (one from each category)
    model_selection = ["none", "vortex_cylinder", "parametrized_global", "self_similar"]
    
    model_configs = {}
    for model in model_selection:
        if model == "none":
            model_configs[model] = {
                "blockage_model": "none",
                "parameters": {}
            }
        elif model == "vortex_cylinder":
            model_configs[model] = {
                "blockage_model": "vortex_cylinder",
                "parameters": {
                    "include_ground_effect": False,
                    "finite_length": False
                }
            }
        elif model == "parametrized_global":
            model_configs[model] = {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.05,
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            }
        elif model == "self_similar":
            model_configs[model] = {
                "blockage_model": "self_similar",
                "parameters": {
                    "alpha": 0.8,
                    "beta": 2.0,
                    "delta_max": 0.2
                }
            }
    
    fi_models = setup_blockage_models(fi_base, model_configs)
    
    # Calculate wake for each model
    for fi in fi_models.values():
        fi.calculate_wake()
    
    # Set up horizontal cut parameters
    x_range = (-5*D, D)
    y_range = (-3*D, 3*D)
    z_plane = fi_base.floris.farm.hub_heights[0][0]
    
    # Generate cut planes for each model
    horizontal_planes = {}
    for model_name, fi in fi_models.items():
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=100,
            y_resolution=100,
            x_min=x_range[0],
            x_max=x_range[1],
            y_min=y_range[0],
            y_max=y_range[1],
            z=z_plane
        )
        horizontal_plane.calculate_wake(fi)
        horizontal_planes[model_name] = horizontal_plane
    
    # Create subplots for comparison
    fig, axarr = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axarr = axarr.flatten()
    
    # Calculate common velocity range for consistent colormaps
    vmin = min([hp.df.u.min() for hp in horizontal_planes.values()])
    vmax = max([hp.df.u.max() for hp in horizontal_planes.values()])
    
    # Plot each model
    for i, (model_name, hp) in enumerate(horizontal_planes.items()):
        ax = axarr[i]
        
        # Reshape data for contour plot
        x_mesh = hp.df.x.values.reshape(hp.resolution[1], hp.resolution[0])
        y_mesh = hp.df.y.values.reshape(hp.resolution[1], hp.resolution[0])
        u_mesh = hp.df.u.values.reshape(hp.resolution[1], hp.resolution[0])
        
        # Create the contour plot
        im = ax.contourf(
            x_mesh / D,  # Normalize by diameter
            y_mesh / D,
            u_mesh,
            levels=50,
            cmap=cm.coolwarm,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add turbine marker
        for x, y in zip(fi_base.layout_x, fi_base.layout_y):
            ax.plot(x/D, y/D, 'ko', markersize=8)
            circle = plt.Circle(
                (x/D, y/D),
                0.5,  # Radius is 0.5D
                fill=False,
                color='k'
            )
            ax.add_patch(circle)
        
        # Add title and labels
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add velocity deficit contour lines
        freestream = fi_base.floris.flow_field.wind_speeds[0]
        deficit_levels = [0.5, 1.0, 2.0, 3.0]  # Percentage levels
        deficit_contours = ax.contour(
            x_mesh / D,
            y_mesh / D,
            100 * (freestream - u_mesh) / freestream,
            levels=deficit_levels,
            colors='k',
            linewidths=0.8,
            alpha=0.7
        )
        ax.clabel(deficit_contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.tolist())
    cbar.set_label('Wind Speed (m/s)')
    
    # Add overall title
    fig.suptitle("Blockage Effect: Horizontal Cut Plane for Single Turbine", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def setup_three_turbine_row(fi_base):
    """
    Set up a three-turbine row perpendicular to the wind direction.
    
    Args:
        fi_base: Base FLORIS interface
        
    Returns:
        Updated FLORIS interface with three turbines in a row
    """
    fi = copy.deepcopy(fi_base)
    
    # Get rotor diameter
    D = fi.floris.farm.rotor_diameters[0][0]
    
    # Set layout with three turbines in a row perpendicular to wind
    # (wind from the west at 270°, so the row is north-south)
    fi.reinitialize(layout_x=[0.0, 0.0, 0.0], layout_y=[-7*D, 0.0, 7*D])
    
    # Set flow conditions
    fi.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    
    return fi


def generate_three_turbine_horizontal_cut(save_path=None):
    """
    Generate horizontal cut planes showing blockage effects for three turbines in a row.
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up three-turbine scenario
    fi_base = setup_three_turbine_row(fi_base)
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # Set up selected blockage models
    model_selection = ["none", "vortex_cylinder", "parametrized_global", "self_similar"]
    
    model_configs = {}
    for model in model_selection:
        if model == "none":
            model_configs[model] = {
                "blockage_model": "none",
                "parameters": {}
            }
        elif model == "vortex_cylinder":
            model_configs[model] = {
                "blockage_model": "vortex_cylinder",
                "parameters": {
                    "include_ground_effect": False,
                    "finite_length": False
                }
            }
        elif model == "parametrized_global":
            model_configs[model] = {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.07,  # Increased for better visualization
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            }
        elif model == "self_similar":
            model_configs[model] = {
                "blockage_model": "self_similar",
                "parameters": {
                    "alpha": 0.8,
                    "beta": 2.0,
                    "delta_max": 0.2
                }
            }
    
    fi_models = setup_blockage_models(fi_base, model_configs)
    
    # Calculate wake for each model
    for fi in fi_models.values():
        fi.calculate_wake()
    
    # Set up horizontal cut parameters
    x_range = (-5*D, D)
    y_range = (-10*D, 10*D)  # Wide enough to capture all three turbines
    z_plane = fi_base.floris.farm.hub_heights[0][0]
    
    # Generate cut planes for each model
    horizontal_planes = {}
    for model_name, fi in fi_models.items():
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=100,
            y_resolution=100,
            x_min=x_range[0],
            x_max=x_range[1],
            y_min=y_range[0],
            y_max=y_range[1],
            z=z_plane
        )
        horizontal_plane.calculate_wake(fi)
        horizontal_planes[model_name] = horizontal_plane
    
    # Create subplots for comparison
    fig, axarr = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    axarr = axarr.flatten()
    
    # Calculate common velocity range for consistent colormaps
    vmin = min([hp.df.u.min() for hp in horizontal_planes.values()])
    vmax = max([hp.df.u.max() for hp in horizontal_planes.values()])
    
    # Plot each model
    for i, (model_name, hp) in enumerate(horizontal_planes.items()):
        ax = axarr[i]
        
        # Reshape data for contour plot
        x_mesh = hp.df.x.values.reshape(hp.resolution[1], hp.resolution[0])
        y_mesh = hp.df.y.values.reshape(hp.resolution[1], hp.resolution[0])
        u_mesh = hp.df.u.values.reshape(hp.resolution[1], hp.resolution[0])
        
        # Create the contour plot
        im = ax.contourf(
            x_mesh / D,  # Normalize by diameter
            y_mesh / D,
            u_mesh,
            levels=50,
            cmap=cm.coolwarm,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add turbine markers
        for x, y in zip(fi_base.layout_x, fi_base.layout_y):
            ax.plot(x/D, y/D, 'ko', markersize=8)
            circle = plt.Circle(
                (x/D, y/D),
                0.5,  # Radius is 0.5D
                fill=False,
                color='k'
            )
            ax.add_patch(circle)
        
        # Add title and labels
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add velocity deficit contour lines
        freestream = fi_base.floris.flow_field.wind_speeds[0]
        deficit_levels = [0.5, 1.0, 2.0, 3.0, 4.0]  # Percentage levels
        deficit_contours = ax.contour(
            x_mesh / D,
            y_mesh / D,
            100 * (freestream - u_mesh) / freestream,
            levels=deficit_levels,
            colors='k',
            linewidths=0.8,
            alpha=0.7
        )
        ax.clabel(deficit_contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.tolist())
    cbar.set_label('Wind Speed (m/s)')
    
    # Add overall title
    fig.suptitle("Blockage Effect: Horizontal Cut Plane for Three Turbines in a Row", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def setup_small_wind_farm(fi_base, layout="3x3"):
    """
    Set up a small wind farm layout.
    
    Args:
        fi_base: Base FLORIS interface
        layout: Layout type, either "3x3" or "5x5"
        
    Returns:
        Updated FLORIS interface with wind farm layout
    """
    fi = copy.deepcopy(fi_base)
    D = fi.floris.farm.rotor_diameters[0][0]
    
    # Create the layout based on the specified type
    if layout == "3x3":
        n = 3
    elif layout == "5x5":
        n = 5
    else:
        raise ValueError(f"Unknown layout type: {layout}")
    
    # Create a grid layout with 7D spacing
    spacing = 7 * D
    layout_x = []
    layout_y = []
    
    for i in range(n):
        for j in range(n):
            layout_x.append(i * spacing)
            layout_y.append(j * spacing)
    
    # Set the layout
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
    
    # Set flow conditions
    fi.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    
    return fi


def generate_small_farm_blockage(save_path=None):
    """
    Generate horizontal cut planes showing blockage effects for a small wind farm (3x3).
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up small wind farm scenario (3x3)
    fi_base = setup_small_wind_farm(fi_base, layout="3x3")
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # Set up selected blockage models
    model_selection = ["none", "vortex_cylinder", "parametrized_global", "engineering_global"]
    
    model_configs = {}
    for model in model_selection:
        if model == "none":
            model_configs[model] = {
                "blockage_model": "none",
                "parameters": {}
            }
        elif model == "vortex_cylinder":
            model_configs[model] = {
                "blockage_model": "vortex_cylinder",
                "parameters": {
                    "include_ground_effect": False,
                    "finite_length": False
                }
            }
        elif model == "parametrized_global":
            model_configs[model] = {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.07,
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            }
        elif model == "engineering_global":
            model_configs[model] = {
                "blockage_model": "engineering_global",
                "parameters": {
                    "blockage_amplitude": 0.1,
                    "lateral_extent": 2.5,
                    "upstream_extent": 3.0,
                    "vertical_extent": 2.0
                }
            }
    
    fi_models = setup_blockage_models(fi_base, model_configs)
    
    # Calculate wake for each model
    for fi in fi_models.values():
        fi.calculate_wake()
    
    # Set up horizontal cut parameters
    # Cover the entire farm plus upstream area
    farm_size = 2 * 7 * D  # 2 spacings of 7D each
    x_range = (-7*D, farm_size + D)
    y_range = (-2*D, farm_size + 2*D)
    z_plane = fi_base.floris.farm.hub_heights[0][0]
    
    # Generate cut planes for each model
    horizontal_planes = {}
    for model_name, fi in fi_models.items():
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=150,
            y_resolution=150,
            x_min=x_range[0],
            x_max=x_range[1],
            y_min=y_range[0],
            y_max=y_range[1],
            z=z_plane
        )
        horizontal_plane.calculate_wake(fi)
        horizontal_planes[model_name] = horizontal_plane
    
    # Create subplots for comparison
    fig, axarr = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    axarr = axarr.flatten()
    
    # Calculate common velocity range for consistent colormaps
    vmin = min([hp.df.u.min() for hp in horizontal_planes.values()])
    vmax = max([hp.df.u.max() for hp in horizontal_planes.values()])
    
    # Plot each model
    for i, (model_name, hp) in enumerate(horizontal_planes.items()):
        ax = axarr[i]
        
        # Reshape data for contour plot
        x_mesh = hp.df.x.values.reshape(hp.resolution[1], hp.resolution[0])
        y_mesh = hp.df.y.values.reshape(hp.resolution[1], hp.resolution[0])
        u_mesh = hp.df.u.values.reshape(hp.resolution[1], hp.resolution[0])
        
        # Create the contour plot
        im = ax.contourf(
            x_mesh / D,  # Normalize by diameter
            y_mesh / D,
            u_mesh,
            levels=50,
            cmap=cm.coolwarm,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add turbine markers
        for x, y in zip(fi_base.layout_x, fi_base.layout_y):
            ax.plot(x/D, y/D, 'ko', markersize=5)
            circle = plt.Circle(
                (x/D, y/D),
                0.5,  # Radius is 0.5D
                fill=False,
                color='k'
            )
            ax.add_patch(circle)
        
        # Add title and labels
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.grid(True, alpha=0.3)
        
        # Add velocity deficit contour lines for blockage visualization
        freestream = fi_base.floris.flow_field.wind_speeds[0]
        deficit_levels = [0.5, 1.0, 2.0, 3.0]  # Percentage levels
        deficit_contours = ax.contour(
            x_mesh / D,
            y_mesh / D,
            100 * (freestream - u_mesh) / freestream,
            levels=deficit_levels,
            colors='k',
            linewidths=0.8,
            alpha=0.7
        )
        ax.clabel(deficit_contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.tolist())
    cbar.set_label('Wind Speed (m/s)')
    
    # Add overall title
    fig.suptitle("Blockage Effect: Horizontal Cut Plane for 3×3 Wind Farm", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def setup_large_wind_farm(fi_base):
    """
    Set up a large wind farm layout (10x10).
    
    Args:
        fi_base: Base FLORIS interface
        
    Returns:
        Updated FLORIS interface with large wind farm layout
    """
    fi = copy.deepcopy(fi_base)
    D = fi.floris.farm.rotor_diameters[0][0]
    
    # Create a 10x10 grid layout with 7D spacing
    n = 10
    spacing = 7 * D
    layout_x = []
    layout_y = []
    
    for i in range(n):
        for j in range(n):
            layout_x.append(i * spacing)
            layout_y.append(j * spacing)
    
    # Set the layout
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
    
    # Set flow conditions
    fi.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    
    return fi


def generate_large_farm_blockage(save_path=None):
    """
    Generate horizontal cut planes showing blockage effects for a large wind farm (10x10).
    This example uses only the global blockage models due to computational efficiency.
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up large wind farm scenario (10x10)
    fi_base = setup_large_wind_farm(fi_base)
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # For large farms, focus on global models for computational efficiency
    model_selection = ["none", "parametrized_global", "engineering_global"]
    
    model_configs = {}
    for model in model_selection:
        if model == "none":
            model_configs[model] = {
                "blockage_model": "none",
                "parameters": {}
            }
        elif model == "parametrized_global":
            model_configs[model] = {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.08,
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            }
        elif model == "engineering_global":
            model_configs[model] = {
                "blockage_model": "engineering_global",
                "parameters": {
                    "blockage_amplitude": 0.12,
                    "lateral_extent": 3.0,
                    "upstream_extent": 3.5,
                    "vertical_extent": 2.0
                }
            }
    
    fi_models = setup_blockage_models(fi_base, model_configs)
    
    # Calculate wake for each model
    for fi in fi_models.values():
        fi.calculate_wake()
    
    # Set up horizontal cut parameters
    # Cover the entire farm plus upstream area
    farm_size = 9 * 7 * D  # 9 spacings of 7D each
    x_range = (-10*D, farm_size + 5*D)
    y_range = (-5*D, farm_size + 5*D)
    z_plane = fi_base.floris.farm.hub_heights[0][0]
    
    # Generate cut planes for each model
    horizontal_planes = {}
    for model_name, fi in fi_models.items():
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=200,
            y_resolution=200,
            x_min=x_range[0],
            x_max=x_range[1],
            y_min=y_range[0],
            y_max=y_range[1],
            z=z_plane
        )
        horizontal_plane.calculate_wake(fi)
        horizontal_planes[model_name] = horizontal_plane
    
    # Create a figure focusing on comparing the global models
    fig, axarr = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    # Calculate common velocity range for consistent colormaps
    vmin = min([hp.df.u.min() for hp in horizontal_planes.values()])
    vmax = max([hp.df.u.max() for hp in horizontal_planes.values()])
    
    # Plot each model
    for i, (model_name, hp) in enumerate(horizontal_planes.items()):
        ax = axarr[i]
        
        # Reshape data for contour plot
        x_mesh = hp.df.x.values.reshape(hp.resolution[1], hp.resolution[0])
        y_mesh = hp.df.y.values.reshape(hp.resolution[1], hp.resolution[0])
        u_mesh = hp.df.u.values.reshape(hp.resolution[1], hp.resolution[0])
        
        # Create the contour plot
        im = ax.contourf(
            x_mesh / D,  # Normalize by diameter
            y_mesh / D,
            u_mesh,
            levels=50,
            cmap=cm.coolwarm,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add a simplified representation of turbines (just dots for large farm)
        for x, y in zip(fi_base.layout_x, fi_base.layout_y):
            ax.plot(x/D, y/D, 'ko', markersize=2)
        
        # Add title and labels
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("x/D")
        if i == 0:
            ax.set_ylabel("y/D")
        ax.grid(True, alpha=0.3)
        
        # For the global models, add velocity deficit contour lines
        if model_name != "none":
            freestream = fi_base.floris.flow_field.wind_speeds[0]
            deficit_levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]  # Percentage levels
            deficit_contours = ax.contour(
                x_mesh / D,
                y_mesh / D,
                100 * (freestream - u_mesh) / freestream,
                levels=deficit_levels,
                colors='k',
                linewidths=0.8,
                alpha=0.7
            )
            ax.clabel(deficit_contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.tolist())
    cbar.set_label('Wind Speed (m/s)')
    
    # Add overall title
    fig.suptitle("Blockage Effect: Horizontal Cut Plane for 10×10 Wind Farm", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def generate_hub_height_influence(save_path=None):
    """
    Generate comparison of blockage effect for different hub heights.
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up single turbine scenario
    fi_base = setup_single_turbine(fi_base)
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # Set different hub heights to test
    hub_heights = [1.5*D, 3*D, 5*D]  # Low, medium, high
    
    # Set up models to test
    models_to_test = ["vortex_cylinder", "mirrored_vortex"]
    
    # Storage for results
    results = {}
    
    for model_name in models_to_test:
        results[model_name] = {}
        
        for hub_height in hub_heights:
            # Create a new FLORIS instance with the specific hub height
            fi = copy.deepcopy(fi_base)
            fi.reinitialize(hub_heights=[hub_height])
            
            # Set up the blockage model
            if model_name == "vortex_cylinder":
                model_config = {
                    model_name: {
                        "blockage_model": "vortex_cylinder",
                        "parameters": {
                            "include_ground_effect": False,
                            "finite_length": False
                        }
                    }
                }
            else:  # mirrored_vortex
                model_config = {
                    model_name: {
                        "blockage_model": "mirrored_vortex",
                        "parameters": {
                            "finite_length": False,
                            "mirror_weight": 1.0
                        }
                    }
                }
            
            fi_model = setup_blockage_models(fi, model_config)[model_name]
            fi_model.calculate_wake()
            
            # Set up evaluation points along centerline upstream
            x_D = np.linspace(-5, 0, 50)  # x/D (negative is upstream)
            x = x_D * D  # Convert to meters
            y = np.zeros_like(x)
            z = np.ones_like(x) * hub_height
            
            # Calculate velocities
            velocities = fi_model.calculate_flow_field(x, y, z)
            
            # Calculate deficit as percentage
            freestream = fi_model.floris.flow_field.wind_speeds[0]
            deficit = 100 * (freestream - velocities) / freestream
            
            # Store results
            results[model_name][hub_height] = {
                "x_D": x_D,
                "deficit": deficit
            }
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Line styles for different hub heights
    linestyles = ['-', '--', ':']
    
    # Plot each model and hub height
    for i, model_name in enumerate(models_to_test):
        for j, hub_height in enumerate(hub_heights):
            label = f"{model_name.replace('_', ' ').title()}, H={hub_height/D:.1f}D"
            plt.plot(
                results[model_name][hub_height]["x_D"],
                results[model_name][hub_height]["deficit"],
                label=label,
                linestyle=linestyles[j],
                linewidth=2,
                color=f"C{i}"
            )
    
    # Add reference line at x=0 (turbine location)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Add labels and legend
    plt.xlabel("Distance Upstream (x/D)")
    plt.ylabel("Velocity Deficit (%)")
    plt.title("Influence of Hub Height on Blockage Effect")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis limits for better visualization
    plt.ylim(0, 4.0)
    
    # Invert x-axis to show upstream direction from right to left
    plt.xlim(0, -5)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def setup_three_turbine_row(fi_base):
    """
    Set up a three-turbine row perpendicular to the wind direction.
    
    Args:
        fi_base: Base FLORIS interface
        
    Returns:
        Updated FLORIS interface with three turbines in a row
    """
    fi = copy.deepcopy(fi_base)
    
    # Get rotor diameter
    D = fi.floris.farm.rotor_diameters[0][0]
    
    # Set layout with three turbines in a row perpendicular to wind
    # (wind from the west at 270°, so the row is north-south)
    fi.reinitialize(layout_x=[0.0, 0.0, 0.0], layout_y=[-7*D, 0.0, 7*D])
    
    # Set flow conditions
    fi.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    
    return fi


def generate_three_turbine_horizontal_cut(save_path=None):
    """
    Generate horizontal cut planes showing blockage effects for three turbines in a row.
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up three-turbine scenario
    fi_base = setup_three_turbine_row(fi_base)
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # Set up selected blockage models
    model_selection = ["none", "vortex_cylinder", "parametrized_global", "self_similar"]
    
    model_configs = {}
    for model in model_selection:
        if model == "none":
            model_configs[model] = {
                "blockage_model": "none",
                "parameters": {}
            }
        elif model == "vortex_cylinder":
            model_configs[model] = {
                "blockage_model": "vortex_cylinder",
                "parameters": {
                    "include_ground_effect": False,
                    "finite_length": False
                }
            }
        elif model == "parametrized_global":
            model_configs[model] = {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.07,  # Increased for better visualization
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            }
        elif model == "self_similar":
            model_configs[model] = {
                "blockage_model": "self_similar",
                "parameters": {
                    "alpha": 0.8,
                    "beta": 2.0,
                    "delta_max": 0.2
                }
            }
    
    fi_models = setup_blockage_models(fi_base, model_configs)
    
    # Calculate wake for each model
    for fi in fi_models.values():
        fi.calculate_wake()
    
    # Set up horizontal cut parameters
    x_range = (-5*D, D)
    y_range = (-10*D, 10*D)  # Wide enough to capture all three turbines
    z_plane = fi_base.floris.farm.hub_heights[0][0]
    
    # Generate cut planes for each model
    horizontal_planes = {}
    for model_name, fi in fi_models.items():
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=100,
            y_resolution=100,
            x_min=x_range[0],
            x_max=x_range[1],
            y_min=y_range[0],
            y_max=y_range[1],
            z=z_plane
        )
        horizontal_plane.calculate_wake(fi)
        horizontal_planes[model_name] = horizontal_plane
    
    # Create subplots for comparison
    fig, axarr = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    axarr = axarr.flatten()
    
    # Calculate common velocity range for consistent colormaps
    vmin = min([hp.df.u.min() for hp in horizontal_planes.values()])
    vmax = max([hp.df.u.max() for hp in horizontal_planes.values()])
    
    # Plot each model
    for i, (model_name, hp) in enumerate(horizontal_planes.items()):
        ax = axarr[i]
        
        # Reshape data for contour plot
        x_mesh = hp.df.x.values.reshape(hp.resolution[1], hp.resolution[0])
        y_mesh = hp.df.y.values.reshape(hp.resolution[1], hp.resolution[0])
        u_mesh = hp.df.u.values.reshape(hp.resolution[1], hp.resolution[0])
        
        # Create the contour plot
        im = ax.contourf(
            x_mesh / D,  # Normalize by diameter
            y_mesh / D,
            u_mesh,
            levels=50,
            cmap=cm.coolwarm,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add turbine markers
        for x, y in zip(fi_base.layout_x, fi_base.layout_y):
            ax.plot(x/D, y/D, 'ko', markersize=8)
            circle = plt.Circle(
                (x/D, y/D),
                0.5,  # Radius is 0.5D
                fill=False,
                color='k'
            )
            ax.add_patch(circle)
        
        # Add title and labels
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add velocity deficit contour lines
        freestream = fi_base.floris.flow_field.wind_speeds[0]
        deficit_levels = [0.5, 1.0, 2.0, 3.0, 4.0]  # Percentage levels
        deficit_contours = ax.contour(
            x_mesh / D,
            y_mesh / D,
            100 * (freestream - u_mesh) / freestream,
            levels=deficit_levels,
            colors='k',
            linewidths=0.8,
            alpha=0.7
        )
        ax.clabel(deficit_contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.tolist())
    cbar.set_label('Wind Speed (m/s)')
    
    # Add overall title
    fig.suptitle("Blockage Effect: Horizontal Cut Plane for Three Turbines in a Row", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def setup_small_wind_farm(fi_base, layout="3x3"):
    """
    Set up a small wind farm layout.
    
    Args:
        fi_base: Base FLORIS interface
        layout: Layout type, either "3x3" or "5x5"
        
    Returns:
        Updated FLORIS interface with wind farm layout
    """
    fi = copy.deepcopy(fi_base)
    D = fi.floris.farm.rotor_diameters[0][0]
    
    # Create the layout based on the specified type
    if layout == "3x3":
        n = 3
    elif layout == "5x5":
        n = 5
    else:
        raise ValueError(f"Unknown layout type: {layout}")
    
    # Create a grid layout with 7D spacing
    spacing = 7 * D
    layout_x = []
    layout_y = []
    
    for i in range(n):
        for j in range(n):
            layout_x.append(i * spacing)
            layout_y.append(j * spacing)
    
    # Set the layout
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
    
    # Set flow conditions
    fi.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    
    return fi


def generate_small_farm_blockage(save_path=None):
    """
    Generate horizontal cut planes showing blockage effects for a small wind farm (3x3).
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up small wind farm scenario (3x3)
    fi_base = setup_small_wind_farm(fi_base, layout="3x3")
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # Set up selected blockage models
    model_selection = ["none", "vortex_cylinder", "parametrized_global", "engineering_global"]
    
    model_configs = {}
    for model in model_selection:
        if model == "none":
            model_configs[model] = {
                "blockage_model": "none",
                "parameters": {}
            }
        elif model == "vortex_cylinder":
            model_configs[model] = {
                "blockage_model": "vortex_cylinder",
                "parameters": {
                    "include_ground_effect": False,
                    "finite_length": False
                }
            }
        elif model == "parametrized_global":
            model_configs[model] = {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.07,
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            }
        elif model == "engineering_global":
            model_configs[model] = {
                "blockage_model": "engineering_global",
                "parameters": {
                    "blockage_amplitude": 0.1,
                    "lateral_extent": 2.5,
                    "upstream_extent": 3.0,
                    "vertical_extent": 2.0
                }
            }
    
    fi_models = setup_blockage_models(fi_base, model_configs)
    
    # Calculate wake for each model
    for fi in fi_models.values():
        fi.calculate_wake()
    
    # Set up horizontal cut parameters
    # Cover the entire farm plus upstream area
    farm_size = 2 * 7 * D  # 2 spacings of 7D each
    x_range = (-7*D, farm_size + D)
    y_range = (-2*D, farm_size + 2*D)
    z_plane = fi_base.floris.farm.hub_heights[0][0]
    
    # Generate cut planes for each model
    horizontal_planes = {}
    for model_name, fi in fi_models.items():
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=150,
            y_resolution=150,
            x_min=x_range[0],
            x_max=x_range[1],
            y_min=y_range[0],
            y_max=y_range[1],
            z=z_plane
        )
        horizontal_plane.calculate_wake(fi)
        horizontal_planes[model_name] = horizontal_plane
    
    # Create subplots for comparison
    fig, axarr = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    axarr = axarr.flatten()
    
    # Calculate common velocity range for consistent colormaps
    vmin = min([hp.df.u.min() for hp in horizontal_planes.values()])
    vmax = max([hp.df.u.max() for hp in horizontal_planes.values()])
    
    # Plot each model
    for i, (model_name, hp) in enumerate(horizontal_planes.items()):
        ax = axarr[i]
        
        # Reshape data for contour plot
        x_mesh = hp.df.x.values.reshape(hp.resolution[1], hp.resolution[0])
        y_mesh = hp.df.y.values.reshape(hp.resolution[1], hp.resolution[0])
        u_mesh = hp.df.u.values.reshape(hp.resolution[1], hp.resolution[0])
        
        # Create the contour plot
        im = ax.contourf(
            x_mesh / D,  # Normalize by diameter
            y_mesh / D,
            u_mesh,
            levels=50,
            cmap=cm.coolwarm,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add turbine markers
        for x, y in zip(fi_base.layout_x, fi_base.layout_y):
            ax.plot(x/D, y/D, 'ko', markersize=5)
            circle = plt.Circle(
                (x/D, y/D),
                0.5,  # Radius is 0.5D
                fill=False,
                color='k'
            )
            ax.add_patch(circle)
        
        # Add title and labels
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.grid(True, alpha=0.3)
        
        # Add velocity deficit contour lines for blockage visualization
        freestream = fi_base.floris.flow_field.wind_speeds[0]
        deficit_levels = [0.5, 1.0, 2.0, 3.0]  # Percentage levels
        deficit_contours = ax.contour(
            x_mesh / D,
            y_mesh / D,
            100 * (freestream - u_mesh) / freestream,
            levels=deficit_levels,
            colors='k',
            linewidths=0.8,
            alpha=0.7
        )
        ax.clabel(deficit_contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.tolist())
    cbar.set_label('Wind Speed (m/s)')
    
    # Add overall title
    fig.suptitle("Blockage Effect: Horizontal Cut Plane for 3×3 Wind Farm", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()


def setup_large_wind_farm(fi_base):
    """
    Set up a large wind farm layout (10x10).
    
    Args:
        fi_base: Base FLORIS interface
        
    Returns:
        Updated FLORIS interface with large wind farm layout
    """
    fi = copy.deepcopy(fi_base)
    D = fi.floris.farm.rotor_diameters[0][0]
    
    # Create a 10x10 grid layout with 7D spacing
    n = 10
    spacing = 7 * D
    layout_x = []
    layout_y = []
    
    for i in range(n):
        for j in range(n):
            layout_x.append(i * spacing)
            layout_y.append(j * spacing)
    
    # Set the layout
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
    
    # Set flow conditions
    fi.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    
    return fi


def generate_large_farm_blockage(save_path=None):
    """
    Generate horizontal cut planes showing blockage effects for a large wind farm (10x10).
    This example uses only the global blockage models due to computational efficiency.
    
    Args:
        save_path: Path to save the figure
    """
    # Set up base model
    fi_base = setup_base_floris()
    
    # Set up large wind farm scenario (10x10)
    fi_base = setup_large_wind_farm(fi_base)
    D = fi_base.floris.farm.rotor_diameters[0][0]
    
    # For large farms, focus on global models for computational efficiency
    model_selection = ["none", "parametrized_global", "engineering_global"]
    
    model_configs = {}
    for model in model_selection:
        if model == "none":
            model_configs[model] = {
                "blockage_model": "none",
                "parameters": {}
            }
        elif model == "parametrized_global":
            model_configs[model] = {
                "blockage_model": "parametrized_global",
                "parameters": {
                    "blockage_intensity": 0.08,
                    "decay_constant": 3.0,
                    "boundary_layer_height": 500.0,
                    "porosity_coefficient": 0.7
                }
            }
        elif model == "engineering_global":
            model_configs[model] = {
                "blockage_model": "engineering_global",
                "parameters": {
                    "blockage_amplitude": 0.12,
                    "lateral_extent": 3.0,
                    "upstream_extent": 3.5,
                    "vertical_extent": 2.0
                }
            }
    
    fi_models = setup_blockage_models(fi_base, model_configs)
    
    # Calculate wake for each model
    for fi in fi_models.values():
        fi.calculate_wake()
    
    # Set up horizontal cut parameters
    # Cover the entire farm plus upstream area
    farm_size = 9 * 7 * D  # 9 spacings of 7D each
    x_range = (-10*D, farm_size + 5*D)
    y_range = (-5*D, farm_size + 5*D)
    z_plane = fi_base.floris.farm.hub_heights[0][0]
    
    # Generate cut planes for each model
    horizontal_planes = {}
    for model_name, fi in fi_models.items():
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=200,
            y_resolution=200,
            x_min=x_range[0],
            x_max=x_range[1],
            y_min=y_range[0],
            y_max=y_range[1],
            z=z_plane
        )
        horizontal_plane.calculate_wake(fi)
        horizontal_planes[model_name] = horizontal_plane
    
    # Create a figure focusing on comparing the global models
    fig, axarr = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    # Calculate common velocity range for consistent colormaps
    vmin = min([hp.df.u.min() for hp in horizontal_planes.values()])
    vmax = max([hp.df.u.max() for hp in horizontal_planes.values()])
    
    # Plot each model
    for i, (model_name, hp) in enumerate(horizontal_planes.items()):
        ax = axarr[i]
        
        # Reshape data for contour plot
        x_mesh = hp.df.x.values.reshape(hp.resolution[1], hp.resolution[0])
        y_mesh = hp.df.y.values.reshape(hp.resolution[1], hp.resolution[0])
        u_mesh = hp.df.u.values.reshape(hp.resolution[1], hp.resolution[0])
        
        # Create the contour plot
        im = ax.contourf(
            x_mesh / D,  # Normalize by diameter
            y_mesh / D,
            u_mesh,
            levels=50,
            cmap=cm.coolwarm,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add a simplified representation of turbines (just dots for large farm)
        for x, y in zip(fi_base.layout_x, fi_base.layout_y):
            ax.plot(x/D, y/D, 'ko', markersize=2)
        
        # Add title and labels
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("x/D")
        if i == 0:
            ax.set_ylabel("y/D")
        ax.grid(True, alpha=0.3)
        
        # For the global models, add velocity deficit contour lines
        if model_name != "none":
            freestream = fi_base.floris.flow_field.wind_speeds[0]
            deficit_levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]  # Percentage levels
            deficit_contours = ax.contour(
                x_mesh / D,
                y_mesh / D,
                100 * (freestream - u_mesh) / freestream,
                levels=deficit_levels,
                colors='k',
                linewidths=0.8,
                alpha=0.7
            )
            ax.clabel(deficit_contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.tolist())
    cbar.set_label('Wind Speed (m/s)')
    
    # Add overall title
    fig.suptitle("Blockage Effect: Horizontal Cut Plane for 10×10 Wind Farm", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.close()
