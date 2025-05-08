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

# See https://floris.readthedocs.io for documentation

"""
This example demonstrates how to use the various blockage models implemented
in FLORIS. The example calculates the velocity deficit due to blockage for different
models and generates comparative visualizations.

The following blockage models are demonstrated:
- Parametrized Global Blockage Model (2025)
- Vortex Cylinder (VC) Model
- Mirrored Vortex Model
- Self-Similar Blockage Model
- Engineering Model for Global Blockage
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import floris.tools as wfct


def generate_velocity_cut_plane(fi, x_range, y_range, z_plane, y_resolution=100, x_resolution=100):
    """
    Generate a cut plane for visualization.
    """
    # Set the horizontal cut plane at specified height
    horizontal_plane = wfct.cut_plane.HorizontalGrid(
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        x_min=x_range[0],
        x_max=x_range[1],
        y_min=y_range[0],
        y_max=y_range[1],
        z=z_plane
    )

    # Calculate wake on the cut-through plane
    horizontal_plane.calculate_wake(fi)
    return horizontal_plane


def plot_results(horizontal_planes, titles, output_path=None):
    """
    Generate comparative plots for different blockage models.
    """
    n_plots = len(horizontal_planes)
    fig, axarr = plt.subplots(1, n_plots, figsize=(5*n_plots, 5), sharey=True)
    
    # Set common min/max values for the colormap
    min_speed = 1000.0
    max_speed = -1000.0
    
    for hp in horizontal_planes:
        min_speed = min(min_speed, np.min(hp.df.u))
        max_speed = max(max_speed, np.max(hp.df.u))
    
    for i, (hp, title) in enumerate(zip(horizontal_planes, titles)):
        ax = axarr[i] if n_plots > 1 else axarr
        
        # Create the contour plot
        im = ax.contourf(
            hp.df.x,
            hp.df.y,
            hp.df.u.reshape(hp.resolution[1], hp.resolution[0]),
            levels=np.linspace(min_speed, max_speed, 20),
            cmap=cm.coolwarm,
            vmin=min_speed,
            vmax=max_speed
        )
        
        # Add turbine markers
        for x, y in zip(fi.layout_x, fi.layout_y):
            ax.plot(x, y, 'ko', markersize=8)
            circle = plt.Circle(
                (x, y),
                fi.floris.farm.rotor_diameters[0][0]/2.0,
                fill=False,
                color='k'
            )
            ax.add_patch(circle)
        
        # Set the title and axis labels
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        if i == 0:
            ax.set_ylabel('y [m]')
        ax.axis('equal')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axarr.ravel().tolist() if n_plots > 1 else axarr)
    cbar.set_label('Wind Speed [m/s]')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Initialize FLORIS with default settings
    fi = wfct.floris_interface.FlorisInterface("../example_input_abl.yaml")
    
    # Set the wind farm layout - a simple row of turbines
    D = fi.floris.farm.rotor_diameters[0][0]  # Get the rotor diameter
    fi.reinitialize(
        layout_x=[0.0, 7*D, 14*D],  # Three turbines in a row
        layout_y=[0.0, 0.0, 0.0]
    )
    
    # Set the flow conditions
    fi.reinitialize(
        wind_speeds=[8.0],
        wind_directions=[270.0],  # Wind from the west (blowing towards the east/right)
        turbulence_intensities=[0.06]
    )
    
    # Make a copy of the baseline configuration (no blockage)
    fi_baseline = copy.deepcopy(fi)
    
    # A list to store all FlorisInterface instances with different blockage models
    fi_models = []
    model_names = []
    
    # 1. Parametrized Global Blockage Model
    fi_pgb = copy.deepcopy(fi)
    fi_pgb.set_wake_model(
        wake={"model_strings": {"velocity_model": "gauss", 
                               "deflection_model": "gauss", 
                               "combination_model": "sosfs",
                               "turbulence_model": "crespo_hernandez",
                               "blockage_model": "parametrized_global"},
              "enable_blockage": True,
              "wake_blockage_parameters": {"parametrized_global": {
                  "blockage_intensity": 0.05,
                  "decay_constant": 3.0,
                  "boundary_layer_height": 500.0,
                  "porosity_coefficient": 0.7
              }}
        }
    )
    fi_models.append(fi_pgb)
    model_names.append("Parametrized Global Blockage")
    
    # 2. Vortex Cylinder Model
    fi_vc = copy.deepcopy(fi)
    fi_vc.set_wake_model(
        wake={"model_strings": {"velocity_model": "gauss", 
                               "deflection_model": "gauss", 
                               "combination_model": "sosfs",
                               "turbulence_model": "crespo_hernandez",
                               "blockage_model": "vortex_cylinder"},
              "enable_blockage": True,
              "wake_blockage_parameters": {"vortex_cylinder": {
                  "include_ground_effect": False,
                  "finite_length": False,
                  "wake_length": 10.0
              }}
        }
    )
    fi_models.append(fi_vc)
    model_names.append("Vortex Cylinder")
    
    # 3. Mirrored Vortex Model
    fi_mv = copy.deepcopy(fi)
    fi_mv.set_wake_model(
        wake={"model_strings": {"velocity_model": "gauss", 
                               "deflection_model": "gauss", 
                               "combination_model": "sosfs",
                               "turbulence_model": "crespo_hernandez",
                               "blockage_model": "mirrored_vortex"},
              "enable_blockage": True,
              "wake_blockage_parameters": {"mirrored_vortex": {
                  "finite_length": False,
                  "wake_length": 10.0,
                  "mirror_weight": 1.0
              }}
        }
    )
    fi_models.append(fi_mv)
    model_names.append("Mirrored Vortex")
    
    # 4. Self-Similar Blockage Model
    fi_ss = copy.deepcopy(fi)
    fi_ss.set_wake_model(
        wake={"model_strings": {"velocity_model": "gauss", 
                               "deflection_model": "gauss", 
                               "combination_model": "sosfs",
                               "turbulence_model": "crespo_hernandez",
                               "blockage_model": "self_similar"},
              "enable_blockage": True,
              "wake_blockage_parameters": {"self_similar": {
                  "alpha": 0.8,
                  "beta": 2.0,
                  "delta_max": 0.2
              }}
        }
    )
    fi_models.append(fi_ss)
    model_names.append("Self-Similar")
    
    # 5. Engineering Global Blockage Model
    fi_egb = copy.deepcopy(fi)
    fi_egb.set_wake_model(
        wake={"model_strings": {"velocity_model": "gauss", 
                               "deflection_model": "gauss", 
                               "combination_model": "sosfs",
                               "turbulence_model": "crespo_hernandez",
                               "blockage_model": "engineering_global"},
              "enable_blockage": True,
              "wake_blockage_parameters": {"engineering_global": {
                  "blockage_amplitude": 0.1,
                  "lateral_extent": 2.5,
                  "upstream_extent": 3.0,
                  "vertical_extent": 2.0
              }}
        }
    )
    fi_models.append(fi_egb)
    model_names.append("Engineering Global")
    
    # Set up the visualization plane parameters
    x_range = [-5*D, 15*D]  # Include area upstream of turbines
    y_range = [-5*D, 5*D]   # Sufficient width to capture blockage effects
    z_plane = fi.floris.farm.hub_heights[0][0]  # At hub height
    
    # Calculate flow field for each model
    horizontal_planes = []
    
    # First, calculate the baseline (no blockage)
    fi_baseline.calculate_wake()
    hp_baseline = generate_velocity_cut_plane(fi_baseline, x_range, y_range, z_plane)
    horizontal_planes.append(hp_baseline)
    plot_titles = ["Baseline (No Blockage)"]
    
    # Then calculate for each blockage model
    for fi_model in fi_models:
        fi_model.calculate_wake()
        hp = generate_velocity_cut_plane(fi_model, x_range, y_range, z_plane)
        horizontal_planes.append(hp)
    
    # Add model names to plot titles
    plot_titles.extend(model_names)
    
    # Plot comparative results for all models
    plot_results(horizontal_planes, plot_titles, output_path="blockage_models_comparison.png")
    
    # Plot difference between each blockage model and baseline to show blockage effects more clearly
    diff_planes = []
    diff_titles = []
    
    for i, hp in enumerate(horizontal_planes[1:]):  # Skip baseline
        # Create a copy of the horizontal plane
        hp_diff = copy.deepcopy(hp)
        # Calculate the difference in velocity field (blockage model - baseline)
        hp_diff.df.u = hp.df.u - hp_baseline.df.u
        diff_planes.append(hp_diff)
        diff_titles.append(f"{model_names[i]} - Baseline")
    
    # Plot difference results
    plot_results(diff_planes, diff_titles, output_path="blockage_models_difference.png")
    
    # Calculate power output for each model to assess impact
    powers = []
    powers.append(fi_baseline.get_farm_power().sum())
    
    for fi_model, name in zip(fi_models, model_names):
        power = fi_model.get_farm_power().sum()
        powers.append(power)
        blockage_impact = (power - powers[0]) / powers[0] * 100
        print(f"{name} Model: Farm Power = {power/1000:.2f} kW, " 
              f"Difference from Baseline = {blockage_impact:.2f}%")
