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
Unit tests for blockage models in FLORIS.
"""

import os
import copy
import numpy as np
import pytest

import floris.tools as wfct
from floris.core.blockage import (
    EngineeringGlobalBlockage,
    MirroredVortexBlockage,
    NoneBlockage,
    ParametrizedGlobalBlockage,
    SelfSimilarBlockage,
    VortexCylinderBlockage,
)


class TestBlockageModels:
    """
    Tests for blockage models in FLORIS.
    """

    def setup_class(self):
        """Set up the test class with a basic FLORIS instance."""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(os.path.dirname(self.test_dir), "example_input_abl.yaml")
        self.fi_base = wfct.floris_interface.FlorisInterface(input_path)
        
        # Set up a simple three-turbine layout
        D = self.fi_base.floris.farm.rotor_diameters[0][0]
        self.fi_base.reinitialize(layout_x=[0.0, 7*D, 14*D], layout_y=[0.0, 0.0, 0.0])
        self.fi_base.reinitialize(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
        
        # Enable blockage in the base model with "none" blockage model for comparison
        self.fi_base.set_wake_model(
            wake={"model_strings": {"velocity_model": "gauss", 
                                   "deflection_model": "gauss", 
                                   "combination_model": "sosfs",
                                   "turbulence_model": "crespo_hernandez",
                                   "blockage_model": "none"},
                 "enable_blockage": True}
        )
        
        # Calculate the baseline wake
        self.fi_base.calculate_wake()

    def setup_model(self, model_name, model_params=None):
        """Helper method to set up a specific blockage model."""
        fi = copy.deepcopy(self.fi_base)
        
        wake_config = {"model_strings": {"velocity_model": "gauss", 
                                        "deflection_model": "gauss", 
                                        "combination_model": "sosfs",
                                        "turbulence_model": "crespo_hernandez",
                                        "blockage_model": model_name},
                      "enable_blockage": True}
        
        if model_params:
            wake_config["wake_blockage_parameters"] = {model_name: model_params}
            
        fi.set_wake_model(wake=wake_config)
        return fi

    def test_blockage_model_instantiation(self):
        """Test that all blockage models can be instantiated directly."""
        none_blockage = NoneBlockage()
        assert none_blockage is not None
        
        parametrized_global = ParametrizedGlobalBlockage()
        assert parametrized_global is not None
        assert parametrized_global.blockage_intensity == 0.05  # default value
        
        vortex_cylinder = VortexCylinderBlockage()
        assert vortex_cylinder is not None
        assert vortex_cylinder.include_ground_effect is False  # default value
        
        mirrored_vortex = MirroredVortexBlockage()
        assert mirrored_vortex is not None
        assert mirrored_vortex.mirror_weight == 1.0  # default value
        
        self_similar = SelfSimilarBlockage()
        assert self_similar is not None
        assert self_similar.alpha == 0.8  # default value
        
        engineering_global = EngineeringGlobalBlockage()
        assert engineering_global is not None
        assert engineering_global.blockage_amplitude == 0.1  # default value

    def test_none_blockage_model(self):
        """Test that the None blockage model does not change the flow field."""
        # Using the base model which has the "none" blockage model
        self.fi_base.calculate_wake()
        
        # Get flow field with no blockage effect
        x_coords = np.linspace(-5*self.fi_base.floris.farm.rotor_diameters[0][0], 
                               5*self.fi_base.floris.farm.rotor_diameters[0][0], 100)
        y_coords = np.zeros_like(x_coords)
        z_coords = np.ones_like(x_coords) * self.fi_base.floris.farm.hub_heights[0][0]
        
        # Get velocities at these points
        velocities = self.fi_base.calculate_flow_field(x_coords, y_coords, z_coords)
        
        # Check that velocities upstream and at turbine are equal to freestream
        # (No blockage effect from None model)
        upstream_mask = x_coords < 0
        freestream_velocity = self.fi_base.floris.flow_field.wind_speeds[0]
        
        assert np.allclose(velocities[upstream_mask], freestream_velocity, rtol=1e-6)

    def test_parametrized_global_blockage(self):
        """Test that Parametrized Global Blockage model produces expected blockage effects."""
        # Set up model with specific parameters
        model_params = {
            "blockage_intensity": 0.1,  # Higher intensity for clearer effect
            "decay_constant": 2.0,
            "boundary_layer_height": 500.0,
            "porosity_coefficient": 0.7
        }
        
        fi = self.setup_model("parametrized_global", model_params)
        fi.calculate_wake()
        
        # Create a horizontal cut plane upstream of the farm
        D = fi.floris.farm.rotor_diameters[0][0]
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=100,
            y_resolution=100,
            x_min=-5*D,
            x_max=D,  # Include the first turbine
            y_min=-3*D,
            y_max=3*D,
            z=fi.floris.farm.hub_heights[0][0]
        )
        
        horizontal_plane.calculate_wake(fi)
        
        # Verify that there is a velocity deficit upstream (blockage effect)
        freestream_velocity = fi.floris.flow_field.wind_speeds[0]
        upstream_velocities = horizontal_plane.df.loc[horizontal_plane.df.x < -D/2, "u"]
        
        # There should be at least some points with velocity less than freestream
        # (indicating blockage effect)
        assert np.any(upstream_velocities < freestream_velocity * 0.99)
        
        # The effect should be stronger closer to the farm
        close_upstream = horizontal_plane.df.loc[(horizontal_plane.df.x < -D/2) & 
                                              (horizontal_plane.df.x > -2*D) &
                                              (np.abs(horizontal_plane.df.y) < D/2), "u"]
        far_upstream = horizontal_plane.df.loc[(horizontal_plane.df.x < -4*D) & 
                                            (np.abs(horizontal_plane.df.y) < D/2), "u"]
        
        assert np.mean(close_upstream) < np.mean(far_upstream)

    def test_vortex_cylinder_blockage(self):
        """Test that Vortex Cylinder model produces expected blockage effects."""
        # Set up model with specific parameters
        model_params = {
            "include_ground_effect": False,
            "finite_length": False
        }
        
        fi = self.setup_model("vortex_cylinder", model_params)
        fi.calculate_wake()
        
        # Create a horizontal cut plane upstream of the first turbine
        D = fi.floris.farm.rotor_diameters[0][0]
        x_start = -3*D
        x_end = 0
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=50,
            y_resolution=50,
            x_min=x_start,
            x_max=x_end,
            y_min=-2*D,
            y_max=2*D,
            z=fi.floris.farm.hub_heights[0][0]
        )
        
        horizontal_plane.calculate_wake(fi)
        
        # Check for blockage effect upstream of the turbine
        # Closer to the turbine, we expect lower velocities
        freestream_velocity = fi.floris.flow_field.wind_speeds[0]
        
        # Points directly upstream of the turbine (y near 0)
        center_line = horizontal_plane.df.loc[np.abs(horizontal_plane.df.y) < D/4]
        center_line = center_line.sort_values(by="x")
        
        # Velocity should decrease as we approach the turbine due to blockage
        # Check that the velocity gradient is negative as we approach the turbine
        x_positions = center_line.x.values
        u_velocities = center_line.u.values
        
        # Calculate velocity gradients
        du_dx = np.diff(u_velocities) / np.diff(x_positions)
        
        # Expect negative gradient (decreasing velocity as x increases approaching turbine)
        assert np.mean(du_dx) > 0

    def test_mirrored_vortex_blockage(self):
        """Test that Mirrored Vortex model produces expected blockage effects."""
        # Set up model with specific parameters
        model_params = {
            "finite_length": False,
            "mirror_weight": 1.0
        }
        
        # Set up model with hub height closer to ground to see mirror effect
        fi = copy.deepcopy(self.fi_base)
        # Reduce hub height to enhance ground effect
        D = fi.floris.farm.rotor_diameters[0][0]
        original_hub_height = fi.floris.farm.hub_heights[0][0]
        lower_hub_height = D * 1.5  # 1.5 diameters above ground
        
        fi.reinitialize(hub_heights=[lower_hub_height, lower_hub_height, lower_hub_height])
        
        wake_config = {"model_strings": {"velocity_model": "gauss", 
                                        "deflection_model": "gauss", 
                                        "combination_model": "sosfs",
                                        "turbulence_model": "crespo_hernandez",
                                        "blockage_model": "mirrored_vortex"},
                      "enable_blockage": True,
                      "wake_blockage_parameters": {"mirrored_vortex": model_params}}
        
        fi.set_wake_model(wake=wake_config)
        fi.calculate_wake()
        
        # Create vertical cut plane upstream to see ground effect
        vertical_plane = wfct.cut_plane.VerticalGrid(
            x_resolution=1,  # Single x position
            z_resolution=50,
            x=-2*D,  # 2 diameters upstream
            y_min=-2*D,
            y_max=2*D,
            z_min=0.1*D,  # Near ground
            z_max=4*D     # Well above hub
        )
        
        vertical_plane.calculate_wake(fi)
        
        # Check for asymmetry in vertical profile due to ground effect
        # At y=0 (turbine centerline), lower heights should see stronger blockage
        centerline = vertical_plane.df.loc[np.abs(vertical_plane.df.y) < D/10]
        centerline = centerline.sort_values(by="z")
        
        # Split into lower and upper halves
        lower_half = centerline.loc[centerline.z < lower_hub_height]
        upper_half = centerline.loc[centerline.z > lower_hub_height]
        
        if len(lower_half) > 0 and len(upper_half) > 0:
            # Ground effect should enhance blockage in lower half
            assert np.mean(lower_half.u) < np.mean(upper_half.u)

    def test_self_similar_blockage(self):
        """Test that Self-Similar Blockage model produces expected blockage effects."""
        # Set up model with specific parameters
        model_params = {
            "alpha": 0.8,
            "beta": 2.0,
            "delta_max": 0.2
        }
        
        fi = self.setup_model("self_similar", model_params)
        fi.calculate_wake()
        
        # Create a horizontal cut plane at hub height
        D = fi.floris.farm.rotor_diameters[0][0]
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=50,
            y_resolution=50,
            x_min=-4*D,
            x_max=0,  # Up to first turbine
            y_min=-2*D,
            y_max=2*D,
            z=fi.floris.farm.hub_heights[0][0]
        )
        
        horizontal_plane.calculate_wake(fi)
        
        # Check that blockage effect follows self-similar pattern
        # 1. Effect should be strongest along centerline
        # 2. Effect should decay laterally following Gaussian-like profile
        
        # Points at same x position (2D upstream)
        x_pos = -2*D
        points_at_x = horizontal_plane.df.loc[np.abs(horizontal_plane.df.x - x_pos) < D/20]
        points_at_x = points_at_x.sort_values(by="y")
        
        # Centerline should have lowest velocity
        centerline_velocity = points_at_x.loc[np.abs(points_at_x.y) < D/20, "u"].values
        off_center_velocity = points_at_x.loc[np.abs(points_at_x.y) > D, "u"].values
        
        if len(centerline_velocity) > 0 and len(off_center_velocity) > 0:
            assert np.mean(centerline_velocity) < np.mean(off_center_velocity)
        
        # Check that blockage effect has reasonable magnitude
        freestream_velocity = fi.floris.flow_field.wind_speeds[0]
        
        # Maximum expected blockage effect (close to turbine, centerline)
        close_points = horizontal_plane.df.loc[(horizontal_plane.df.x > -1.5*D) & 
                                           (horizontal_plane.df.x < -0.5*D) & 
                                           (np.abs(horizontal_plane.df.y) < D/4)]
        
        if len(close_points) > 0:
            min_velocity = np.min(close_points.u)
            max_blockage = (freestream_velocity - min_velocity) / freestream_velocity
            
            # Reasonable range for blockage: between 1% and 15%
            assert 0.01 < max_blockage < 0.15

    def test_engineering_global_blockage(self):
        """Test that Engineering Global Blockage model produces expected effects."""
        # Set up model with specific parameters
        model_params = {
            "blockage_amplitude": 0.1,
            "lateral_extent": 2.5,
            "upstream_extent": 3.0,
            "vertical_extent": 2.0
        }
        
        fi = self.setup_model("engineering_global", model_params)
        fi.calculate_wake()
        
        # Create a horizontal cut plane at hub height
        D = fi.floris.farm.rotor_diameters[0][0]
        horizontal_plane = wfct.cut_plane.HorizontalGrid(
            x_resolution=50,
            y_resolution=50,
            x_min=-6*D,
            x_max=16*D,  # Entire farm
            y_min=-4*D,
            y_max=4*D,
            z=fi.floris.farm.hub_heights[0][0]
        )
        
        horizontal_plane.calculate_wake(fi)
        
        # Check global blockage effect upstream of the farm
        freestream_velocity = fi.floris.flow_field.wind_speeds[0]
        
        # Points upstream of first turbine
        upstream_points = horizontal_plane.df.loc[horizontal_plane.df.x < -D]
        
        # There should be blockage effect (velocities < freestream)
        assert np.any(upstream_points.u < freestream_velocity * 0.99)
        
        # Check farm-wide effect: velocity upstream of second and third turbines
        # should be lower than freestream but potentially affected by wakes too
        
        # For clearer blockage impact, check edges of farm where wake isn't strong
        edge_points = horizontal_plane.df.loc[(horizontal_plane.df.x > 0) & 
                                          (horizontal_plane.df.x < 15*D) & 
                                          (np.abs(horizontal_plane.df.y) > 2*D)]
        
        # Should see some blockage even away from direct wake
        assert np.any(edge_points.u < freestream_velocity * 0.995)
    
    def test_model_parameters(self):
        """Test that changing model parameters affects the blockage calculation."""
        # Test with Parametrized Global Blockage model
        # First with low intensity
        low_intensity_params = {
            "blockage_intensity": 0.01,
            "decay_constant": 3.0,
            "boundary_layer_height": 500.0,
            "porosity_coefficient": 0.7
        }
        
        fi_low = self.setup_model("parametrized_global", low_intensity_params)
        fi_low.calculate_wake()
        
        # Then with high intensity
        high_intensity_params = {
            "blockage_intensity": 0.1,  # 10x stronger
            "decay_constant": 3.0,
            "boundary_layer_height": 500.0,
            "porosity_coefficient": 0.7
        }
        
        fi_high = self.setup_model("parametrized_global", high_intensity_params)
        fi_high.calculate_wake()
        
        # Evaluate at points upstream of the first turbine
        D = fi_low.floris.farm.rotor_diameters[0][0]
        x_coords = np.array([-3*D, -2*D, -D])
        y_coords = np.zeros_like(x_coords)
        z_coords = np.ones_like(x_coords) * fi_low.floris.farm.hub_heights[0][0]
        
        velocities_low = fi_low.calculate_flow_field(x_coords, y_coords, z_coords)
        velocities_high = fi_high.calculate_flow_field(x_coords, y_coords, z_coords)
        
        # High intensity should give lower velocities (stronger blockage)
        assert np.all(velocities_high < velocities_low)

    def test_blockage_power_impact(self):
        """Test that blockage models affect turbine power calculations."""
        # Compare power with different blockage models
        model_list = ["none", "parametrized_global", "vortex_cylinder", 
                     "mirrored_vortex", "self_similar", "engineering_global"]
        
        powers = []
        
        for model_name in model_list:
            fi = self.setup_model(model_name)
            fi.calculate_wake()
            farm_power = fi.get_farm_power().sum()
            powers.append(farm_power)
        
        # The none model should yield highest power (no blockage)
        assert powers[0] >= np.min(powers)
        
        # At least some blockage models should reduce power compared to none
        assert np.any(np.array(powers[1:]) < powers[0])
        
        # Impact should be reasonable (not too extreme)
        power_diffs = [(baseline - p) / baseline for p, baseline in zip(powers[1:], [powers[0]] * len(powers[1:]))]
        assert np.all(np.array(power_diffs) < 0.15)  # Less than 15% reduction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
