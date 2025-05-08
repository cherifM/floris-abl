"""
Parametrized Global Blockage Model (2025).

This model represents the wind farm site as a parametrized porous object 
subjected to an ambient flow field, offering efficient energy production estimates
with lower computational requirements compared to local blockage models.
"""

from typing import Any, Dict

import numpy as np
from attrs import define, field

from floris.core import BaseModel, Grid, FlowField, Farm
from floris.utilities import cosd


@define
class ParametrizedGlobalBlockage(BaseModel):
    """
    The Parametrized Global Blockage Model (2025) represents a state-of-the-art approach
    to modeling blockage effects in offshore wind farms.
    
    This model treats the wind farm as a parametrized porous object subjected to an ambient flow field.
    It provides energy production estimates with significantly lower computational 
    requirements compared to local blockage models and demonstrates improved accuracy in
    turbine-level energy production prediction.

    Attributes:
        blockage_intensity (float): Intensity parameter controlling the strength of the blockage effect.
            Default value is 0.05, representing a 5% maximum velocity deficit.
        decay_constant (float): Controls the rate of decay of blockage effects with distance.
            Default value is 3.0.
        boundary_layer_height (float): Effective height of the atmospheric boundary layer,
            influences the vertical extent of blockage effects. Default value is 500.0 (meters).
        porosity_coefficient (float): Coefficient representing the porosity of the wind farm, 
            with 0 being fully blocked and 1 being completely porous. Default value is 0.7.
    """

    blockage_intensity: float = field(default=0.05)
    decay_constant: float = field(default=3.0)
    boundary_layer_height: float = field(default=500.0)
    porosity_coefficient: float = field(default=0.7)

    def __attrs_post_init__(self) -> None:
        """
        Initialize the model parameters.
        """
        pass

    def prepare_function(self, grid: Grid, flow_field: FlowField) -> dict:
        """
        Prepare the function by gathering necessary field data.

        Args:
            grid (Grid): Grid object containing coordinates for velocity calculation
            flow_field (FlowField): FlowField object with initial velocity field

        Returns:
            dict: Dictionary containing parameters needed for the blockage function
        """
        # Return the required parameters for the blockage function
        return {
            "grid": grid,
            "flow_field": flow_field,
            "blockage_intensity": self.blockage_intensity,
            "decay_constant": self.decay_constant,
            "boundary_layer_height": self.boundary_layer_height,
            "porosity_coefficient": self.porosity_coefficient
        }

    def _calculate_farm_bounding_box(self, grid, farm_x, farm_y):
        """
        Calculate the bounding box of the wind farm.
        
        Args:
            grid (Grid): Grid object containing coordinates
            farm_x (np.ndarray): x-coordinates of farm turbines
            farm_y (np.ndarray): y-coordinates of farm turbines
            
        Returns:
            tuple: (x_min, x_max, y_min, y_max) farm boundaries
        """
        x_min = np.min(farm_x)
        x_max = np.max(farm_x)
        y_min = np.min(farm_y)
        y_max = np.max(farm_y)
        
        # Add some margin to the bounding box
        margin = max(x_max - x_min, y_max - y_min) * 0.1
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        return x_min, x_max, y_min, y_max

    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        u_i: np.ndarray,
        v_i: np.ndarray,
        ct_i: np.ndarray,
        grid: Grid = None,
        flow_field: FlowField = None,
        blockage_intensity: float = None,
        decay_constant: float = None,
        boundary_layer_height: float = None,
        porosity_coefficient: float = None,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate the velocity deficit due to global blockage effects.

        The model creates a parametrized porous representation of the wind farm and calculates
        the resulting velocity deficit in the flow field, particularly upstream of the farm.

        Args:
            x_i (np.ndarray): x-coordinates of evaluation points
            y_i (np.ndarray): y-coordinates of evaluation points
            z_i (np.ndarray): z-coordinates of evaluation points
            u_i (np.ndarray): flow speed at evaluation points
            v_i (np.ndarray): lateral flow at evaluation points
            ct_i (np.ndarray): thrust coefficient at current turbine
            grid (Grid): Grid object containing coordinates
            flow_field (FlowField): FlowField object with initial velocity field
            blockage_intensity (float): Intensity of blockage effect
            decay_constant (float): Decay rate of blockage with distance
            boundary_layer_height (float): Height of boundary layer
            porosity_coefficient (float): Porosity of the wind farm

        Returns:
            np.ndarray: Velocity deficit due to blockage
        """
        if blockage_intensity is None:
            blockage_intensity = self.blockage_intensity
        if decay_constant is None:
            decay_constant = self.decay_constant
        if boundary_layer_height is None:
            boundary_layer_height = self.boundary_layer_height
        if porosity_coefficient is None:
            porosity_coefficient = self.porosity_coefficient
            
        # Get wind farm geometry
        farm_x = np.mean(grid.x_sorted, axis=(2, 3))
        farm_y = np.mean(grid.y_sorted, axis=(2, 3))
        
        # Calculate farm bounding box
        x_min, x_max, y_min, y_max = self._calculate_farm_bounding_box(grid, farm_x, farm_y)
        
        # Farm centroid
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Farm dimensions
        farm_width = x_max - x_min
        farm_length = y_max - y_min
        
        # Wind direction (assuming wind is along the positive x-axis as default)
        wind_directions = flow_field.wind_directions[:, None, None, None]
        
        # Rotate coordinates to align with wind direction
        x_rel = x_i - x_center
        y_rel = y_i - y_center
        
        # Calculate rotated coordinates
        x_rot = x_rel * cosd(wind_directions) + y_rel * sind(wind_directions)
        y_rot = -x_rel * sind(wind_directions) + y_rel * cosd(wind_directions)
        
        # Calculate distance to farm in wind-aligned coordinates
        # Negative values are upstream, positive values are downstream
        x_dist = x_rot - (-farm_length / 2)  # Distance to upstream farm boundary
        
        # Calculate blockage effect only upstream of the farm (x_dist < 0)
        upstream_mask = x_dist < 0
        
        # Initialize velocity deficit to zeros
        velocity_deficit = np.zeros_like(x_i)
        
        # Calculate velocity deficit due to blockage for upstream points
        if np.any(upstream_mask):
            # Normalized distance upstream (positive values)
            norm_dist = np.abs(x_dist[upstream_mask]) / farm_length
            
            # Height factor (decreases effect with height)
            height_factor = np.exp(-z_i[upstream_mask] / boundary_layer_height)
            
            # Lateral decay (gaussian-like)
            lateral_dist = np.abs(y_rot[upstream_mask]) / (farm_width / 2)
            lateral_factor = np.exp(-(lateral_dist**2))
            
            # Calculate the upstream decay
            upstream_decay = np.exp(-decay_constant * norm_dist)
            
            # Combine all factors to get the velocity deficit
            farm_thrust = np.mean(ct_i) * porosity_coefficient
            deficit_factor = blockage_intensity * farm_thrust * upstream_decay * lateral_factor * height_factor
            
            # Apply the velocity deficit
            velocity_deficit[upstream_mask] = deficit_factor * u_i[upstream_mask]
        
        return velocity_deficit
