"""
Engineering Model for Global Blockage with Wake Model.

Developed by Nygaard et al. (2020), this approach couples an engineering model
for global blockage with a modified wind turbine wake model to better represent
far wind farm or cluster wake effects.
"""

from typing import Any, Dict

import numpy as np
from attrs import define, field

from floris.core import BaseModel, Grid, FlowField, Farm
from floris.utilities import cosd, sind


@define
class EngineeringGlobalBlockage(BaseModel):
    """
    The Engineering Model for Global Blockage couples an engineering approach
    for blockage with a modified wind turbine wake model.
    
    Developed by Nygaard et al. (2020), this model:
    - Successfully predicts trends in power variation in the front row of turbines
    - Better represents far wind farm or cluster wake effects
    - Couples blockage effects with the wake model
    
    Attributes:
        blockage_amplitude (float): Controls the amplitude of the blockage effect.
            Default value is 0.1.
        lateral_extent (float): Controls the lateral extent of the blockage effect
            in rotor diameters. Default value is 2.5.
        upstream_extent (float): Controls how far upstream the blockage effect 
            extends in rotor diameters. Default value is 3.0.
        vertical_extent (float): Controls the vertical extent of the blockage 
            effect in rotor diameters. Default value is 2.0.
    """

    blockage_amplitude: float = field(default=0.1)
    lateral_extent: float = field(default=2.5)
    upstream_extent: float = field(default=3.0)
    vertical_extent: float = field(default=2.0)

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
        return {
            "grid": grid,
            "flow_field": flow_field,
            "blockage_amplitude": self.blockage_amplitude,
            "lateral_extent": self.lateral_extent,
            "upstream_extent": self.upstream_extent,
            "vertical_extent": self.vertical_extent
        }

    def _calculate_farm_area(self, turbine_x, turbine_y):
        """
        Calculate the area of the wind farm based on turbine positions.
        
        Args:
            turbine_x (np.ndarray): x-coordinates of turbines
            turbine_y (np.ndarray): y-coordinates of turbines
            
        Returns:
            float: Approximate area of the wind farm in square meters
        """
        # Simple approximation: area of the bounding box
        x_min, x_max = np.min(turbine_x), np.max(turbine_x)
        y_min, y_max = np.min(turbine_y), np.max(turbine_y)
        
        return (x_max - x_min) * (y_max - y_min)

    def _calculate_farm_density(self, farm_area, n_turbines, rotor_area):
        """
        Calculate the wind farm density.
        
        Args:
            farm_area (float): Area of the wind farm in square meters
            n_turbines (int): Number of turbines in the farm
            rotor_area (float): Rotor area in square meters
            
        Returns:
            float: Wind farm density (ratio of total rotor area to farm area)
        """
        total_rotor_area = n_turbines * rotor_area
        return total_rotor_area / farm_area

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
        blockage_amplitude: float = None,
        lateral_extent: float = None,
        upstream_extent: float = None,
        vertical_extent: float = None,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate the velocity deficit due to global blockage using the engineering model.

        This model uses a simplified engineering approach to model blockage effects
        and is designed to be coupled with wake models.

        Args:
            x_i (np.ndarray): x-coordinates of evaluation points
            y_i (np.ndarray): y-coordinates of evaluation points
            z_i (np.ndarray): z-coordinates of evaluation points
            u_i (np.ndarray): flow speed at evaluation points
            v_i (np.ndarray): lateral flow at evaluation points
            ct_i (np.ndarray): thrust coefficient at current turbine
            grid (Grid): Grid object containing coordinates
            flow_field (FlowField): FlowField object with initial velocity field
            blockage_amplitude (float): Amplitude of blockage effect
            lateral_extent (float): Lateral extent in rotor diameters
            upstream_extent (float): Upstream extent in rotor diameters
            vertical_extent (float): Vertical extent in rotor diameters

        Returns:
            np.ndarray: Velocity deficit due to blockage
        """
        if blockage_amplitude is None:
            blockage_amplitude = self.blockage_amplitude
        if lateral_extent is None:
            lateral_extent = self.lateral_extent
        if upstream_extent is None:
            upstream_extent = self.upstream_extent
        if vertical_extent is None:
            vertical_extent = self.vertical_extent
            
        # Get turbine geometry
        turbine_x = np.mean(grid.x_sorted, axis=(2, 3))
        turbine_y = np.mean(grid.y_sorted, axis=(2, 3))
        turbine_z = np.mean(grid.z_sorted, axis=(2, 3))
        
        # Get turbine parameters
        D = grid.turbine_map.turbines[0].rotor_diameter
        R = D / 2.0
        rotor_area = np.pi * R**2
        
        # Wind direction (assuming wind is along the positive x-axis as default)
        wind_directions = flow_field.wind_directions[:, None, None, None]
        
        # Calculate the farm properties
        n_turbines = grid.n_turbines
        farm_area = self._calculate_farm_area(turbine_x, turbine_y)
        farm_density = self._calculate_farm_density(farm_area, n_turbines, rotor_area)
        
        # Identify front-row turbines based on wind direction
        # For simplicity, we consider all turbines for global blockage
        
        # Calculate farm centroid
        farm_centroid_x = np.mean(turbine_x)
        farm_centroid_y = np.mean(turbine_y)
        
        # Calculate farm dimensions
        farm_width = np.max(turbine_y) - np.min(turbine_y)
        farm_length = np.max(turbine_x) - np.min(turbine_x)
        
        # Initialize velocity deficit
        velocity_deficit = np.zeros_like(x_i)
        
        # Calculate relative coordinates to farm centroid
        x_rel = x_i - farm_centroid_x
        y_rel = y_i - farm_centroid_y
        
        # Rotate coordinates to align with wind direction
        x_rot = x_rel * cosd(wind_directions) + y_rel * sind(wind_directions)
        y_rot = -x_rel * sind(wind_directions) + y_rel * cosd(wind_directions)
        
        # Calculate the global blockage effect
        # Only apply upstream of the farm
        upstream_mask = x_rot < 0
        
        if np.any(upstream_mask):
            # Calculate normalized distances
            x_norm = np.abs(x_rot[upstream_mask]) / D
            y_norm = np.abs(y_rot[upstream_mask]) / D
            z_norm = np.abs(z_i[upstream_mask] - np.mean(turbine_z)) / D
            
            # Calculate average thrust coefficient as indicator of blockage strength
            avg_ct = np.mean(ct_i)
            
            # Calculate the blockage based on farm density and thrust
            blockage_factor = blockage_amplitude * avg_ct * farm_density
            
            # Limit the blockage factor
            blockage_factor = np.minimum(blockage_factor, 0.4)
            
            # Calculate the decay with distance upstream
            # Using an exponential decay
            x_decay = np.exp(-x_norm / upstream_extent)
            
            # Calculate the lateral decay
            y_decay = np.exp(-(y_norm / lateral_extent)**2)
            
            # Calculate the vertical decay
            z_decay = np.exp(-(z_norm / vertical_extent)**2)
            
            # Combined blockage effect
            blockage_effect = blockage_factor * x_decay * y_decay * z_decay
            
            # Apply the velocity deficit
            velocity_deficit[upstream_mask] = blockage_effect * u_i[upstream_mask]
        
        return velocity_deficit
