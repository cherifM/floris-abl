"""
Self-Similar Blockage Model.

A local blockage model that assumes self-similarity in the velocity deficit 
field, mentioned in contrast to the newer parametrized global blockage model.
"""

from typing import Any, Dict

import numpy as np
from attrs import define, field

from floris.core import BaseModel, Grid, FlowField, Farm
from floris.utilities import cosd, sind


@define
class SelfSimilarBlockage(BaseModel):
    """
    The Self-Similar Blockage Model assumes that the velocity deficit field
    due to blockage effects has a self-similar shape.
    
    This model calculates local blockage effects for each turbine and is mentioned
    in contrast to the newer parametrized global blockage model.

    Attributes:
        alpha (float): Self-similar shape parameter, controlling the deficit profile.
            Default value is 0.8.
        beta (float): Decay parameter controlling how quickly the blockage effects
            dissipate with distance. Default value is 2.0.
        delta_max (float): Maximum induction factor at the rotor plane. Default
            value is 0.2, representing a 20% maximum velocity deficit.
    """

    alpha: float = field(default=0.8)
    beta: float = field(default=2.0)
    delta_max: float = field(default=0.2)

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
            "alpha": self.alpha,
            "beta": self.beta,
            "delta_max": self.delta_max
        }

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
        alpha: float = None,
        beta: float = None,
        delta_max: float = None,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate the velocity deficit due to blockage using the self-similar model.

        This model assumes that the velocity deficit field due to blockage effects has
        a self-similar shape, which is modeled using a Gaussian-like profile.

        Args:
            x_i (np.ndarray): x-coordinates of evaluation points
            y_i (np.ndarray): y-coordinates of evaluation points
            z_i (np.ndarray): z-coordinates of evaluation points
            u_i (np.ndarray): flow speed at evaluation points
            v_i (np.ndarray): lateral flow at evaluation points
            ct_i (np.ndarray): thrust coefficient at current turbine
            grid (Grid): Grid object containing coordinates
            flow_field (FlowField): FlowField object with initial velocity field
            alpha (float): Self-similar shape parameter
            beta (float): Decay parameter
            delta_max (float): Maximum induction factor

        Returns:
            np.ndarray: Velocity deficit due to blockage
        """
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if delta_max is None:
            delta_max = self.delta_max
            
        # Get turbine geometry
        turbine_x = np.mean(grid.x_sorted, axis=(2, 3))
        turbine_y = np.mean(grid.y_sorted, axis=(2, 3))
        turbine_z = np.mean(grid.z_sorted, axis=(2, 3))
        
        # Get turbine parameters
        D = grid.turbine_map.turbines[0].rotor_diameter
        R = D / 2.0
        
        # Wind direction (assuming wind is along the positive x-axis as default)
        wind_directions = flow_field.wind_directions[:, None, None, None]
        
        # Initialize velocity deficit
        velocity_deficit = np.zeros_like(x_i)
        
        # Iterate through turbines
        n_turbines = grid.n_turbines
        for j in range(n_turbines):
            # Skip current turbine if it's the same as the evaluation point
            if np.allclose(x_i, turbine_x[:, j:j+1]):
                continue
                
            # Turbine coordinates
            x_t = turbine_x[:, j:j+1]
            y_t = turbine_y[:, j:j+1]
            z_t = turbine_z[:, j:j+1]
            
            # Calculate relative position to the turbine
            x_rel = x_i - x_t
            y_rel = y_i - y_t
            z_rel = z_i - z_t
            
            # Rotate coordinates to align with wind direction
            x_rot = x_rel * cosd(wind_directions) + y_rel * sind(wind_directions)
            y_rot = -x_rel * sind(wind_directions) + y_rel * cosd(wind_directions)
            z_rot = z_rel
            
            # Calculate distance to turbine in wind-aligned coordinates
            r_rot = np.sqrt(y_rot**2 + z_rot**2)  # radial distance
            
            # Calculate blockage effect only upstream of the turbine (x_rot < 0)
            upstream_mask = x_rot < 0
            
            if np.any(upstream_mask):
                # Normalized upstream distance
                x_norm = np.abs(x_rot[upstream_mask]) / D
                r_norm = r_rot[upstream_mask] / D
                
                # Calculate the induction factor at the rotor plane
                a_0 = 0.5 * (1 - np.sqrt(1 - ct_i[upstream_mask]))
                
                # Limit the induction factor to delta_max
                a_0 = np.minimum(a_0, delta_max)
                
                # Calculate radial profile based on self-similarity assumption
                # Using a Gaussian-like profile with width parameter alpha
                radial_profile = np.exp(-(r_norm / alpha)**2)
                
                # Calculate axial decay with distance upstream
                # Using a power law decay with exponent beta
                axial_decay = 1 / (1 + (x_norm)**beta)
                
                # Combined blockage effect
                blockage_effect = a_0 * radial_profile * axial_decay
                
                # Apply the velocity deficit
                velocity_deficit[upstream_mask] += blockage_effect * u_i[upstream_mask]
        
        return velocity_deficit
