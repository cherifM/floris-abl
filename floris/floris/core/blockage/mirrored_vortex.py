"""
Mirrored Vortex Blockage Model.

An extension of the vortex cylinder approach specifically addressing ground effects
by using a "mirror" vorticity distribution placed symmetrically with respect to the ground.
"""

from typing import Any, Dict

import numpy as np
from attrs import define, field
from scipy.special import ellipk, ellipe

from floris.core import BaseModel, Grid, FlowField, Farm
from floris.utilities import cosd, sind


@define
class MirroredVortexBlockage(BaseModel):
    """
    The Mirrored Vortex Blockage Model extends the vortex cylinder approach to specifically
    address ground effects.
    
    This model:
    - Uses a "mirror" vorticity distribution placed symmetrically with respect to the ground
    - Calculates velocity fields by summing contributions from two elementary models mirrored across the ground plane
    - Creates greater induction in front of the rotor while satisfying slip conditions at the ground
    
    Attributes:
        finite_length (bool): Whether to model a finite wake length. Default is False.
        wake_length (float): Length of the wake in rotor diameters, only used when finite_length is True.
            Default value is 10.0.
        mirror_weight (float): Weight factor for the mirrored vortex contribution. Default is 1.0.
    """

    finite_length: bool = field(default=False)
    wake_length: float = field(default=10.0)
    mirror_weight: float = field(default=1.0)

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
            "finite_length": self.finite_length,
            "wake_length": self.wake_length,
            "mirror_weight": self.mirror_weight
        }

    def _vortex_cylinder_induced_velocity(
        self,
        x: np.ndarray,
        r: np.ndarray,
        gamma_t: np.ndarray,
        R: float,
        finite_length: bool = False,
        wake_length: float = 10.0
    ) -> tuple:
        """
        Calculate induced velocity from a vortex cylinder.

        Args:
            x (np.ndarray): Axial position relative to cylinder origin
            r (np.ndarray): Radial position relative to cylinder axis
            gamma_t (np.ndarray): Tangential vorticity strength
            R (float): Cylinder radius
            finite_length (bool): If True, model finite wake length
            wake_length (float): Length of wake in rotor diameters

        Returns:
            tuple: (u_ind, v_ind) induced velocities in axial and radial directions
        """
        # Initialize induced velocities
        u_ind = np.zeros_like(x)
        v_ind = np.zeros_like(x)
        
        # Points inside or outside the cylinder
        mask_in = r <= R
        mask_out = ~mask_in
        
        # Semi-infinite vortex cylinder
        if not finite_length:
            # Points outside the cylinder
            if np.any(mask_out):
                r_out = r[mask_out]
                x_out = x[mask_out]
                
                # Dimensionless parameters
                m_out = 4 * R * r_out / ((R + r_out)**2 + x_out**2)
                
                # Complete elliptic integrals
                K_out = ellipk(m_out)
                E_out = ellipe(m_out)
                
                # Induced velocities
                u_ind[mask_out] = gamma_t[mask_out] * R / (2 * np.pi) * m_out / r_out * (
                    x_out / np.sqrt((R + r_out)**2 + x_out**2) * (K_out - E_out)
                )
                
                v_ind[mask_out] = gamma_t[mask_out] * R / (2 * np.pi) * m_out / r_out * (
                    ((R**2 - r_out**2) / ((R + r_out)**2 + x_out**2) + 1) * K_out - E_out
                )
            
            # Points inside the cylinder
            if np.any(mask_in):
                r_in = r[mask_in]
                x_in = x[mask_in]
                r_in = np.maximum(r_in, 1e-10)  # Avoid division by zero
                
                # Dimensionless parameters
                m_in = 4 * R * r_in / ((R + r_in)**2 + x_in**2)
                
                # Complete elliptic integrals
                K_in = ellipk(m_in)
                E_in = ellipe(m_in)
                
                # Induced velocities
                u_ind[mask_in] = gamma_t[mask_in] * R / (2 * np.pi) * m_in / r_in * (
                    x_in / np.sqrt((R + r_in)**2 + x_in**2) * (K_in - E_in)
                )
                
                v_ind[mask_in] = gamma_t[mask_in] * R / (2 * np.pi) * m_in / r_in * (
                    -K_in + E_in * (R**2 + r_in**2 + x_in**2) / ((R - r_in)**2 + x_in**2)
                )
        else:
            # Finite length vortex cylinder
            # Calculate both the semi-infinite cylinder and the mirrored one at wake_length*R
            x_mirror = x - wake_length * 2 * R
            
            # Calculate induced velocities for the main cylinder
            u_main, v_main = self._vortex_cylinder_induced_velocity(
                x, r, gamma_t, R, finite_length=False
            )
            
            # Calculate induced velocities for the mirrored cylinder (with negative strength)
            u_mirror, v_mirror = self._vortex_cylinder_induced_velocity(
                x_mirror, r, -gamma_t, R, finite_length=False
            )
            
            # Combine the two contributions
            u_ind = u_main + u_mirror
            v_ind = v_main + v_mirror
        
        return u_ind, v_ind

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
        finite_length: bool = None,
        wake_length: float = None,
        mirror_weight: float = None,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate the velocity deficit due to blockage using the mirrored vortex model.

        This model uses a "mirror" vorticity distribution placed symmetrically with respect 
        to the ground to account for ground effects.

        Args:
            x_i (np.ndarray): x-coordinates of evaluation points
            y_i (np.ndarray): y-coordinates of evaluation points
            z_i (np.ndarray): z-coordinates of evaluation points
            u_i (np.ndarray): flow speed at evaluation points
            v_i (np.ndarray): lateral flow at evaluation points
            ct_i (np.ndarray): thrust coefficient at current turbine
            grid (Grid): Grid object containing coordinates
            flow_field (FlowField): FlowField object with initial velocity field
            finite_length (bool): Whether to model a finite wake length
            wake_length (float): Length of wake in rotor diameters
            mirror_weight (float): Weight factor for the mirrored vortex contribution

        Returns:
            np.ndarray: Velocity deficit due to blockage
        """
        if finite_length is None:
            finite_length = self.finite_length
        if wake_length is None:
            wake_length = self.wake_length
        if mirror_weight is None:
            mirror_weight = self.mirror_weight
            
        # Get turbine geometry
        turbine_x = np.mean(grid.x_sorted, axis=(2, 3))
        turbine_y = np.mean(grid.y_sorted, axis=(2, 3))
        turbine_z = np.mean(grid.z_sorted, axis=(2, 3))
        
        # Get turbine parameters
        D = grid.turbine_map.turbines[0].rotor_diameter
        R = D / 2.0
        
        # Wind direction (assuming wind is along the positive x-axis as default)
        wind_directions = flow_field.wind_directions[:, None, None, None]
        
        # Calculate tangential vorticity strength from thrust coefficient
        # gamma_t = -0.5 * U_inf * Ct / R (negative for induction)
        gamma_t = -0.5 * u_i * ct_i / R
        
        # Initialize velocity deficit
        velocity_deficit = np.zeros_like(x_i)
        
        # Iterate through turbines
        n_turbines = grid.n_turbines
        for j in range(n_turbines):
            # Skip current turbine if it's the same as the evaluation point
            # (we're calculating blockage from other turbines on the current point)
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
            
            # Calculate radial distance to cylinder axis
            r = np.sqrt(y_rot**2 + z_rot**2)
            
            # Calculate induced velocity components from the primary vortex cylinder
            u_ind, v_rad = self._vortex_cylinder_induced_velocity(
                x_rot, r, gamma_t, R, finite_length, wake_length * D
            )
            
            # Mirror the vortex cylinder below the ground
            # Z_ground is assumed to be at z=0
            z_mirror = -z_t
            
            # Calculate relative position to the mirror turbine
            z_rel_mirror = z_i - z_mirror
            
            # Rotated coordinates for the mirror
            z_rot_mirror = z_rel_mirror
            
            # Calculate radial distance to mirror cylinder axis
            r_mirror = np.sqrt(y_rot**2 + z_rot_mirror**2)
            
            # Calculate induced velocity components from mirror vortex cylinder
            u_ind_mirror, v_rad_mirror = self._vortex_cylinder_induced_velocity(
                x_rot, r_mirror, gamma_t, R, finite_length, wake_length * D
            )
            
            # Apply mirroring principle: u-component adds, v-component (normal to ground) cancels at ground
            # The z-component of v_rad cancels out at the ground plane to satisfy slip condition
            
            # Extract y and z components of the radial velocity
            # (we need to project v_rad into y and z components)
            r_non_zero = r > 0
            v_y = np.zeros_like(v_rad)
            v_z = np.zeros_like(v_rad)
            v_y_mirror = np.zeros_like(v_rad_mirror)
            v_z_mirror = np.zeros_like(v_rad_mirror)
            
            if np.any(r_non_zero):
                v_y[r_non_zero] = v_rad[r_non_zero] * y_rot[r_non_zero] / r[r_non_zero]
                v_z[r_non_zero] = v_rad[r_non_zero] * z_rot[r_non_zero] / r[r_non_zero]
            
            r_mirror_non_zero = r_mirror > 0
            if np.any(r_mirror_non_zero):
                v_y_mirror[r_mirror_non_zero] = v_rad_mirror[r_mirror_non_zero] * y_rot[r_mirror_non_zero] / r_mirror[r_mirror_non_zero]
                v_z_mirror[r_mirror_non_zero] = v_rad_mirror[r_mirror_non_zero] * z_rot_mirror[r_mirror_non_zero] / r_mirror[r_mirror_non_zero]
            
            # Mirroring principle: u and v_y add, v_z cancels
            u_combined = u_ind + mirror_weight * u_ind_mirror
            
            # Add the induced velocity to the total deficit
            velocity_deficit += u_combined
        
        return velocity_deficit
