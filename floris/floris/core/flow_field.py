from __future__ import annotations

import copy

import attrs
import matplotlib.path as mpltPath
import numpy as np
from attrs import define, field
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

from floris.core import (
    BaseClass,
    Grid,
)
from floris.type_dec import (
    floris_array_converter,
    NDArrayFloat,
    NDArrayObject,
)

# Added for stability implementation
KAPPA = 0.4 # von Karman constant
OMEGA_EARTH = 7.2921e-5  # Earth's rotation rate in rad/s

def phi_m(zeta: NDArrayFloat) -> NDArrayFloat:
    """
    Calculates the Monin-Obukhov stability correction function for momentum (phi_m).
    Uses Dyer (1974) forms.

    Args:
        zeta (NDArrayFloat): Stability parameter (z / L).

    Returns:
        NDArrayFloat: Value of phi_m.
    """
    phi = np.ones_like(zeta)
    stable = zeta >= 0
    unstable = ~stable

    # Stable case (phi_m = 1 + 5*zeta)
    phi[stable] = 1 + 5 * zeta[stable]

    # Unstable case (phi_m = (1 - 16*zeta)^(-1/4))
    # Calculate x for potentially unstable points
    zeta_unstable = zeta[unstable]
    # Ensure argument for sqrt is non-negative (should be for zeta < 0)
    arg_x_sq = np.maximum(0.0, 1 - 16 * zeta_unstable)
    x = np.sqrt(np.sqrt(arg_x_sq)) # Equivalent to **0.25

    # Calculate phi_m for these unstable points
    # Avoid division by zero if x is zero (highly unstable)
    x_safe = np.maximum(x, 1e-9)
    phi_unstable_values = 1.0 / x_safe

    # Assign the calculated values back to the original phi array using the unstable mask
    phi[unstable] = phi_unstable_values

    return phi

def psi_m(zeta: NDArrayFloat) -> NDArrayFloat:
    """
    Calculates the integrated Monin-Obukhov stability correction function for momentum (psi_m).
    Uses Dyer (1974) forms integrated.

    Args:
        zeta (NDArrayFloat): Stability parameter (z / L).

    Returns:
        NDArrayFloat: Value of psi_m.
    """
    psi = np.zeros_like(zeta)
    stable = zeta >= 0
    unstable = ~stable

    # Stable case (psi_m = -5*zeta)
    psi[stable] = -5 * zeta[stable]

    # Unstable case (psi_m = 2*ln((1+x)/2) + ln((1+x^2)/2) - 2*atan(x) + pi/2)
    # where x = (1-16*zeta)^0.25
    # Calculate x for all potentially unstable points first
    zeta_unstable = zeta[unstable]
    # Ensure argument for sqrt is non-negative (should be for zeta < 0)
    arg_x_sq = np.maximum(0.0, 1 - 16 * zeta_unstable)
    x = np.sqrt(np.sqrt(arg_x_sq)) # Equivalent to **0.25

    # Calculate psi_m for these unstable points
    # Handle potential log(0) or division by zero if x approaches 0 (very unstable)
    x_safe = np.maximum(x, 1e-9) # Avoid issues at x=0
    term1 = 2 * np.log((1 + x_safe) / 2)
    term2 = np.log((1 + x_safe**2) / 2)
    term3 = -2 * np.arctan(x_safe)
    psi_unstable_values = term1 + term2 + term3 + np.pi / 2

    # Assign the calculated values back to the original psi array using the unstable mask
    psi[unstable] = psi_unstable_values

    return psi


@define
class FlowField(BaseClass):
    # Mandatory attributes first
    wind_speeds: NDArrayFloat = field(converter=floris_array_converter)
    wind_directions: NDArrayFloat = field(converter=floris_array_converter)
    air_density: float = field(converter=float)
    turbulence_intensities: NDArrayFloat = field(converter=floris_array_converter)
    reference_wind_height: float = field(converter=float)

    # Attributes with defaults
    wind_veer: float = field(converter=float, default=0.0)
    wind_shear: float = field(converter=float, default=0.0)
    surface_roughness: float = field(converter=float, default=0.03)
    obukhov_length: float | None = field(default=None)
    latitude: float | None = field(default=None)
    heterogeneous_inflow_config: dict = field(default=None)
    multidim_conditions: dict = field(default=None)

    # Internal/calculated attributes
    n_findex: int = field(init=False)
    u_initial_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    v_initial_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    w_initial_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    u_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    v_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    w_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    u: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    v: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    w: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    het_map: NDArrayObject = field(init=False, default=None)
    dudz_initial_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))

    turbulence_intensity_field: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    turbulence_intensity_field_sorted: NDArrayFloat = field(
        init=False, factory=lambda: np.array([])
    )
    turbulence_intensity_field_sorted_avg: NDArrayFloat = field(
        init=False, factory=lambda: np.array([])
    )

    @turbulence_intensities.validator
    def turbulence_intensities_validator(
        self, instance: attrs.Attribute, value: NDArrayFloat
    ) -> None:

        # Check that the array is 1-dimensional
        if value.ndim != 1:
            raise ValueError(
                "turbulence_intensities must have 1-dimension"
            )

        # Check the turbulence intensity is length n_findex
        if len(value) != self.n_findex:
            raise ValueError("turbulence_intensities must be length n_findex")



    @wind_directions.validator
    def wind_directions_validator(self, instance: attrs.Attribute, value: NDArrayFloat) -> None:
        # Check that the array is 1-dimensional
        if self.wind_directions.ndim != 1:
            raise ValueError(
                "wind_directions must have 1-dimension"
            )

        """Using the validator method to keep the `n_findex` attribute up to date."""
        self.n_findex = value.size

    @wind_speeds.validator
    def wind_speeds_validator(self, instance: attrs.Attribute, value: NDArrayFloat) -> None:

        # Check that the array is 1-dimensional
        if self.wind_speeds.ndim != 1:
            raise ValueError(
                "wind_speeds must have 1-dimension"
            )

        """Confirm wind speeds and wind directions have the same length"""
        if len(self.wind_directions) != len(self.wind_speeds):
            raise ValueError(
                f"wind_directions (length = {len(self.wind_directions)}) and "
                f"wind_speeds (length = {len(self.wind_speeds)}) must have the same length"
            )

    @heterogeneous_inflow_config.validator
    def heterogeneous_config_validator(self, instance: attrs.Attribute, value: dict | None) -> None:
        """Using the validator method to check that the heterogeneous_inflow_config dictionary has
        the correct key-value pairs.
        """
        if value is None:
            return

        # Check that the correct keys are supplied for the heterogeneous_inflow_config dict
        for k in ["speed_multipliers", "x", "y"]:
            if k not in value.keys():
                raise ValueError(
                    "heterogeneous_inflow_config must contain entries for 'speed_multipliers',"
                    f"'x', and 'y', with 'z' optional. Missing '{k}'."
                )
        if "z" not in value:
            # If only a 2D case, add "None" for the z locations
            value["z"] = None

    @het_map.validator
    def het_map_validator(self, instance: attrs.Attribute, value: list | None) -> None:
        """Using this validator to make sure that the het_map has an interpolant defined for
        each findex.
        """
        if value is None:
            return

        if self.n_findex != np.array(value).shape[0]:
            raise ValueError(
                "The het_map's first dimension not equal to the FLORIS first dimension."
            )


    def __attrs_post_init__(self) -> None:
        if self.heterogeneous_inflow_config is not None:
            self.generate_heterogeneous_wind_map()


    def initialize_velocity_field(self, grid: Grid) -> None:
        """
        Initialize the velocity field based on atmospheric boundary layer (ABL) physics.
        
        This method calculates the initial wind velocity field for each grid point using a
        combination of wind speed profiles and directional wind veer. It supports:
        
        1. Wind speed profiles:
           - Logarithmic profile for neutral conditions
           - Monin-Obukhov Similarity Theory (MOST) for stable/unstable conditions
        
        2. Wind veer mechanisms:
           - Simple linear wind veer (degrees per meter of height)
           - Physics-based Coriolis veer using latitude and stability
        
        Parameters affecting the ABL profile:
        - surface_roughness (z₀): Aerodynamic roughness length in meters, typical values:
          * 0.0002 for open water
          * 0.03 for short grass
          * 0.1 for cropland
          * 0.5-1.0 for forests or urban areas
        
        - obukhov_length (L): Stability parameter in meters
          * L = None or L = ∞: neutral conditions (logarithmic profile)
          * L > 0: stable conditions (reduced mixing, steeper profiles)
          * L < 0: unstable conditions (enhanced mixing, flatter profiles)
          * Typical values: -1000 to 1000, with magnitudes < 100 representing strong stability/instability
        
        - latitude: Site latitude in degrees, used for Coriolis-induced wind veer
          * When provided, calculates physics-based wind veer with height
          * When None, uses simple linear veer rate (wind_veer parameter)
        
        - wind_veer: Simple linear veer rate in degrees per meter (only used when latitude is None)
        
        Args:
            grid (Grid): Grid object containing coordinates for velocity calculation
        
        Returns:
            None: Updates self.u_initial_sorted, self.v_initial_sorted, and related fields
        """


        # Get parameters
        z = grid.z_sorted # Shape: (n_turbines, n_grid_y, n_grid_z)
        z0 = self.surface_roughness
        L = self.obukhov_length
        ref_hh = self.reference_wind_height
        ref_ws = self.wind_speeds # Shape: (n_findex,)

        # Handle heights below roughness length for log calculation stability
        z_clipped = np.maximum(z, z0)
        # Ensure reference height is above roughness length
        if ref_hh <= z0:
            self.logger.error(
                f"Reference wind height {ref_hh} <= surface roughness {z0}. Cannot compute log profile."
            )
            raise ValueError("Reference wind height must be greater than surface roughness.")

        # Calculate stability parameters (zeta = z/L)
        if L is None or L == 0 or not np.isfinite(L): # Neutral case
            psi_m_ref = 0.0
            psi_m_z = np.zeros_like(z_clipped)
            phi_m_z = np.ones_like(z_clipped)
            is_neutral = True
            self.logger.info("Assuming neutral stability (L is None, 0, or non-finite).")
        else:
            zeta_ref = ref_hh / L
            zeta_z = z_clipped / L
            psi_m_ref = psi_m(np.array([zeta_ref]))[0] # psi_m expects array
            psi_m_z = psi_m(zeta_z) # Shape: (n_turbines, n_grid_y, n_grid_z)
            phi_m_z = phi_m(zeta_z) # Shape: (n_turbines, n_grid_y, n_grid_z)
            is_neutral = False
            self.logger.info(f"Using stability parameter L = {L:.2f} m.")

        # Calculate u_star (friction velocity) for each findex
        # u_star = kappa * U(ref) / (ln(ref/z0) - psi_m(ref/L))
        log_term_ref = np.log(ref_hh / z0)
        denominator = log_term_ref - psi_m_ref

        if denominator <= 1e-6: # Check for non-positive or very small denominator
             self.logger.warning(
                 f"Log profile denominator (ln(ref/z0) - psi_m(ref/L)) = {denominator:.2e} "
                 f"is near zero or negative (ref_hh={ref_hh}, z0={z0}, L={L}). "
                 f"This may indicate very stable conditions or invalid inputs. "
                 f"Clamping denominator to 1e-6."
             )
             denominator = 1e-6

        # ref_ws shape (n_findex,) -> u_star shape (n_findex,)
        u_star = (KAPPA * ref_ws) / denominator

        # Calculate wind profile U(z) = (u*/kappa) * (ln(z/z0) - psi_m(z/L))
        # Final shape: (n_findex, n_turbines, n_grid_y, n_grid_z)
        log_term_z = np.log(z_clipped / z0) # Shape: (n_turbines, n_grid_y, n_grid_z)

        # Broadcast u_star (n_findex,) and (log_term_z - psi_m_z) (n_turbines, n_grid_y, n_grid_z)
        # Add dimensions to u_star: u_star[:, None, None, None]
        # Add dimensions to (log_term_z - psi_m_z): [None, :, :, :]
        # Wind speed profile based on MOST
        wind_profile_plane_stable = (
            (u_star[:, None, None, None] / KAPPA) * 
            (log_term_z[None, :, :, :] - psi_m_z[None, :, :, :])
        )
        # Ensure non-negative wind speeds (can happen with extreme stability/inputs)
        wind_profile_plane_stable = np.maximum(wind_profile_plane_stable, 0.0)

        # Calculate gradient dU/dz = (u*/(kappa*z)) * phi_m(z/L)
        # Avoid division by zero for z
        denominator_dz = KAPPA * z_clipped # Shape: (n_turbines, n_grid_y, n_grid_z)
        # Handle points where z_clipped might still be zero if z0 was zero
        denominator_dz[denominator_dz == 0] = 1e-6 # Avoid division by zero

        dwind_profile_plane_stable = (
            (u_star[:, None, None, None] / denominator_dz[None, :, :, :]) * 
            phi_m_z[None, :, :, :]
        )

        # Check for heterogeneous inflow
        if self.het_map is None:
            speed_ups = 1.0
        else:
            # Determine speed ups from heterogeneous map (rest of this logic is unchanged)
            bounds = np.array(list(zip(
                self.heterogeneous_inflow_config['x'],
                self.heterogeneous_inflow_config['y']
            )))
            hull = ConvexHull(bounds)
            polygon = Polygon(bounds[hull.vertices])
            path = mpltPath.Path(polygon.boundary.coords)
            points = np.column_stack(
                (
                    grid.x_sorted_inertial_frame.flatten(),
                    grid.y_sorted_inertial_frame.flatten(),
                )
            )
            inside = path.contains_points(points)
            if not np.all(inside):
                self.logger.warning(
                    "The calculated flow field contains points outside of the the user-defined "
                    "heterogeneous inflow bounds. For these points, the interpolated value has "
                    "been filled with the freestream wind speed. If this is not the desired "
                    "behavior, the user will need to expand the heterogeneous inflow bounds to "
                    "fully cover the calculated flow field area."
                )

            if len(self.het_map[0].points[0]) == 2:
                speed_ups = self.calculate_speed_ups(
                    self.het_map,
                    grid.x_sorted_inertial_frame,
                    grid.y_sorted_inertial_frame
                )
            elif len(self.het_map[0].points[0]) == 3:
                speed_ups = self.calculate_speed_ups(
                    self.het_map,
                    grid.x_sorted_inertial_frame,
                    grid.y_sorted_inertial_frame,
                    grid.z_sorted
                )

        # Apply speed_ups to the stability-derived profile magnitude
        U_magnitude = wind_profile_plane_stable * speed_ups
        dUdz_magnitude = dwind_profile_plane_stable * speed_ups

        # Calculate height-dependent veer angle relative to reference height
        if self.latitude is not None and self.latitude != 0.0:
            # Use physics-based Coriolis veer calculation when latitude is provided
            # Calculate Coriolis parameter: f = 2*Omega*sin(latitude)
            latitude_rad = np.radians(self.latitude)
            f_coriolis = 2 * OMEGA_EARTH * np.sin(latitude_rad)  # Coriolis parameter
            
            # In neutral conditions, veer angle uses Ekman spiral approach
            # For stable/unstable conditions, stability affects eddy viscosity (K_m)
            if is_neutral:
                # Simple Ekman spiral model for neutral conditions
                # Use log-law surface friction velocity
                # Note: more complex models would adjust eddy viscosity directly
                z_diff = z_clipped - self.reference_wind_height
                # Ekman spiral angle: Δθ ≈ ln(z2/z1) * f/(κ*u*) * r
                # where r is an empirical constant (~0.6-0.7)
                # Scale by 0.7 to match typical observed values at mid-latitudes
                # Reshape to ensure broadcasting compatibility: (n_findex, n_turbines, n_grid_y, n_grid_z)
                delta_theta_rad = 0.7 * (f_coriolis / (KAPPA * u_star[:, None, None, None])) * z_diff[None, :, :, :]
                delta_theta_deg = np.degrees(delta_theta_rad)
                
                # Reshape to ensure consistency with expected shapes
                # This handles the case where delta_theta_rad has shape (n_findex, n_turbines, n_grid_y, n_grid_z)
                # but downstream code expects (n_turbines, n_grid_y, n_grid_z)
                if delta_theta_rad.ndim == 4 and delta_theta_rad.shape[0] == 1:
                    delta_theta_rad = delta_theta_rad[0]  # Take first index if it's singleton
                self.logger.info(f"Using Coriolis-based veer at latitude {self.latitude}°")
            else:
                # Stability-adjusted Ekman model
                # In stable conditions, veer increases due to reduced mixing
                # In unstable conditions, veer decreases due to enhanced mixing
                z_diff = z_clipped - self.reference_wind_height
                # Get average phi_m value as stability factor
                phi_m_avg = np.mean(phi_m_z)
                # Enhanced veer in stable, reduced in unstable conditions
                # Reshape to ensure broadcasting compatibility: (n_findex, n_turbines, n_grid_y, n_grid_z)
                delta_theta_rad = 0.7 * (f_coriolis / (KAPPA * u_star[:, None, None, None])) * z_diff[None, :, :, :] * phi_m_avg
                delta_theta_deg = np.degrees(delta_theta_rad)
                
                # Reshape to ensure consistency with expected shapes
                if delta_theta_rad.ndim == 4 and delta_theta_rad.shape[0] == 1:
                    delta_theta_rad = delta_theta_rad[0]  # Take first index if it's singleton
                self.logger.info(f"Using stability-adjusted Coriolis veer at latitude {self.latitude}°")
        else:
            # Use simple linear veer rate when no latitude provided (backward compatibility)
            # wind_veer here represents degrees of veer per meter of height change
            delta_theta_deg = self.wind_veer * (z_clipped - self.reference_wind_height)
            delta_theta_rad = np.radians(delta_theta_deg) # Shape: (n_turbines, n_grid_y, n_grid_z)

        # Calculate u and v components aligned with reference wind direction
        # Need to broadcast delta_theta_rad correctly for findex dimension
        # delta_theta_rad shape: (n_turbines, n_grid_y, n_grid_z)
        # U_magnitude shape: (n_findex, n_turbines, n_grid_y, n_grid_z)
        
        # Handle different possible shapes of delta_theta_rad
        if delta_theta_rad.ndim == 4:  # Shape: (n_findex, n_turbines, n_grid_y, n_grid_z)
            cos_veer = np.cos(delta_theta_rad)
            sin_veer = np.sin(delta_theta_rad) 
            # Direct multiplication, shapes already aligned
            self.u_initial_sorted = U_magnitude * cos_veer
            self.v_initial_sorted = U_magnitude * sin_veer
            self.dudz_initial_sorted = dUdz_magnitude * cos_veer
        else:  # Shape: (n_turbines, n_grid_y, n_grid_z)
            cos_veer = np.cos(delta_theta_rad) 
            sin_veer = np.sin(delta_theta_rad)
            # Need to broadcast for n_findex dimension
            self.u_initial_sorted = U_magnitude * cos_veer[np.newaxis, :, :, :]
            self.v_initial_sorted = U_magnitude * sin_veer[np.newaxis, :, :, :]
            self.dudz_initial_sorted = dUdz_magnitude * cos_veer[np.newaxis, :, :, :]

        # Log shape information for debugging
        self.logger.debug(f"u_initial_sorted shape: {self.u_initial_sorted.shape}")
        # Note: v-component gradient (dv/dz) is not explicitly stored/used currently.

        # w component remains zero
        self.w_initial_sorted = np.zeros_like(self.u_initial_sorted)

        # Copy to sorted fields
        self.u_sorted = self.u_initial_sorted.copy()
        self.v_sorted = self.v_initial_sorted.copy()
        self.w_sorted = self.w_initial_sorted.copy()

        # Initialize turbulence intensity field (remains unchanged for now)
        self.turbulence_intensity_field = self.turbulence_intensities[:, None, None, None]
        self.turbulence_intensity_field = np.repeat(
            self.turbulence_intensity_field,
            grid.n_turbines,
            axis=1
        )
        self.turbulence_intensity_field_sorted = self.turbulence_intensity_field.copy()

    def finalize(self, unsorted_indices):
        self.u = np.take_along_axis(self.u_sorted, unsorted_indices, axis=1)
        self.v = np.take_along_axis(self.v_sorted, unsorted_indices, axis=1)
        self.w = np.take_along_axis(self.w_sorted, unsorted_indices, axis=1)

        self.turbulence_intensity_field = np.mean(
            np.take_along_axis(
                self.turbulence_intensity_field_sorted,
                unsorted_indices,
                axis=1
            ),
            axis=(2,3)
        )

    def calculate_speed_ups(self, het_map, x, y, z=None):
        if z is not None:
            # Calculate the 3-dimensional speed ups; squeeze is needed as the generator
            # adds an extra dimension
            speed_ups = np.squeeze(
                [het_map[i](x[i:i+1], y[i:i+1], z[i:i+1]) for i in range( len(het_map))],
                axis=1,
            )

        else:
            # Calculate the 2-dimensional speed ups; squeeze is needed as the generator
            # adds an extra dimension
            speed_ups = np.squeeze(
                [het_map[i](x[i:i+1], y[i:i+1]) for i in range(len(het_map))],
                axis=1,
            )

        return speed_ups

    def generate_heterogeneous_wind_map(self):
        """This function creates the heterogeneous interpolant used to calculate heterogeneous
        inflows. The interpolant is for computing wind speed based on an x and y location in the
        flow field. This is computed using SciPy's LinearNDInterpolator and uses a fill value
        equal to the freestream for interpolated values outside of the user-defined heterogeneous
        map bounds.

        Args:
            heterogeneous_inflow_config (dict): The heterogeneous inflow configuration dictionary.
            The configuration should have the following inputs specified.
                - **speed_multipliers** (list): A list of speed up factors that will multiply
                    the specified freestream wind speed. This 2-dimensional array should have an
                    array of multiplicative factors defined for each wind direction.
                - **x** (list): A list of x locations at which the speed up factors are defined.
                - **y**: A list of y locations at which the speed up factors are defined.
                - **z** (optional): A list of z locations at which the speed up factors are defined.
        """
        speed_multipliers = np.array(self.heterogeneous_inflow_config['speed_multipliers'])
        x = self.heterogeneous_inflow_config['x']
        y = self.heterogeneous_inflow_config['y']
        z = self.heterogeneous_inflow_config['z']

        # Declare an empty list to store interpolants by findex
        interps_f = np.empty(self.n_findex, dtype=object)
        if z is not None:
            # Compute the 3-dimensional interpolants for each wind direction
            # Linear interpolation is used for points within the user-defined area of values,
            # while the freestream wind speed is used for points outside that region.

            # Because the (x,y,z) points are the same for each findex, we create the triangulation
            # once and then overwrite the values for each findex.

            # Create triangulation using zeroth findex
            interp_3d = self.interpolate_multiplier_xyz(
                x, y, z, speed_multipliers[0], fill_value=1.0
            )
            # Copy the interpolant for each findex and overwrite the values
            for findex in range(self.n_findex):
                interp_3d.values = speed_multipliers[findex, :].reshape(-1, 1)
                interps_f[findex] = copy.deepcopy(interp_3d)

        else:
            # Compute the 2-dimensional interpolants for each wind direction
            # Linear interpolation is used for points within the user-defined area of values,
            # while the freestream wind speed is used for points outside that region

            # Because the (x,y) points are the same for each findex, we create the triangulation
            # once and then overwrite the values for each findex.

            # Create triangulation using zeroth findex
            interp_2d = self.interpolate_multiplier_xy(x, y, speed_multipliers[0], fill_value=1.0)
            # Copy the interpolant for each findex and overwrite the values
            for findex in range(self.n_findex):
                interp_2d.values = speed_multipliers[findex, :].reshape(-1, 1)
                interps_f[findex] = copy.deepcopy(interp_2d)

        self.het_map = interps_f

    @staticmethod
    def interpolate_multiplier_xy(x: NDArrayFloat,
                                  y: NDArrayFloat,
                                  multiplier: NDArrayFloat,
                                  fill_value: float = 1.0):
        """Return an interpolant for a 2D multiplier field.

        Args:
            x (NDArrayFloat): x locations
            y (NDArrayFloat): y locations
            multiplier (NDArrayFloat): multipliers
            fill_value (float): fill value for points outside the region

        Returns:
            LinearNDInterpolator: interpolant
        """

        return LinearNDInterpolator(list(zip(x, y)), multiplier, fill_value=fill_value)


    @staticmethod
    def interpolate_multiplier_xyz(x: NDArrayFloat,
                                   y: NDArrayFloat,
                                   z: NDArrayFloat,
                                   multiplier: NDArrayFloat,
                                   fill_value: float = 1.0):
        """Return an interpolant for a 3D multiplier field.

        Args:
            x (NDArrayFloat): x locations
            y (NDArrayFloat): y locations
            z (NDArrayFloat): z locations
            multiplier (NDArrayFloat): multipliers
            fill_value (float): fill value for points outside the region

        Returns:
            LinearNDInterpolator: interpolant
        """

        return LinearNDInterpolator(list(zip(x, y, z)), multiplier, fill_value=fill_value)
