# Atmospheric Boundary Layer Physics Implementation in FLORIS

**Date:** May 2, 2025  
**Author:**  Cherif  Mihoubi 

## Executive Summary

This report documents the implementation of atmospheric boundary layer (ABL) physics in the FLORIS wake modeling tool. The implementation includes Monin-Obukhov Similarity Theory (MOST) for modeling stability effects on wind profiles, and Coriolis-based wind veer calculation. These improvements enable FLORIS to simulate more realistic atmospheric conditions, including neutral, stable, and unstable atmospheric stability regimes, as well as directional wind changes with height.

The implementation has been thoroughly validated through unit tests and comparison with theoretical profiles. Results confirm that the ABL physics implementation correctly models the expected behavior of wind profiles under different stability conditions and latitudes.

## Introduction

Accurate modeling of the atmospheric boundary layer is essential for wind farm performance prediction. Traditional wake models often assume simplified wind profiles with power-law or logarithmic shapes that don't account for stability effects or more complex wind behavior with height. 

This implementation enhances FLORIS's ability to model atmospheric conditions by incorporating:

1. **Surface roughness effects** on wind profiles
2. **Atmospheric stability** using Monin-Obukhov Similarity Theory
3. **Coriolis-induced wind veer** with height based on latitude
4. **Stability-dependent wind profiles and veer**

These improvements allow for more accurate simulation of wind turbine performance across different atmospheric conditions and geographic locations.

## Theoretical Foundation

### Monin-Obukhov Similarity Theory

Monin-Obukhov Similarity Theory (MOST) provides a framework for describing the vertical structure of the atmospheric boundary layer. It is based on dimensional analysis and the assumption that the flow in the surface layer is governed by a few key parameters.

The key dimensionless parameter in MOST is the stability parameter zeta = z/L, where z is height above ground and L is the Obukhov length, which represents the height at which buoyancy effects become as important as mechanical (shear) production of turbulence.

The mean wind speed profile under MOST is given by:

U(z) = (u*/k) * [ln(z/z0) - psi_m(z/L)]

where:
- U(z) is the mean wind speed at height z
- u* is the friction velocity
- k is the von Karman constant (approximately 0.4)
- z0 is the surface roughness length
- psi_m is the stability correction function for momentum

The friction velocity u* is calculated from the reference wind speed at a known height:

u* = k*U(z_ref) / [ln(z_ref/z0) - psi_m(z_ref/L)]

### Atmospheric Stability Effects

The stability function psi_m depends on the atmospheric stability regime:

1. **Neutral conditions** (L very large or L = null):
   - psi_m(zeta) = 0
   - This leads to the standard logarithmic profile

2. **Stable conditions** (L > 0):
   - psi_m(zeta) = -5*zeta
   - Wind shear increases compared to neutral conditions
   - Wind speeds decrease at a given height

3. **Unstable conditions** (L < 0):
   - A more complex formula involving logarithms and arctangent functions
   - Wind shear decreases compared to neutral conditions
   - Wind speeds increase at a given height

The stability functions are based on the Dyer (1974) formulations, which are widely used and validated in boundary layer meteorology.

### Coriolis Effects and Wind Veer

Wind direction typically changes with height in the boundary layer due to the Coriolis force acting on the wind flow. This effect, known as the Ekman spiral, causes winds to veer (change direction clockwise in the Northern Hemisphere) with increasing height.

The basic Ekman spiral model predicts wind direction change with height based on:
- The Coriolis parameter (f = 2*Omega*sin(latitude))
- Earth's rotation rate (Omega = 7.2921×10^-5 rad/s)
- Latitude of the location
- Friction velocity and stability conditions

In stable conditions, wind veer tends to be stronger due to reduced vertical mixing, while in unstable conditions, enhanced vertical mixing reduces the wind veer.

## Implementation Details

### Code Structure

The ABL physics implementation is integrated into the `FlowField` class in FLORIS's core module. Key functions and parameters added include:

1. Global constants:
   - `KAPPA`: von Karman constant (0.4)
   - `OMEGA_EARTH`: Earth's rotation rate (7.2921e-5 rad/s)

2. New stability functions:
   - `phi_m`: Monin-Obukhov stability function for momentum
   - `psi_m`: Integrated Monin-Obukhov stability function for momentum

3. New parameters in `FlowField` class:
   - `surface_roughness`: Aerodynamic roughness length (default: 0.03 m)
   - `obukhov_length`: Monin-Obukhov length for stability (default: None, representing neutral conditions)
   - `latitude`: Site latitude for Coriolis effects (default: None)

4. Modified `initialize_velocity_field` method to implement ABL physics

### Stability Functions

The stability functions `phi_m` and `psi_m` are implemented to handle both scalar and array inputs using NumPy's broadcasting capabilities, with special handling to avoid numerical issues with extreme values.

### Velocity Profile Calculation

The wind profile calculation has been updated to use MOST instead of the traditional power-law approach:

1. For neutral conditions (L = None, 0, or non-finite):
   - The standard logarithmic profile is used
   - `psi_m` is set to 0

2. For stable/unstable conditions:
   - Calculate the stability parameter zeta = z/L
   - Compute the stability correction `psi_m(zeta)`
   - Calculate the friction velocity `u_*` using the reference wind speed
   - Apply the MOST profile

The implementation carefully handles dimension broadcasting to ensure compatibility with FLORIS's multi-dimensional arrays (for multiple flow conditions, turbines, and grid points).

### Wind Veer Implementation

Two methods are provided for calculating wind veer:

1. **Simple linear veer** (when latitude is not provided):
   - Uses the `wind_veer` parameter as a linear veer rate (degrees per meter)
   - Wind direction change: `Delta_theta = wind_veer * (z - reference_height)`

2. **Physics-based Coriolis veer** (when latitude is provided):
   - Calculates the Coriolis parameter based on latitude
   - For neutral conditions: Uses simplified Ekman spiral model
   - For stable/unstable conditions: Adjusts veer based on stability

The wind components are then calculated using cosine and sine of the veer angle.

### Backward Compatibility

To maintain compatibility with the existing codebase, several adaptations were made:

1. Array shape handling:
   - Added conditional logic to handle different shapes of `delta_theta_rad`
   - Ensured proper broadcasting between wind magnitude and directional components
   - Maintained the expected shape for velocity arrays

2. Parameter defaults:
   - New parameters have sensible defaults
   - When new parameters are not provided, behavior defaults to the original approach

## Validation

### Test Methodology

The implementation was validated through comprehensive unit tests in `tests/flow_field_unit_test.py`. A new test function `test_initialize_velocity_field_abl` was created to verify:

1. **Neutral profile** behavior:
   - Logarithmic wind profile
   - Correct scaling with surface roughness
   - Zero v-component when no veer is applied

2. **Stable profile** behavior:
   - Reduced wind speeds compared to neutral profile
   - Correct application of stability functions

3. **Unstable profile** behavior:
   - Increased wind speeds compared to neutral profile
   - Correct application of stability functions

4. **Wind veer effects**:
   - Correct calculation of u and v components
   - Preservation of wind speed magnitude under veer
   - Proper application of veer angle with height

Additional validations included:
- Compatibility with existing codebase (core_unit_test.py)
- Integration with turbine and farm modules (turbine_unit_test.py, farm_unit_test.py)

### Test Results

The key validation results include:

1. **Wind profiles under different stability conditions**:
   
   | Stability Condition | Expected Behavior | Test Result |
   |---------------------|-------------------|-------------|
   | Neutral (L = None)  | Logarithmic profile | PASS |
   | Stable (L = 500 m)  | Lower speed than neutral | PASS |
   | Unstable (L = -500 m) | Higher speed than neutral | PASS |

2. **Wind veer validation**:
   
   | Veer Condition | Expected Behavior | Test Result |
   |----------------|-------------------|-------------|
   | No veer (wind_veer = 0) | v-component = 0 | PASS |
   | Linear veer (wind_veer = 0.1) | Correct u,v components | PASS |
   | Coriolis veer (latitude = 45) | Height-dependent veer | PASS |

3. **Array dimension handling**:
   
   | Test | Expected Behavior | Test Result |
   |------|-------------------|-------------|
   | FlowField initialization | Correct attribute order | PASS |
   | Velocity field shape | Correct dimensions | PASS |
   | Integration with other modules | No broadcasting errors | PASS |

Key observations from wind profile analyses:
- Neutral profiles follow the expected logarithmic shape
- Stable profiles show increased shear and reduced speeds at higher heights
- Unstable profiles show reduced shear and increased speeds at higher heights
- Wind veer angles match theoretical Ekman spiral behavior at mid-latitudes

## Usage Examples

### Configuration Parameters

The implementation adds the following parameters to the FLORIS configuration:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| surface_roughness | Aerodynamic roughness length (m) | 0.0002 (water), 0.03 (grass), 0.1 (cropland), 0.5-1.0 (forest/urban) |
| obukhov_length | Stability parameter (m) | null (neutral), 10-500 (stable), -10 to -500 (unstable) |
| latitude | Site latitude for Coriolis calculations (degrees) | 0 (equator) to +/-90 (poles) |
| wind_veer | Simple linear veer rate (degrees/meter) | 0.0-0.1 |

### Example Scenarios

A sample configuration file `example_input_abl.yaml` demonstrates various atmospheric scenarios:

1. **Neutral conditions**: obukhov_length: null
2. **Stable conditions**: obukhov_length: 200
3. **Unstable conditions**: obukhov_length: -200
4. **Neutral with Coriolis**: obukhov_length: null, latitude: 40.0
5. **Stable with Coriolis**: obukhov_length: 200, latitude: 40.0

### API Usage

The ABL parameters can also be set programmatically using the FLORIS API, allowing for dynamic changes to atmospheric conditions during a simulation.

## Future Work

Several enhancements could further improve the ABL physics implementation:

1. **Enhanced stability models**:
   - Implement more complex stability correction functions for very stable/unstable conditions
   - Add support for stability transitions and time-varying stability

2. **Advanced Ekman spiral models**:
   - Implement height-dependent eddy viscosity models
   - Account for baroclinic effects in complex terrain

3. **Integration with meteorological data**:
   - Add capability to ingest stability parameters from meteorological measurements or models
   - Support for time series of stability conditions

4. **Validation with field measurements**:
   - Compare model predictions with wind profile measurements
   - Calibrate parameters against measured data

5. **Performance optimization**:
   - Optimize array operations for large simulation domains
   - Explore parallelization opportunities for stability calculations

## Conclusion

The implementation of atmospheric boundary layer physics in FLORIS represents a significant enhancement to the wake modeling tool's capabilities. By incorporating Monin-Obukhov Similarity Theory and Coriolis effects, FLORIS can now simulate more realistic atmospheric conditions, leading to improved accuracy in wind farm performance prediction.

The implementation has been thoroughly validated through unit tests and comparison with theoretical profiles. It maintains backward compatibility with the existing codebase while offering new capabilities for researchers and wind farm operators.

These improvements enable FLORIS users to:
- Model complex atmospheric stability conditions
- Account for site-specific terrain roughness
- Include latitude-dependent wind veer effects
- Simulate wind farms in diverse geographical locations and climate regimes

## References

1. Dyer, A. J. (1974). A review of flux-profile relationships. Boundary-Layer Meteorology, 7(3), 363-372.
2. Monin, A. S., & Obukhov, A. M. (1954). Basic laws of turbulent mixing in the atmosphere near the ground. Tr. Akad. Nauk SSSR Geofiz. Inst, 24(151), 163-187.
3. Stull, R. B. (1988). An introduction to boundary layer meteorology. Springer Science & Business Media.
4. Kaimal, J. C., & Finnigan, J. J. (1994). Atmospheric boundary layer flows: their structure and measurement. Oxford University Press.
5. Pena, A., Gryning, S. E., & Hasager, C. B. (2010). Comparing mixing-length models of the diabatic wind profile over homogeneous terrain. Theoretical and Applied Climatology, 100(3), 325-335.
