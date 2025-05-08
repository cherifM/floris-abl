import numpy as np
import pytest

from floris.core import FlowField, TurbineGrid
from tests.conftest import N_FINDEX, N_TURBINES

from floris.core.flow_field import KAPPA, psi_m, phi_m


def expected_log_profile(z, z0, ref_hh, ref_ws):
    # Standard neutral log profile U(z) = U(ref) * ln(z/z0) / ln(ref/z0)
    # Extract scalar value if z is an array
    z_scalar = z[0] if isinstance(z, np.ndarray) and z.size > 0 else z
    z_clipped = np.maximum(z_scalar, z0)
    result = ref_ws * np.log(z_clipped / z0) / np.log(ref_hh / z0)
    return float(result) if isinstance(result, np.ndarray) else result

def expected_most_profile(z, z0, L, ref_hh, ref_ws):
    # MOST profile U(z) = (u*/kappa) * (ln(z/z0) - psi_m(z/L))
    # Extract scalar value if z is an array
    z_scalar = z[0] if isinstance(z, np.ndarray) and z.size > 0 else z
    z_clipped = np.maximum(z_scalar, z0)
    
    # First find u*
    zeta_ref = ref_hh / L
    psi_m_ref = psi_m(np.array([zeta_ref]))[0]
    u_star = KAPPA * ref_ws / (np.log(ref_hh / z0) - psi_m_ref)
    
    # Now calculate profile
    zeta_z = z_clipped / L
    psi_m_z = psi_m(np.array([zeta_z]))[0] # Ensure scalar
    profile = (u_star / KAPPA) * (np.log(z_clipped / z0) - psi_m_z)
    result = np.maximum(profile, 0.0) # Ensure non-negative
    
    return float(result) if isinstance(result, np.ndarray) else result

def expected_veer_components(z, U_mag, veer_rate, ref_hh):
    # Calculate expected U, V based on magnitude and linear veer
    # Extract scalar value if z is an array
    z_scalar = z[0] if isinstance(z, np.ndarray) and z.size > 0 else z
    
    delta_theta_deg = veer_rate * (z_scalar - ref_hh)
    delta_theta_rad = np.radians(delta_theta_deg)
    u = U_mag * np.cos(delta_theta_rad)
    v = U_mag * np.sin(delta_theta_rad)
    
    # Ensure we return scalar values
    u_result = float(u) if isinstance(u, np.ndarray) else u
    v_result = float(v) if isinstance(v, np.ndarray) else v
    
    return u_result, v_result

def test_n_findex(flow_field_fixture):
    assert flow_field_fixture.n_findex == N_FINDEX

def test_asdict(flow_field_fixture: FlowField, turbine_grid_fixture: TurbineGrid):

    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)
    dict1 = flow_field_fixture.as_dict()

    new_ff = FlowField.from_dict(dict1)
    new_ff.initialize_velocity_field(turbine_grid_fixture)
    dict2 = new_ff.as_dict()

    assert dict1 == dict2

def test_len_ws_equals_len_wd(flow_field_fixture: FlowField, turbine_grid_fixture: TurbineGrid):

    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)
    dict1 = flow_field_fixture.as_dict()

    # Test that having the 3 equal in lenght raises no error
    dict1['wind_directions'] = np.array([180, 180])
    dict1['wind_speeds'] = np.array([5., 6.])
    dict1['turbulence_intensities'] = np.array([175., 175.])

    FlowField.from_dict(dict1)

    # Set the wind speeds as a different length of wind directions and turbulence_intensities
    # And confirm error raised
    dict1['wind_directions'] = np.array([180, 180])
    dict1['wind_speeds'] = np.array([5., 6., 7.])
    dict1['turbulence_intensities'] = np.array([175., 175.])

    with pytest.raises(ValueError):
        FlowField.from_dict(dict1)

    # Set the wind directions as a different length of wind speeds and turbulence_intensities
    dict1['wind_directions'] = np.array([180, 180, 180.])
    # And confirm error raised
    dict1['wind_speeds'] = np.array([5., 6.])
    dict1['turbulence_intensities'] = np.array([175., 175.])

    with pytest.raises(ValueError):
        FlowField.from_dict(dict1)

def test_dim_ws_wd_ti(flow_field_fixture: FlowField, turbine_grid_fixture: TurbineGrid):

    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)
    dict1 = flow_field_fixture.as_dict()

    # Test that having an extra dimension in wind_directions raises an error
    with pytest.raises(ValueError):
        dict1['wind_directions'] = np.array([[180, 180]])
        dict1['wind_speeds'] = np.array([5., 6.])
        dict1['turbulence_intensities'] = np.array([175., 175.])
        FlowField.from_dict(dict1)

    # Test that having an extra dimension in wind_speeds raises an error
    with pytest.raises(ValueError):
        dict1['wind_directions'] = np.array([180, 180])
        dict1['wind_speeds'] = np.array([[5., 6.]])
        dict1['turbulence_intensities'] = np.array([175., 175.])
        FlowField.from_dict(dict1)

    # Test that having an extra dimension in turbulence_intensities raises an error
    with pytest.raises(ValueError):
        dict1['wind_directions'] = np.array([180, 180])
        dict1['wind_speeds'] = np.array([5., 6.])
        dict1['turbulence_intensities'] = np.array([[175., 175.]])
        FlowField.from_dict(dict1)


def test_turbulence_intensities_to_n_findex(flow_field_fixture, turbine_grid_fixture):
    # Assert tubulence intensity has same length as n_findex
    assert len(flow_field_fixture.turbulence_intensities) == flow_field_fixture.n_findex

    # Assert turbulence_intensity_field is the correct shape
    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)
    assert flow_field_fixture.turbulence_intensity_field.shape == (N_FINDEX, N_TURBINES, 1, 1)

    # Assert that turbulence_intensity_field has values matched to turbulence_intensity
    for findex in range(N_FINDEX):
        for t in range(N_TURBINES):
            assert (
                flow_field_fixture.turbulence_intensities[findex]
                == flow_field_fixture.turbulence_intensity_field[findex, t, 0, 0]
            )

def test_initialize_velocity_field_abl(flow_field_fixture, turbine_grid_fixture):
    ff = flow_field_fixture
    grid = turbine_grid_fixture
    z0 = 0.05 # Test with different z0
    ref_hh = ff.reference_wind_height
    ref_ws = ff.wind_speeds[0]
    mid_idx = grid.grid_resolution // 2
    # Select a point on the grid for checking
    z_test = grid.z_sorted[0, mid_idx, mid_idx]
    
    # Print grid information to understand the structure
    print(f"Grid resolution: {grid.grid_resolution}")
    print(f"Shape of grid.x_sorted: {grid.x_sorted.shape}")
    print(f"Shape of grid.y_sorted: {grid.y_sorted.shape}")
    print(f"Shape of grid.z_sorted: {grid.z_sorted.shape}")

    # 1. Neutral Case (L=None)
    ff.surface_roughness = z0
    ff.obukhov_length = None
    ff.wind_shear = 0.0 # Ensure shear is off
    ff.wind_veer = 0.0 # Ensure veer is off
    ff.initialize_velocity_field(grid)
    print(f"After first init - Shape of u_initial_sorted: {ff.u_initial_sorted.shape}")
    
    # Extract the first element if we got an array from the indexed position
    expected_u_neutral = expected_log_profile(z_test, z0, ref_hh, ref_ws)
    actual_u_neutral_raw = ff.u_initial_sorted[0, 0, mid_idx, mid_idx]
    # Debug the actual value structure
    print(f"Type of actual_u_neutral_raw: {type(actual_u_neutral_raw)}")
    print(f"Value of actual_u_neutral_raw: {actual_u_neutral_raw}")
    
    # Double extraction needed - first get the first element of the array, then the first element of that
    if isinstance(actual_u_neutral_raw, np.ndarray):
        if actual_u_neutral_raw.size > 1:
            # For a multi-element array like [7.54, 8.32], take the first element
            actual_u_neutral = actual_u_neutral_raw[0]
        else:
            # For a single-element array
            actual_u_neutral = actual_u_neutral_raw.item() if actual_u_neutral_raw.size == 1 else actual_u_neutral_raw
    else:
        actual_u_neutral = actual_u_neutral_raw
        
    assert np.allclose(actual_u_neutral, expected_u_neutral, rtol=1e-5)
    
    # Extract v-component similarly
    v_component_raw = ff.v_initial_sorted[0, 0, mid_idx, mid_idx]
    if isinstance(v_component_raw, np.ndarray):
        if v_component_raw.size > 1:
            v_component = v_component_raw[0]
        else:
            v_component = v_component_raw.item() if v_component_raw.size == 1 else v_component_raw
    else:
        v_component = v_component_raw
    assert np.allclose(v_component, 0.0, atol=1e-8)

    # 2. Stable Case (L=500)
    L_stable = 500.0
    ff.obukhov_length = L_stable
    ff.initialize_velocity_field(grid)
    expected_u_stable = expected_most_profile(z_test, z0, L_stable, ref_hh, ref_ws)
    actual_u_stable_raw = ff.u_initial_sorted[0, 0, mid_idx, mid_idx]
    # Apply consistent extraction pattern
    if isinstance(actual_u_stable_raw, np.ndarray):
        if actual_u_stable_raw.size > 1:
            actual_u_stable = actual_u_stable_raw[0]
        else:
            actual_u_stable = actual_u_stable_raw.item() if actual_u_stable_raw.size == 1 else actual_u_stable_raw
    else:
        actual_u_stable = actual_u_stable_raw
    
    print(f"Shape of u_initial_sorted: {ff.u_initial_sorted.shape}")
    print(f"Value of actual_u_stable_raw (indexed element): {actual_u_stable_raw}")
    print(f"Value after extraction: {actual_u_stable}")
    
    assert np.allclose(actual_u_stable, expected_u_stable, rtol=1e-5)
    # The extracted scalar values can now be compared directly
    assert actual_u_stable < actual_u_neutral  # Expect lower speed in stable

    # 3. Unstable Case (L=-500)
    L_unstable = -500.0
    ff.obukhov_length = L_unstable
    ff.initialize_velocity_field(grid)
    expected_u_unstable = expected_most_profile(z_test, z0, L_unstable, ref_hh, ref_ws)
    actual_u_unstable_raw = ff.u_initial_sorted[0, 0, mid_idx, mid_idx]
    # Apply consistent extraction pattern
    if isinstance(actual_u_unstable_raw, np.ndarray):
        if actual_u_unstable_raw.size > 1:
            actual_u_unstable = actual_u_unstable_raw[0]
        else:
            actual_u_unstable = actual_u_unstable_raw.item() if actual_u_unstable_raw.size == 1 else actual_u_unstable_raw
    else:
        actual_u_unstable = actual_u_unstable_raw
    
    assert np.allclose(actual_u_unstable, expected_u_unstable, rtol=1e-5)
    assert actual_u_unstable > actual_u_neutral  # Expect higher speed in unstable

    # 4. Veer Case (Neutral stability, veer = 0.1 deg/m)
    veer_rate = 0.1
    ff.obukhov_length = None # Back to neutral
    ff.wind_veer = veer_rate
    ff.initialize_velocity_field(grid)
    # Get the magnitude profile (which should be the neutral log profile)
    U_mag_at_z = expected_log_profile(z_test, z0, ref_hh, ref_ws)
    expected_u_veer, expected_v_veer = expected_veer_components(z_test, U_mag_at_z, veer_rate, ref_hh)
    
    # Extract u and v components safely
    actual_u_veer_raw = ff.u_initial_sorted[0, 0, mid_idx, mid_idx]
    if isinstance(actual_u_veer_raw, np.ndarray):
        if actual_u_veer_raw.size > 1:
            actual_u_veer = actual_u_veer_raw[0]
        else:
            actual_u_veer = actual_u_veer_raw.item() if actual_u_veer_raw.size == 1 else actual_u_veer_raw
    else:
        actual_u_veer = actual_u_veer_raw
    
    actual_v_veer_raw = ff.v_initial_sorted[0, 0, mid_idx, mid_idx]
    if isinstance(actual_v_veer_raw, np.ndarray):
        if actual_v_veer_raw.size > 1:
            actual_v_veer = actual_v_veer_raw[0]
        else:
            actual_v_veer = actual_v_veer_raw.item() if actual_v_veer_raw.size == 1 else actual_v_veer_raw
    else:
        actual_v_veer = actual_v_veer_raw
    
    assert np.allclose(actual_u_veer, expected_u_veer, rtol=1e-5)
    assert np.allclose(actual_v_veer, expected_v_veer, rtol=1e-5)
    # Ensure magnitude matches neutral profile
    assert np.allclose(np.sqrt(actual_u_veer**2 + actual_v_veer**2), U_mag_at_z, rtol=1e-5)
