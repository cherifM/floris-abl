# FLORIS Blockage Models

This directory contains the implementation, validation, and documentation for the blockage models in FLORIS. Blockage effects are the velocity deficits that occur upstream of wind turbines due to their resistance to the incoming flow, which can lead to measurable power reductions in wind farms.

## Features

The FLORIS blockage models implementation includes:

1. **Five Blockage Models**:
   - **Parametrized Global Blockage Model**: Treats wind farm as parametrized porous object in ambient flow
   - **Vortex Cylinder (VC) Model**: Semi-infinite cylinder of tangential vorticity
   - **Mirrored Vortex Model**: Extension of VC with ground effects via mirror vorticity distribution
   - **Self-Similar Blockage Model**: Local blockage with self-similar velocity deficit profile
   - **Engineering Global Blockage Model**: Engineering approach coupled with wake model

2. **Validation Scripts**:
   - Centerline velocity deficit validation (Meyer Forsting et al., 2017)
   - Lateral profiles validation (Branlard & Meyer Forsting, 2020)
   - Ground effect on vertical profiles validation (Branlard et al., 2022)
   - Atmospheric stability effects validation (Schneemann et al., 2021)

3. **Documentation**:
   - Comprehensive blockage report with theoretical background and implementation details
   - Validation report with benchmark results
   - Extended validation report with detailed analysis of validation cases

## Installation

To install FLORIS with the blockage models, run the following command:

```bash
# Install in development mode
python floris/install_floris.py

# Install in regular mode (not recommended for development)
python floris/install_floris.py --regular

# Force reinstallation
python floris/install_floris.py --force

# Clean and reinstall
python floris/install_floris.py --clean
```

## Running Validation Cases

To run the validation cases and generate the validation plots:

```bash
# Run all validation cases
python floris/docs/blockage_models/run_validations.py

# Run a specific validation case
python floris/docs/blockage_models/run_validations.py --script validate_centerline_deficit.py

# Run validations with less verbose output
python floris/docs/blockage_models/run_validations.py --quiet
```

The validation scripts generate plots in the `validation_images` directory and detailed results in markdown files.

## Converting Documentation to PDF

You can convert the markdown reports to LaTeX and PDF format using:

```bash
# Convert all reports
python floris/docs/blockage_models/convert_reports.py

# Convert a specific report
python floris/docs/blockage_models/convert_reports.py --report extended_validation_report.md

# Specify a custom output directory
python floris/docs/blockage_models/convert_reports.py --output custom_output_dir
```

The conversion requires [Pandoc](https://pandoc.org/) and [LaTeX](https://www.latex-project.org/get/) to be installed on your system.

## Directory Structure

```bash
floris/docs/blockage_models/
├── comprehensive_blockage_report.md      # Complete report with theory and implementation
├── convert_reports.py                   # Script to convert MD to LaTeX and PDF
├── extended_validation_report.md         # Detailed validation report with analysis
├── README.md                            # This file
├── run_validations.py                   # Script to run all validation cases
├── validate_centerline_deficit.py       # Validation for centerline velocity deficit
├── validate_ground_effect.py            # Validation for ground effect on vertical profiles
├── validate_lateral_profiles.py         # Validation for lateral profiles
├── validate_stability_effects.py        # Validation for atmospheric stability effects
└── validation_images/                   # Directory containing validation plots
```

## References

1. Meyer Forsting, A. R., Troldborg, N., & Gaunaa, M. (2017). The flow upstream of a row of aligned wind turbine rotors and its effect on power production. Wind Energy, 20(1), 63-77.

2. Branlard, E., & Meyer Forsting, A. R. (2020). Assessing the blockage effect of wind turbines and wind farms using an analytical vortex model. Wind Energy, 23(11), 2068-2086.

3. Branlard, E., Meyer Forsting, A. R., van der Laan, M. P., & Réthoré, P. E. (2022). Validation of a vortex-based model for the prediction of wind farm blockage. Wind Energy Science, 7(5), 1911-1926.

4. Schneemann, J., Theuer, F., Rott, A., Dörenkämper, M., & Kühn, M. (2021). Offshore wind farm global blockage measured with scanning lidar. Wind Energy Science, 6(2), 521-538.

## Contributing Measurement Data

We welcome contributions of measurement data to improve and validate the blockage models. This section provides guidance on how to contribute such data.

### Measurement Data Requirements

To effectively use measurement data for blockage model validation or enhancement, the data should include:

1. **Flow Field Measurements**:
   - Velocity measurements upstream of turbine(s) (ideally 1-5 diameters upstream)
   - Measurement coordinates in turbine-relative coordinates (x/D, y/D, z/D)
   - Temporal resolution and averaging periods
   - Measurement uncertainty estimates

2. **Turbine Data**:
   - Turbine specifications (diameter, hub height, rated power)
   - Operating conditions during measurements (thrust coefficient, power output, rotor speed)
   - Yaw angle and any intentional misalignment

3. **Atmospheric Conditions**:
   - Wind speed and direction at reference height
   - Turbulence intensity
   - Atmospheric stability parameters (e.g., Richardson number, Obukhov length)
   - Vertical wind shear profile

4. **Site Information**:
   - Terrain description or elevation data
   - Roughness information
   - Layout coordinates for multi-turbine setups

### Data Format

Measurement data should be provided in one of the following formats:

1. **CSV files** with clear headers and units
2. **NetCDF files** following CF conventions
3. **HDF5 files** with self-describing metadata
4. **MATLAB .mat files** with structure documentation

Each dataset should include a README file that explains:

- Data collection methodology
- Instrument specifications and calibration
- Data processing methods
- Known limitations or uncertainties
- Relevant references to published work

### Workflow for Model Enhancement

To use measurement data for model enhancement or calibration:

1. **Data Processing**:

   ```python
   # Example of processing measurement data
   import numpy as np
   import pandas as pd
   from pathlib import Path
   
   def process_measurement_data(data_path):
       # Load data
       if data_path.suffix == '.csv':
           data = pd.read_csv(data_path)
       elif data_path.suffix == '.nc':
           import xarray as xr
           data = xr.open_dataset(data_path)
       
       # Normalize coordinates by rotor diameter
       D = data['rotor_diameter'].values[0]  # rotor diameter
       data['x_norm'] = data['x'] / D  # normalized x-coordinate
       data['y_norm'] = data['y'] / D  # normalized y-coordinate
       data['z_norm'] = data['z'] / D  # normalized z-coordinate
       
       # Calculate velocity deficit
       u_inf = data['reference_velocity'].values[0]  # freestream velocity
       data['velocity_deficit'] = (u_inf - data['velocity']) / u_inf
       
       return data
   ```

2. **Model Parameter Calibration**:

   ```python
   # Example of calibrating model parameters
   from scipy.optimize import minimize
   
   def calibrate_model_parameters(data, model_function, initial_params):
       # Define error function to minimize
       def error_function(params):
           # Predict velocity deficits with current parameters
           predicted_deficits = model_function(data['x_norm'], data['y_norm'], 
                                             data['z_norm'], params)
           # Calculate RMSE between predictions and measurements
           rmse = np.sqrt(np.mean((predicted_deficits - data['velocity_deficit'])**2))
           return rmse
       
       # Minimize error function
       result = minimize(error_function, initial_params, method='Nelder-Mead')
       return result.x  # Return optimized parameters
   ```

3. **Model Validation**:

   ```python
   # Example of validating calibrated model
   def validate_model(data, model_function, calibrated_params):
       # Predict velocity deficits with calibrated parameters
       predicted_deficits = model_function(data['x_norm'], data['y_norm'], 
                                         data['z_norm'], calibrated_params)
       
       # Calculate error metrics
       mae = np.mean(np.abs(predicted_deficits - data['velocity_deficit']))
       rmse = np.sqrt(np.mean((predicted_deficits - data['velocity_deficit'])**2))
       corr = np.corrcoef(predicted_deficits, data['velocity_deficit'])[0, 1]
       
       # Return validation metrics
       return {
           'MAE': mae,
           'RMSE': rmse,
           'Correlation': corr
       }
   ```

### Contributing Data and Enhancements

To contribute measurement data or model enhancements:

1. **Format your data** according to the guidelines above

2. **Create a pull request** with:
   - Your measurement data in the `validation_data` directory
   - Documentation describing the data collection and processing
   - Any proposed model enhancements based on your data
   - Updated validation scripts that incorporate your data

3. **Contact us** at [cmihoubi@gmail.com](mailto:cmihoubi@gmail.com) with any questions about data formatting or the contribution process

## License

This project is available under a dual-licensing model:

[![Non-Commercial License](https://img.shields.io/badge/Non--Commercial-GPL%20v3-blue.svg)](GPL-LICENSE.md)
[![Commercial License](https://img.shields.io/badge/Commercial-Contact%20for%20License-orange.svg)](mailto:cmihoubi@gmail.com)

### Non-Commercial Use

This software is free to use, modify and distribute under the GNU GPL v3.0 license for non-commercial purposes only. "Non-commercial" means personal, educational, research, or other uses that are not primarily intended for commercial advantage or monetary compensation.

See the [LICENSE.md](LICENSE.md) file for details.

### Commercial Use

For commercial use, a separate commercial license is required. Please contact [cmihoubi@gmail.com](mailto:cmihoubi@gmail.com) for pricing and terms.

Commercial users cannot use the GPL-licensed version of this software for any commercial purposes.
