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
