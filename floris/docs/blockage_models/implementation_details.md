# Blockage Models in FLORIS: Implementation Details

## 1. Overview

The blockage model implementation in FLORIS extends the existing framework to account for velocity deficits upstream of wind turbines. This document describes the software architecture, implementation strategy, and integration with the FLORIS codebase.

## 2. Software Architecture

### 2.1 Directory Structure

The blockage models are implemented within their own dedicated directory in the FLORIS core structure:

```bash
floris/
└── floris/
    └── core/
        ├── blockage/               # Blockage models implementation
        │   ├── __init__.py         # Exports the model classes
        │   ├── none.py             # No-op blockage model
        │   ├── parametrized_global_blockage.py
        │   ├── vortex_cylinder.py
        │   ├── mirrored_vortex.py
        │   ├── self_similar_blockage.py
        │   └── engineering_global_blockage.py
        ├── wake_velocity/          # Existing wake velocity models
        ├── wake_deflection/        # Existing wake deflection models
        ├── wake_combination/       # Existing wake combination models
        ├── wake_turbulence/        # Existing wake turbulence models
        ├── wake.py                 # Modified to include blockage models
        └── solver.py               # Updated to incorporate blockage effects
```

### 2.2 Integration with Core FLORIS Components

The blockage models are integrated into the existing FLORIS framework by extending:

1. **WakeModelManager**: To manage blockage model selection and configuration
2. **Solver**: To apply blockage effects before wake calculations
3. **FLORIS Interface**: To enable user configuration of blockage models

## 3. Class Hierarchy and Design Patterns

### 3.1 Base Model Interface

All blockage models inherit from the `BaseModel` class, which defines the common interface:

```python
class BaseModel:
    def prepare_function(self, grid, flow_field):
        """Prepare model arguments"""
        pass
        
    def function(self, x_i, y_i, z_i, u_i, v_i, ct_i, **kwargs):
        """Calculate velocity deficit"""
        pass
```

This common interface ensures that all blockage models can be interchangeably used within the solver.

### 3.2 Model Parameter Handling

Parameters for each blockage model are defined using the `attrs` library with the `@define` decorator, which provides:

- Automatic initialization of attributes
- Type checking through type annotations
- Default values for optional parameters
- Conversion functions for input validation

Example from the Parametrized Global Blockage model:

```python
@define
class ParametrizedGlobalBlockage(BaseModel):
    blockage_intensity: float = field(default=0.05)
    decay_constant: float = field(default=3.0)
    boundary_layer_height: float = field(default=500.0)
    porosity_coefficient: float = field(default=0.7)
```

## 4. Key Implementation Details

### 4.1 WakeModelManager Extension

The `WakeModelManager` class in `wake.py` has been extended to incorporate blockage models:

```python
@define
class WakeModelManager(BaseClass):
    # Existing fields
    enable_blockage: bool = field(converter=bool, default=False)
    blockage_model_string: str = field(default="none")
    blockage_parameters: dict = field(converter=dict, factory=dict)
    blockage_model: BaseModel = field(init=False)
    
    def __attrs_post_init__(self) -> None:
        # Existing initialization
        # Initialize blockage model
        blockage_model_string = self.blockage_model_string.lower()
        
        # Create the blockage model instance
        if blockage_model_string == "none" and not self.enable_blockage:
            self.blockage_model = NoneBlockage()
        elif blockage_model_string == "parametrized_global":
            self.blockage_model = ParametrizedGlobalBlockage(**self.blockage_parameters)
        elif blockage_model_string == "vortex_cylinder":
            self.blockage_model = VortexCylinderBlockage(**self.blockage_parameters)
        elif blockage_model_string == "mirrored_vortex":
            self.blockage_model = MirroredVortexBlockage(**self.blockage_parameters)
        elif blockage_model_string == "self_similar":
            self.blockage_model = SelfSimilarBlockage(**self.blockage_parameters)
        elif blockage_model_string == "engineering_global":
            self.blockage_model = EngineeringGlobalBlockage(**self.blockage_parameters)
        else:
            raise ValueError(f"Blockage model {blockage_model_string} is not available.")
```

This implementation ensures that the `WakeModelManager` can instantiate any of the available blockage models based on user configuration. The model parameters are passed directly to the model constructor.
            # Use the specified blockage model
            model: BaseModel = MODEL_MAP["blockage_model"][blockage_model_string]
            if blockage_model_string == "none":
                model_parameters = None
            else:
                model_parameters = self.wake_blockage_parameters.get(blockage_model_string, None)
            if model_parameters is None:
                self.blockage_model = model()
            else:
                self.blockage_model = model.from_dict(model_parameters)
```

The `MODEL_MAP` dictionary has been updated to include blockage models:

```python
MODEL_MAP = {
    "blockage_model": {
        "none": NoneBlockage,
        "parametrized_global": ParametrizedGlobalBlockage,
        "vortex_cylinder": VortexCylinderBlockage,
        "mirrored_vortex": MirroredVortexBlockage,
        "self_similar": SelfSimilarBlockage,
        "engineering_global": EngineeringGlobalBlockage
    },
    # Existing model mappings
}
```

### 4.2 Solver Integration

The solvers in `solver.py` have been modified to calculate blockage effects before wake calculations. The key modification is to apply blockage effects to the velocity field prior to applying wake effects:


```python
def sequential_solver(farm, flow_field, grid, model_manager):
    # Initialize velocity field with freestream conditions
    u_field = flow_field.u_initial.copy()
    v_field = flow_field.v_initial.copy()
    w_field = flow_field.w_initial.copy()
    
    # Apply blockage effects before wake calculations
    if model_manager.enable_blockage:
        # Prepare blockage function with necessary arguments
        blockage_args = model_manager.blockage_model.prepare_function(
            grid=grid,
            flow_field=flow_field
        )
        
        # Calculate blockage velocity deficit
        for i in range(grid.n_turbines):
            # Extract turbine information
            x_i = grid.x_sorted[:, :, i:i+1, :]
            y_i = grid.y_sorted[:, :, i:i+1, :]
            z_i = grid.z_sorted[:, :, i:i+1, :]
            u_i = u_field[:, :, i:i+1, :]
            v_i = v_field[:, :, i:i+1, :]
            ct_i = farm.turbines[i].ct
            
            # Calculate velocity deficit from blockage
            blockage_deficit = model_manager.blockage_model.function(
                x_i=x_i,
                y_i=y_i,
                z_i=z_i,
                u_i=u_i,
                v_i=v_i,
                ct_i=ct_i,
                **blockage_args
            )
            
            # Apply blockage deficit to velocity field
            u_field = u_field - blockage_deficit
    
    # Continue with wake calculations using updated velocity field
    # ...
```

This implementation ensures that blockage effects are calculated and applied to the flow field before any wake calculations are performed, which is physically consistent with how blockage and wake effects interact in real wind farms.
    
    # Prepare blockage model if enabled
    if model_manager.enable_blockage:
        blockage_model_args = model_manager.blockage_model.prepare_function(grid, flow_field)
        blockage_field = np.zeros_like(flow_field.u_initial_sorted)
        
        # Calculate blockage effects for all turbines
        for i in range(grid.n_turbines):
            # Get turbine properties
            # Calculate blockage deficit
            blockage_deficit = model_manager.blockage_function(
                x_i=x_i, y_i=y_i, z_i=z_i, u_i=u_i, v_i=v_i, ct_i=ct_i,
                **blockage_model_args
            )
            blockage_field += blockage_deficit
            
        # Apply blockage to flow field
        flow_field.u_sorted = flow_field.u_initial_sorted - blockage_field
    
    # Proceed with wake calculations
```

This implementation strategy has been applied to all solver types: sequential, CC (Curl-Curl), TurbOPark, and Empirical Gauss.

### 4.3 Integration in User Interface

The blockage models are accessible through the FLORIS interface using the `set_wake_model` method:

```python
fi.set_wake_model(
    wake={"model_strings": {"velocity_model": "gauss", 
                           "deflection_model": "gauss", 
                           "combination_model": "sosfs",
                           "turbulence_model": "crespo_hernandez",
                           "blockage_model": "parametrized_global"},
          "enable_blockage": True,
          "wake_blockage_parameters": {"parametrized_global": {
              "blockage_intensity": 0.05,
              "decay_constant": 3.0,
              "boundary_layer_height": 500.0,
              "porosity_coefficient": 0.7
          }}
    }
)
```

## 5. Model-Specific Implementation Details

### 5.1 Parametrized Global Blockage Model

The Parametrized Global Blockage model treats the wind farm as a porous object and calculates:

1. Farm bounding box and dimensions
2. Coordinate transformation to wind-aligned coordinates
3. Upstream distance to farm boundary
4. Blockage effect calculation based on distance and farm properties

Key components:
- Lateral decay based on Gaussian-like function
- Vertical decay based on boundary layer height
- Upstream decay based on exponential function

### 5.2 Vortex Cylinder Model

The Vortex Cylinder model implements:

1. Calculation of tangential vorticity strength from thrust coefficient
2. Separate handling for points inside and outside the cylinder
3. Calculation of induced velocities using elliptic integrals
4. Optional finite-length wake modeling

The implementation includes special handling for:
- Small radius values to avoid numerical issues
- Proper vectorization for efficient calculation

### 5.3 Mirrored Vortex Model

The Mirrored Vortex model extends the Vortex Cylinder model with:

1. Mirror vortex system reflected across the ground plane
2. Vector component calculation to maintain proper boundary conditions
3. Combination of original and mirrored vortex systems

Special attention is given to enforcing the no-penetration condition at the ground plane.

### 5.4 Self-Similar Blockage Model

The Self-Similar model implementation includes:

1. Induction factor calculation from thrust coefficient
2. Radial profile based on Gaussian function
3. Axial decay based on power law
4. Turbine-by-turbine calculation of local blockage

### 5.5 Engineering Global Blockage Model

The Engineering Global Blockage model implements:

1. Farm area and density calculations
2. Calculation of farm centroid and dimensions
3. Blockage factor calculation based on thrust and farm density
4. Three-dimensional decay functions for blockage propagation

## 6. Computational Considerations

### 6.1 Performance Optimization

The blockage model implementations include several optimizations:

1. Vectorized operations using NumPy for efficient calculation
2. Pre-calculation of common terms to avoid redundant computation
3. Masks to apply calculations only where needed (e.g., upstream of turbines)
4. Efficient coordinate transformations

### 6.2 Memory Usage

Memory usage is optimized by:

1. Reusing existing array structures where possible
2. Creating new arrays only when necessary
3. Operating in-place on flow fields when applicable

### 6.3 Numerical Stability

Numerical stability is ensured through:

1. Checks for zero or small values to avoid division by zero
{
    "blockage_intensity": 0.05,  # Intensity parameter (0-1)
    "decay_constant": 3.0,       # Rate of decay with distance
    "boundary_layer_height": 500.0,  # Atmospheric boundary layer height (m)
    "porosity_coefficient": 0.7  # Farm porosity (0-1)
}
```

#### Core Algorithm

The model calculates the velocity deficit using a parametrized approach that includes:

1. Farm geometry calculations (bounding box and dimensions)
2. Wind-aligned coordinate transformation
3. Distance-based decay functions
4. Height-dependent attenuation
5. Lateral spreading factors

The mathematical implementation follows the formula:

$$\Delta u(x, y, z) = B \cdot C_T \cdot p \cdot e^{-\alpha |x|/L} \cdot e^{-(y/W)^2} \cdot e^{-z/H}$$

In code, this is implemented as:

```python
# Calculate upstream decay
upstream_decay = np.exp(-decay_constant * norm_dist)

# Height factor (decreases effect with height)
height_factor = np.exp(-z_i[upstream_mask] / boundary_layer_height)

# Lateral decay (gaussian-like)
lateral_factor = np.exp(-(lateral_dist**2))
```

## 8. Extension Points

The blockage model implementation provides several extension points:

1. New blockage models can be added by:
   - Creating a new model class that inherits from `BaseModel`
   - Implementing the required interface methods
   - Adding the model to the `MODEL_MAP` dictionary

2. Existing models can be extended by:
   - Adding new parameters to the model class
   - Enhancing the model's physics
   - Improving numerical performance

3. Integration with other FLORIS components:
   - Coupling with stability models
   - Integration with heterogeneous inflow capabilities
   - Connection to optimization frameworks
