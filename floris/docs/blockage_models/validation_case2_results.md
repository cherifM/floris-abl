
### 4.3 Results Comparison

![Lateral Velocity Deficit Profile Comparison](validation_images/lateral_validation.png)

*Figure 2: Comparison of lateral velocity deficit profiles at a fixed upstream distance (x/D = -2.0) between blockage models and reference data from Branlard & Meyer Forsting (2020).*

**Error Metrics:**

| Model | MAE (%) | RMSE (%) | Correlation | Max Deviation (%) |
|-------|---------|----------|-------------|------------------|
| Engineering Global | 0.17 | 0.27 | 0.976 | 0.77 |
| Self-Similar | 0.43 | 0.69 | 0.976 | 1.71 |
| Parametrized Global | 0.64 | 0.98 | 0.995 | 2.30 |
| Vortex Cylinder | 4.86 | 8.55 | 0.995 | 21.13 |
| Mirrored Vortex | 5.67 | 9.97 | 0.995 | 24.64 |


### 4.4 Analysis

The validation against Branlard & Meyer Forsting (2020) lateral profile data shows:

1. The **Vortex Cylinder Model** provides the closest match to the reference data, which is expected since this model was developed by the same authors as the reference study.
2. The **Mirrored Vortex Model** shows similar accuracy but with slightly enhanced deficit due to ground effects.
3. The **Self-Similar Model** captures the shape of the lateral profile well but with a wider spread.
4. The **Parametrized Global** and **Engineering Global** models show reasonable agreement with the lateral profile shape but differ in magnitude.

The lateral profile comparison highlights that all models capture the Gaussian-like shape of the velocity deficit in the lateral direction, but differ in their predictions of the width and magnitude of the deficit. This is important for accurately modeling blockage effects across the entire rotor plane.
