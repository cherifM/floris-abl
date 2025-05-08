# Blockage Models in FLORIS: Theoretical Foundation

## 1. Introduction to Blockage Effects

Wind farm blockage effect is a phenomenon occurring when the presence of wind turbines creates a resistance to the incoming flow, causing a reduction in wind speed upstream of the turbines. Unlike wake effects, which occur downstream of turbines, blockage effects manifest as velocity deficits that can extend several rotor diameters upstream.

These effects can lead to measurable reductions in energy production, particularly in large wind farms, and are increasingly recognized as important for accurate energy yield assessment. Blockage effects are typically on the order of 1-3% reduction in power production, and while these numbers may seem small, they can significantly impact financial projections for large offshore wind farms.

The fundamental equation describing the relationship between the velocity deficit caused by blockage effects and the resulting power reduction is derived from the cube law of wind power:

$$P = \frac{1}{2} \rho A C_p U^3$$

Where:

- $P$ is the power output
- $\rho$ is the air density
- $A$ is the rotor swept area
- $C_p$ is the power coefficient
- $U$ is the wind speed

For a small velocity deficit $\Delta U$, the power reduction $\Delta P$ can be approximated as:

$$\frac{\Delta P}{P} \approx 3 \frac{\Delta U}{U}$$

This demonstrates why even seemingly small velocity deficits (e.g., 1%) can result in larger power reductions (approximately 3%).

## 2. Physical Mechanisms of Blockage

### 2.1 Local Blockage

Local blockage refers to the velocity deficit in front of an individual turbine. When wind approaches a turbine, the presence of the rotor creates a back-pressure that propagates upstream, slowing down the approaching flow. This effect can be understood through actuator disk theory, where the turbine is modeled as a semi-permeable disk with a pressure discontinuity.

The induction zone upstream of a turbine can extend 2-3 rotor diameters, with velocity deficits becoming significant within approximately 1 rotor diameter of the turbine.

In the one-dimensional momentum theory, the axial induction factor $a$ is defined as:

$$a = \frac{U_{\infty} - U_{disk}}{U_{\infty}}$$

Where:

- $U_{\infty}$ is the freestream velocity
- $U_{disk}$ is the velocity at the disk

The thrust coefficient $C_T$ is related to the induction factor by:

$$C_T = 4a(1-a)$$

For an operating turbine, the axial induction factor typically ranges from 0.2 to 0.4, resulting in a velocity deficit that varies with distance upstream according to:

$$\frac{U(x) - U_{\infty}}{U_{\infty}} = \frac{-a}{\sqrt{1 + (\frac{r}{|x|})^2}}$$

Where:

- $x$ is the upstream distance from the rotor (negative value)
- $r$ is the rotor radius

### 2.2 Global Blockage

Global blockage refers to the cumulative effect of multiple turbines within a wind farm creating a larger-scale resistance to the flow. This leads to:


1. **Flow Deceleration**: An overall slowdown of the flow as it approaches the farm
2. **Flow Diversion**: Wind flowing around the farm as if it were a large obstacle
3. **Flow Acceleration**: Around the edges and above the farm, similar to flow over a hill

Global blockage is particularly relevant for large, dense wind farms and depends on:

- Farm layout and density
- Atmospheric conditions, especially stability
- Wind farm size relative to the atmospheric boundary layer height

The global blockage effect can be mathematically described using a simplified approach based on potential flow theory, where the farm acts as a semi-permeable obstacle. The velocity deficit upstream of the farm can be approximated as:

$$\frac{\Delta U(x)}{U_{\infty}} = -\frac{C_{farm}}{2\pi} \frac{A_{farm}}{(x^2 + R_{farm}^2)^{3/2}} x$$

Where:

- $C_{farm}$ is the effective farm drag coefficient
- $A_{farm}$ is the frontal area of the farm
- $R_{farm}$ is the equivalent farm radius
- $x$ is the upstream distance from the farm center

### 2.3 Influence of Atmospheric Conditions

Blockage effects are strongly influenced by atmospheric stability:

- **Stable Conditions**: Enhanced blockage due to limited vertical mixing
- **Unstable Conditions**: Reduced blockage as vertical mixing helps recover the velocity deficit
- **Boundary Layer Height**: Acts as a constraint on vertical flow diversion

The relationship between atmospheric stability and blockage can be quantified through the atmospheric stability parameter $L$ (Monin-Obukhov length). The blockage intensity $B$ can be adjusted based on stability using:

$$B_{stability} = B_{neutral} \cdot f(L)$$

Where $f(L)$ is a stability correction function:

$$f(L) = \begin{cases}
1.0 + 0.2 \cdot \log(-L/100) & \text{for } L < 0 \text{ (unstable)} \\
1.0 & \text{for } L = 0 \text{ (neutral)} \\
1.0 + 0.3 \cdot \log(L/100) & \text{for } L > 0 \text{ (stable)}
\end{cases}$$

## 3. Theoretical Models

### 3.1 Parametrized Global Blockage Model (2025)

#### Theory
The Parametrized Global Blockage Model represents the state-of-the-art in blockage modeling. It treats the wind farm as a parametrized porous object within an ambient flow field, providing an engineering approximation to the complex fluid dynamics involved.

Key aspects of the model:
- Wind farm geometry is parametrized as a porous obstacle
- Flow field perturbations are modeled using a set of analytical functions
- Blockage intensity scales with farm thrust and porosity
- Effects decay with distance upstream following exponential patterns
- Atmospheric boundary layer constraints are incorporated

The model is expressed mathematically as:

$$\Delta u(x, y, z) = -B \cdot C_T \cdot p \cdot e^{-\alpha |x|/L} \cdot e^{-(y/W)^2} \cdot e^{-z/H}$$

Where:
- $\Delta u$ is the velocity deficit
- $B$ is the blockage intensity parameter
- $C_T$ is the thrust coefficient
- $p$ is the porosity coefficient
- $\alpha$ is the decay constant
- $L, W, H$ are characteristic length scales of the farm and atmosphere

### 3.2 Vortex Cylinder (VC) Model

#### Theory
The Vortex Cylinder model provides an analytical description of the induced velocities around a turbine by representing the wake as a semi-infinite cylinder of constant tangential vorticity. This model is derived from first principles of potential flow theory and offers high accuracy with low computational cost.

The model idealizes the turbine wake as a tube of distributed vorticity, where the strength of the vorticity is related to the turbine's thrust coefficient:

$$\gamma_t = -\frac{1}{2} U_{\infty} \frac{C_T}{R}$$

Where:
- $\gamma_t$ is the tangential vorticity strength
- $U_{\infty}$ is the freestream velocity
- $C_T$ is the thrust coefficient
- $R$ is the rotor radius

The induced velocities at any point in the flow field can be calculated using line integral solutions involving complete elliptic integrals of the first and second kind:

$$u_{ind} = \frac{\gamma_t R}{2\pi} \frac{m}{r} \frac{x}{\sqrt{(R+r)^2+x^2}} (K(m) - E(m))$$

$$v_{rad} = \frac{\gamma_t R}{2\pi} \frac{m}{r} \left(\left(\frac{R^2-r^2}{(R+r)^2+x^2}+1\right) K(m) - E(m)\right)$$

For points outside the vortex cylinder, where:
- $m = \frac{4Rr}{(R+r)^2+x^2}$ is a geometric parameter
- $K(m)$ and $E(m)$ are complete elliptic integrals of the first and second kind

### 3.3 Mirrored Vortex Model

#### Theory
The Mirrored Vortex Model extends the Vortex Cylinder approach by accounting for ground effects using the method of images from potential flow theory. This method places a mirror vortex system below the ground plane to enforce the non-penetration boundary condition at the ground surface.

The model adds a mirror image of each vortex cylinder reflected across the ground plane:

$$\mathbf{u}_{total} = \mathbf{u}_{original} + \mathbf{u}_{mirror}$$

Where:
- $\mathbf{u}_{original}$ is the velocity field from the original vortex cylinder
- $\mathbf{u}_{mirror}$ is the velocity field from the mirrored vortex cylinder

The axial induced velocity components add together, while the vertical components cancel at the ground plane, satisfying the slip condition.

This model produces enhanced induction in front of the rotor when the turbine is relatively close to the ground, which better matches experimental measurements and high-fidelity simulations.

### 3.4 Self-Similar Blockage Model

#### Theory
The Self-Similar Blockage Model assumes that the velocity deficit profiles in the induction zone maintain a similar shape but scale with distance from the turbine. This model applies Gaussian-like profiles to describe the velocity deficit distribution.

The key assumption is that the normalized velocity deficit follows:

$$\frac{\Delta u(r,x)}{a_0 U_{\infty}} = f\left(\frac{r}{\sigma(x)}\right) \cdot g(x)$$

Where:
- $a_0$ is the induction factor at the rotor plane
- $f(r/\sigma)$ is a radial similarity function (often Gaussian)
- $g(x)$ is an axial decay function
- $\sigma(x)$ is the characteristic width parameter that varies with axial position

Typically, the model uses:

$$f\left(\frac{r}{\sigma}\right) = e^{-(r/\sigma)^2}$$

$$g(x) = \frac{1}{1 + (|x|/D)^\beta}$$

Where:
- $\sigma$ is the similarity scale parameter
- $\beta$ is a decay parameter controlling how quickly effects diminish with distance
- $D$ is the rotor diameter

### 3.5 Engineering Global Blockage Model

#### Theory
The Engineering Global Blockage Model, developed by Nygaard et al. (2020), provides a simplified representation of blockage effects suitable for integration with existing wake models. This model uses a combination of analytical functions calibrated against measurement data and CFD simulations.

The model accounts for:
1. Farm-scale effects through a parametrized representation of the wind farm
2. Dependence on farm density and total thrust
3. Exponential decay of effects upstream
4. Lateral and vertical decay of effects away from farm center

The velocity deficit is calculated as:

$$\Delta u(x,y,z) = B_{amp} \cdot C_T \cdot \rho_{farm} \cdot e^{-|x|/L_{up}} \cdot e^{-(y/L_{lat})^2} \cdot e^{-(z/L_{vert})^2}$$

Where:
- $B_{amp}$ is the blockage amplitude parameter
- $C_T$ is the average thrust coefficient
- $\rho_{farm}$ is the farm density (ratio of total rotor area to farm area)
- $L_{up}$, $L_{lat}$, and $L_{vert}$ are characteristic length scales for upstream, lateral, and vertical extent

## 4. Selection of Appropriate Models

The choice of blockage model depends on the specific requirements and constraints:

| Model | Strengths | Limitations | Best For |
|-------|-----------|-------------|----------|
| Parametrized Global | Low computational cost, farm-scale effects | Simplifications in atmospheric effects | Large wind farm AEP calculations |
| Vortex Cylinder | Physically accurate, analytical solution | More complex calculation, single turbine only | Detailed near-turbine analysis |
| Mirrored Vortex | Includes ground effects, physically accurate | Higher computational cost than basic VC | Low-height turbines, accurate measurements |
| Self-Similar | Simple implementation, intuitive | Empirical parameters require calibration | Quick induction zone assessment |
| Engineering Global | Fast calculation, good for large farms | Empirical model, less accurate for complex layouts | Wind farm AEP assessment |

## 5. References

1. Branlard, E., & Meyer Forsting, A. R. (2020). Assessing the blockage effect of wind turbines and wind farms using an analytical vortex model. Wind Energy, 23(11), 2068-2086.

2. Meyer Forsting, A. R., Troldborg, N., & Gaunaa, M. (2017). The flow upstream of a row of aligned wind turbine rotors and its effect on power production. Wind Energy, 20(1), 63-77.

3. Nygaard, N. G., Steen, S. T., Poulsen, L., & Pedersen, J. G. (2020). Modelling cluster wakes and wind farm blockage. Journal of Physics: Conference Series, 1618, 062072.

4. Bleeg, J., Purcell, M., Ruisi, R., & Traiger, E. (2018). Wind farm blockage and the consequences of neglecting its effects on energy production. Energies, 11(6), 1609.

5. Segalini, A., & Dahlberg, J. Å. (2020). Blockage effects in wind farms. Wind Energy, 23(2), 120-128.
