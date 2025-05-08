#!/usr/bin/env python

"""
Script to generate a comparison of the key equations used in the five blockage models.
This creates a mathematical visualization for the comprehensive blockage report.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches

# Directory to save figures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_equation_comparison():
    """Create a visual comparison of the key equations for each blockage model."""
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Hide axes
    ax.axis('off')
    
    # Model titles
    models = [
        "Parametrized Global\nBlockage Model",
        "Vortex Cylinder\nModel",
        "Mirrored Vortex\nModel",
        "Self-Similar\nBlockage Model",
        "Engineering Global\nBlockage Model"
    ]
    
    # Key equations for each model (using LaTeX format)
    equations = [
        # Parametrized Global
        r"$\Delta u(x, y, z) = -B \cdot C_T \cdot p \cdot e^{-\alpha |x|/L} \cdot e^{-(y/W)^2} \cdot e^{-z/H}$",
        
        # Vortex Cylinder
        r"$u_{ind} = \frac{\gamma_t R}{2\pi} \frac{m}{r} \frac{x}{\sqrt{(R+r)^2+x^2}} (K(m) - E(m))$" + "\n" + 
        r"$\gamma_t = -\frac{1}{2} U_{\infty} \frac{C_T}{1-a} \frac{1}{R}$",
        
        # Mirrored Vortex
        r"$\mathbf{u}_{total} = \mathbf{u}_{original} + \mathbf{u}_{mirror}$" + "\n" +
        r"$u_{mirror,x} = u_{original,x}(x, y, -z + 2h)$" + "\n" +
        r"$u_{mirror,z} = -u_{original,z}(x, y, -z + 2h)$",
        
        # Self-Similar
        r"$\frac{\Delta u(r,x)}{a_0 U_{\infty}} = f\left(\frac{r}{\sigma(x)}\right) \cdot g(x)$" + "\n" +
        r"$f\left(\frac{r}{\sigma}\right) = e^{-(r/\sigma)^\alpha}$" + "\n" +
        r"$g(x) = \frac{1}{1 + (|x|/D)^\beta}$",
        
        # Engineering Global
        r"$\Delta u(x,y,z) = B_{amp} \cdot C_T \cdot \rho_{farm} \cdot e^{-|x|/L_{up}} \cdot e^{-(y/L_{lat})^2} \cdot e^{-(z/L_{vert})^2}$"
    ]
    
    # Simplified descriptions of what each equation represents
    descriptions = [
        "Where:\n" +
        r"$B$ = blockage intensity parameter" + "\n" +
        r"$C_T$ = thrust coefficient" + "\n" +
        r"$p$ = porosity coefficient" + "\n" +
        r"$\alpha$ = decay constant" + "\n" +
        r"$L, W, H$ = characteristic length scales",
        
        "Where:\n" +
        r"$\gamma_t$ = tangential vorticity strength" + "\n" +
        r"$R$ = rotor radius" + "\n" +
        r"$m$ = geometric parameter" + "\n" +
        r"$K(m), E(m)$ = elliptic integrals" + "\n" +
        r"$a$ = axial induction factor",
        
        "Where:\n" +
        r"$\mathbf{u}_{original}$ = original vortex velocity" + "\n" +
        r"$\mathbf{u}_{mirror}$ = mirrored vortex velocity" + "\n" +
        r"$h$ = turbine hub height" + "\n" +
        "Ground effects enhance blockage",
        
        "Where:\n" +
        r"$a_0$ = induction factor at rotor" + "\n" +
        r"$\sigma$ = similarity scale parameter" + "\n" +
        r"$\alpha$ = radial profile parameter" + "\n" +
        r"$\beta$ = axial decay parameter" + "\n" +
        r"$D$ = rotor diameter",
        
        "Where:\n" +
        r"$B_{amp}$ = blockage amplitude" + "\n" +
        r"$\rho_{farm}$ = farm density" + "\n" +
        r"$L_{up}, L_{lat}, L_{vert}$ = characteristic length scales" + "\n" +
        "Simple engineering approach"
    ]
    
    # Physical interpretation of each model
    physics = [
        "• Treats wind farm as parametrized porous object\n• Accounts for global blockage effects\n• Models farm-scale interaction with atmosphere",
        "• Based on vortex theory\n• Wake represented as vortex cylinder\n• Analyzes induced velocities from vorticity",
        "• Extends vortex cylinder with ground effects\n• Uses mirror image method\n• Enhanced induction near ground",
        "• Based on velocity deficit similarity\n• Preserves profile shape at different distances\n• Simple empirical formulation",
        "• Simplified engineering approach\n• Farm-scale blockage model\n• Efficient for large wind farms"
    ]
    
    # Set up the cells
    rows = 5
    cols = 3
    
    # Calculate cell dimensions
    width = 1.0 / cols
    height = 1.0 / rows
    
    # Cell colors
    colors = [
        "#D6EAF8",  # Light blue
        "#D5F5E3",  # Light green
        "#FADBD8",  # Light red
        "#E8DAEF",  # Light purple
        "#FCF3CF"   # Light yellow
    ]
    
    # Draw cells and add text
    for i in range(rows):
        # Model title (col 0)
        title_rect = patches.Rectangle(
            (0.01, 1 - (i+1)*height), width-0.02, height-0.02, 
            facecolor=colors[i], alpha=0.5, transform=ax.transAxes
        )
        ax.add_patch(title_rect)
        ax.text(
            0.01 + width/2, 1 - (i+0.5)*height, models[i],
            ha='center', va='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes
        )
        
        # Equation (col 1)
        eq_rect = patches.Rectangle(
            (width+0.01, 1 - (i+1)*height), width-0.02, height-0.02, 
            facecolor=colors[i], alpha=0.3, transform=ax.transAxes
        )
        ax.add_patch(eq_rect)
        ax.text(
            width+0.01 + width/2, 1 - (i+0.5)*height, equations[i],
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes
        )
        
        # Description and physics (col 2)
        desc_rect = patches.Rectangle(
            (2*width+0.01, 1 - (i+1)*height), width-0.02, height-0.02, 
            facecolor=colors[i], alpha=0.3, transform=ax.transAxes
        )
        ax.add_patch(desc_rect)
        
        # Split the last column into description and physical interpretation
        ax.text(
            2*width+0.01 + width/2, 1 - (i+0.3)*height, descriptions[i],
            ha='center', va='center', fontsize=9,
            transform=ax.transAxes
        )
        
        ax.text(
            2*width+0.01 + width/2, 1 - (i+0.75)*height, physics[i],
            ha='center', va='center', fontsize=9, style='italic',
            transform=ax.transAxes
        )
    
    # Add column headers
    headers = ["Blockage Model", "Key Equation", "Parameters & Physics"]
    for j in range(cols):
        ax.text(
            j*width + width/2, 0.975, headers[j],
            ha='center', va='center', fontsize=14, fontweight='bold',
            transform=ax.transAxes
        )
    
    # Add overall title
    fig.suptitle("Comparison of Blockage Model Equations", fontsize=16, y=0.99)
    
    # Add note at bottom
    ax.text(
        0.5, 0.01, 
        "Note: Each model represents a different approach to calculating upstream velocity deficit due to blockage effects.",
        ha='center', va='center', fontsize=10, style='italic',
        transform=ax.transAxes
    )
    
    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "equation_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved equation comparison to {save_path}")
    plt.close()

if __name__ == "__main__":
    create_equation_comparison()
