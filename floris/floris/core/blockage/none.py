"""
None blockage model.

This model is used when no blockage effects should be applied.
"""

from typing import Any, Dict

import numpy as np
from attrs import define, field

from floris.core import BaseModel, Grid, FlowField, Farm


@define
class NoneBlockage(BaseModel):
    """
    The None blockage model does not modify the flow field,
    effectively modeling no blockage effects.
    """

    def __attrs_post_init__(self) -> None:
        """
        No initialization needed for the None blockage model.
        """
        pass

    def prepare_function(self, grid: Grid, flow_field: FlowField) -> dict:
        """
        This is a no-op for the None blockage model.

        Args:
            grid (Grid): Grid object containing coordinates for velocity calculation
            flow_field (FlowField): FlowField object with initial velocity field

        Returns:
            dict: Empty dictionary as no arguments are needed
        """
        return {}

    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        u_i: np.ndarray,
        v_i: np.ndarray,
        ct_i: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        This function returns zero blockage velocity deficit for all points.

        Args:
            x_i (np.ndarray): x-coordinates of current turbine 
            y_i (np.ndarray): y-coordinates of current turbine
            z_i (np.ndarray): z-coordinates of current turbine
            u_i (np.ndarray): flow speed at current turbine
            v_i (np.ndarray): lateral flow at current turbine
            ct_i (np.ndarray): thrust coefficient at current turbine

        Returns:
            np.ndarray: Zero velocity deficit (no blockage effect)
        """
        return np.zeros_like(x_i)
