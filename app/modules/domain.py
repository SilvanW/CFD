"""
This Module is responsible for Domain Generation
"""

from enum import Enum
from modules.configuration import DomainConfig
import numpy as np


class Layer(Enum):
    """
    Layer Definition for the Simulation Grid
    """

    PRESSURE = 0
    VELOCITY_X = 1
    VELOCITY_Y = 2


def generate_simulation_grid(domain_config: DomainConfig) -> np.ndarray:
    """Generate Simulation Grid based on Layers and domain config

    Args:
        domain_config (DomainConfig): domain configuration

    Returns:
        np.ndarray: Generated Simulation Grid
    """
    return np.zeros(
        (len(Layer), domain_config.grid_points_x, domain_config.grid_points_y)
    )
