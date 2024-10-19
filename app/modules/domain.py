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
    simulation_grid = np.zeros(
        (len(Layer), domain_config.grid_points_x + 2, domain_config.grid_points_y + 2)
    )

    # Add Boundary Conditions
    # Velocity Top
    simulation_grid[Layer.VELOCITY_X.value, domain_config.grid_points_y + 1] = float(
        domain_config.boundary_conditions.velocity_top.x_direction
    )

    simulation_grid[Layer.VELOCITY_Y.value, domain_config.grid_points_y + 1] = float(
        domain_config.boundary_conditions.velocity_top.y_direction
    )

    # Velocity Right
    simulation_grid[Layer.VELOCITY_X.value, :, domain_config.grid_points_x + 1] = float(
        domain_config.boundary_conditions.velocity_right.x_direction
    )

    simulation_grid[Layer.VELOCITY_Y.value, :, domain_config.grid_points_x + 1] = float(
        domain_config.boundary_conditions.velocity_right.y_direction
    )

    # Velocity Bottom
    simulation_grid[Layer.VELOCITY_X.value, 0] = float(
        domain_config.boundary_conditions.velocity_bottom.x_direction
    )

    simulation_grid[Layer.VELOCITY_Y.value, 0] = float(
        domain_config.boundary_conditions.velocity_bottom.y_direction
    )

    # Velocity Left
    simulation_grid[Layer.VELOCITY_X.value, :, 0] = float(
        domain_config.boundary_conditions.velocity_left.x_direction
    )

    simulation_grid[Layer.VELOCITY_Y.value, :, 0] = float(
        domain_config.boundary_conditions.velocity_left.y_direction
    )

    # Pressure Top
    simulation_grid[Layer.PRESSURE.value, domain_config.grid_points_y + 1] = float(
        domain_config.boundary_conditions.pressure_top.value
    )

    # Pressure Right
    simulation_grid[Layer.PRESSURE.value, :, domain_config.grid_points_x + 1] = float(
        domain_config.boundary_conditions.pressure_right.value
    )

    # Pressure Bottom
    simulation_grid[Layer.PRESSURE.value, 0] = float(
        domain_config.boundary_conditions.pressure_bottom.value
    )

    # Pressure Left
    simulation_grid[Layer.PRESSURE.value, :, 0] = float(
        domain_config.boundary_conditions.pressure_left.value
    )

    return simulation_grid
