"""
This Module is responsible for Domain Generation
"""

from enum import Enum

import numpy as np
from modules.configuration import BoundaryConditionType, DomainConfig


class Layer(Enum):
    """
    Layer Definition for the Simulation Grid
    """

    PRESSURE = 0
    VELOCITY_X = 1
    VELOCITY_Y = 2


def set_simulation_grid_value(
    simulation_grid: np.ndarray,
    layer: Layer,
    x_coordinate: int,
    y_coordinate: int,
    value: float,
) -> None:
    simulation_grid[layer.value, y_coordinate, x_coordinate] = value


def get_simulation_grid_value(
    simulation_grid: np.ndarray,
    layer: Layer,
    x_coordinate: int,
    y_coordinate: int,
) -> float:
    return simulation_grid[layer.value, y_coordinate, x_coordinate]


def enforce_velocity_boundary_conditions(
    simulation_grid: np.ndarray,
    domain_config: DomainConfig,
):
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

    # Velocity Top
    simulation_grid[Layer.VELOCITY_X.value, domain_config.grid_points_y + 1] = float(
        domain_config.boundary_conditions.velocity_top.x_direction
    )

    simulation_grid[Layer.VELOCITY_Y.value, domain_config.grid_points_y + 1] = float(
        domain_config.boundary_conditions.velocity_top.y_direction
    )


def enforce_pressure_boundary_condition(
    simulation_grid: np.ndarray, domain_config: DomainConfig
) -> None:
    # Pressure Top
    simulation_grid[Layer.PRESSURE.value, domain_config.grid_points_y + 1] = float(
        domain_config.boundary_conditions.pressure_top.value
    )

    # Left Border
    simulation_grid[Layer.PRESSURE.value, 1:-1, 0] = simulation_grid[
        Layer.PRESSURE.value, 1:-1, 1
    ]

    # Right Border
    simulation_grid[Layer.PRESSURE.value, 1:-1, simulation_grid.shape[2] - 1] = (
        simulation_grid[Layer.PRESSURE.value, 1:-1, simulation_grid.shape[2] - 2]
    )

    # Lower Border
    simulation_grid[Layer.PRESSURE.value, 0] = simulation_grid[Layer.PRESSURE.value, 1]


def generate_simulation_grid(domain_config: DomainConfig) -> np.ndarray:
    """Generate Simulation Grid based on Layers and domain config

    Args:
        domain_config (DomainConfig): domain configuration

    Returns:
        np.ndarray: Generated Simulation Grid
    """
    simulation_grid = np.zeros(
        (len(Layer), domain_config.grid_points_y + 2, domain_config.grid_points_x + 2)
    )

    # Add Boundary Conditions
    enforce_velocity_boundary_conditions(simulation_grid, domain_config)
    enforce_pressure_boundary_condition(simulation_grid, domain_config)

    return simulation_grid
