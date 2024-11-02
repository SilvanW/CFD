"""
This Module is responsible for Simulation
"""

from enum import Enum

import numpy as np
from modules.domain import Layer


class Direction(Enum):
    X = 0
    Y = 1


def central_difference(
    simulation_grid: np.ndarray,
    layer: Layer,
    target_x_coordinate: int,
    target_y_coordinate: int,
    grid_sell_size: int,
    direction: Direction,
) -> np.ndarray:
    """
    - Layer wird verwendet um zwischen u und v zu unterscheiden
    - Direction wird verwendet um die abzuleitende richtung zu definieren
    """
    grid_copy = simulation_grid.copy()

    if direction == Direction.X:
        grid_copy[layer.value, target_y_coordinate, target_x_coordinate] = (
            simulation_grid[layer.value, target_y_coordinate, target_x_coordinate + 1]
            - simulation_grid[layer.value, target_y_coordinate, target_x_coordinate - 1]
        ) / (2 * grid_sell_size)

    if direction == Direction.Y:
        grid_copy[layer.value, target_y_coordinate, target_x_coordinate] = (
            simulation_grid[layer.value, target_y_coordinate + 1, target_x_coordinate]
            - simulation_grid[layer.value, target_y_coordinate - 1, target_x_coordinate]
        ) / (2 * grid_sell_size)

    return grid_copy


def laplace_operator(
    simulation_grid: np.ndarray,
    layer: Layer,
    target_x_coordinate: int,
    target_y_coordinate: int,
    grid_cell_size: int,
) -> np.ndarray:
    grid_copy = simulation_grid.copy()

    grid_copy[layer.value, target_y_coordinate, target_x_coordinate] = (
        simulation_grid[layer.value, target_y_coordinate - 1, target_x_coordinate]
        + simulation_grid[layer.value, target_y_coordinate + 1, target_x_coordinate]
        + simulation_grid[layer.value, target_y_coordinate, target_x_coordinate - 1]
        + simulation_grid[layer.value, target_y_coordinate, target_x_coordinate + 1]
        - 4 * simulation_grid[layer.value, target_y_coordinate, target_x_coordinate]
    ) / (grid_cell_size**2)

    return grid_copy
