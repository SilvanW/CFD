"""
This Module is responsible for Simulation
"""

from enum import Enum

import numpy as np
from modules.configuration import DomainConfig
from modules.domain import (
    Layer,
    enforce_pressure_boundary_condition,
    get_simulation_grid_value,
    set_simulation_grid_value,
)


class Direction(Enum):
    X = 0
    Y = 1


def central_difference_value(
    simulation_grid: np.ndarray,
    layer: Layer,
    target_x_coordinate: int,
    target_y_coordinate: int,
    direction: Direction,
    grid_cell_size: int,
) -> float:
    if direction == Direction.X:
        return (
            simulation_grid[layer.value, target_y_coordinate, target_x_coordinate + 1]
            - simulation_grid[layer.value, target_y_coordinate, target_x_coordinate - 1]
        ) / (2 * grid_cell_size)

    if direction == Direction.Y:
        return (
            simulation_grid[layer.value, target_y_coordinate + 1, target_x_coordinate]
            - simulation_grid[layer.value, target_y_coordinate - 1, target_x_coordinate]
        ) / (2 * grid_cell_size)

    raise NotImplementedError(f"The direction {direction} is not implemented")


def laplace_operator_value(
    simulation_grid: np.ndarray,
    layer: Layer,
    target_x_coordinate: int,
    target_y_coordinate: int,
    grid_cell_size: int,
) -> float:
    return (
        simulation_grid[layer.value, target_y_coordinate - 1, target_x_coordinate]
        + simulation_grid[layer.value, target_y_coordinate + 1, target_x_coordinate]
        + simulation_grid[layer.value, target_y_coordinate, target_x_coordinate - 1]
        + simulation_grid[layer.value, target_y_coordinate, target_x_coordinate + 1]
        - 4 * simulation_grid[layer.value, target_y_coordinate, target_x_coordinate]
    ) / (grid_cell_size**2)


def get_intermediate_velocity(
    simulation_grid: np.ndarray,
    layer: Layer,
    direction: Direction,
    x_coord: int,
    y_coord: int,
    dynamic_viscosity: float,
    delta_t: float,
    grid_cell_size: float,
) -> float:
    if layer == Layer.VELOCITY_X:
        return get_simulation_grid_value(
            simulation_grid, layer, x_coord, y_coord
        ) + delta_t * (
            -(
                get_simulation_grid_value(simulation_grid, layer, x_coord, y_coord)
                * central_difference_value(
                    simulation_grid,
                    layer,
                    x_coord,
                    y_coord,
                    Direction.X,
                    grid_cell_size,
                )
                + get_simulation_grid_value(
                    simulation_grid, Layer.VELOCITY_Y, x_coord, y_coord
                )
                * central_difference_value(
                    simulation_grid,
                    layer,
                    x_coord,
                    y_coord,
                    Direction.Y,
                    grid_cell_size,
                )
            )
            + dynamic_viscosity
            * laplace_operator_value(
                simulation_grid, layer, x_coord, y_coord, grid_cell_size
            )
        )

    if layer == Layer.VELOCITY_Y:
        return get_simulation_grid_value(
            simulation_grid, layer, x_coord, y_coord
        ) + delta_t * (
            -(
                get_simulation_grid_value(
                    simulation_grid, Layer.VELOCITY_X, x_coord, y_coord
                )
                * central_difference_value(
                    simulation_grid,
                    layer,
                    x_coord,
                    y_coord,
                    Direction.X,
                    grid_cell_size,
                )
                + get_simulation_grid_value(simulation_grid, layer, x_coord, y_coord)
                * central_difference_value(
                    simulation_grid,
                    layer,
                    x_coord,
                    y_coord,
                    Direction.Y,
                    grid_cell_size,
                )
            )
            + dynamic_viscosity
            * laplace_operator_value(
                simulation_grid, layer, x_coord, y_coord, grid_cell_size
            )
        )

    raise NotImplementedError()


def right_hand_side_value(
    simulation_grid: np.ndarray,
    density: float,
    delta_t: float,
    target_x_coordinate: int,
    target_y_coordinate: int,
    grid_cell_size: int,
) -> None:
    return (
        density
        / delta_t
        * (
            central_difference_value(
                simulation_grid,
                Layer.VELOCITY_X,
                target_x_coordinate,
                target_y_coordinate,
                Direction.X,
                grid_cell_size,
            )
            + central_difference_value(
                simulation_grid,
                Layer.VELOCITY_Y,
                target_x_coordinate,
                target_y_coordinate,
                Direction.Y,
                grid_cell_size,
            )
        )
    )


def get_new_pressure_value(
    simulation_grid: np.ndarray,
    target_x_coordinate: int,
    target_y_coordinate: int,
    delta_t: float,
    density: float,
    grid_cell_size: int,
) -> float:
    return (
        1
        / 4
        * (
            get_simulation_grid_value(
                simulation_grid,
                Layer.PRESSURE,
                target_x_coordinate + 1,
                target_y_coordinate,
            )
            + get_simulation_grid_value(
                simulation_grid,
                Layer.PRESSURE,
                target_x_coordinate - 1,
                target_y_coordinate,
            )
            + get_simulation_grid_value(
                simulation_grid,
                Layer.PRESSURE,
                target_x_coordinate,
                target_y_coordinate + 1,
            )
            + get_simulation_grid_value(
                simulation_grid,
                Layer.PRESSURE,
                target_x_coordinate,
                target_y_coordinate - 1,
            )
            - grid_cell_size**2
            * right_hand_side_value(
                simulation_grid,
                density,
                delta_t,
                target_x_coordinate,
                target_y_coordinate,
                grid_cell_size,
            )
        )
    )


def calculate_pressure_residual(
    simulation_grid: np.ndarray, density: float, delta_t: int, grid_cell_size: float
) -> float:
    residual = np.zeros_like(simulation_grid)
    for x_coord in range(1, simulation_grid.shape[2] - 1):
        for y_coord in range(1, simulation_grid.shape[1] - 1):
            set_simulation_grid_value(
                residual,
                Layer.PRESSURE,
                x_coord,
                y_coord,
                laplace_operator_value(
                    simulation_grid, Layer.PRESSURE, x_coord, y_coord, grid_cell_size
                )
                - right_hand_side_value(
                    simulation_grid, density, delta_t, x_coord, y_coord, grid_cell_size
                ),
            )

    return np.sqrt(np.sum(residual[Layer.PRESSURE.value] ** 2))


def update_pressure_grid(
    simulation_grid: np.ndarray,
    density: float,
    delta_t: int,
    domain_config: DomainConfig,
    grid_cell_size: float,
    max_iterations: int = 50,
):
    for iteration in range(max_iterations):
        # Use temp pressure grid
        sim_grid_next = np.zeros_like(simulation_grid)
        for x_coord in range(1, simulation_grid.shape[2] - 1):
            for y_coord in range(1, simulation_grid.shape[1] - 1):
                set_simulation_grid_value(
                    sim_grid_next,  # store pressure value calculated form original grid in temp
                    Layer.PRESSURE,
                    x_coord,
                    y_coord,
                    get_new_pressure_value(
                        simulation_grid,
                        x_coord,
                        y_coord,
                        delta_t,
                        density,
                        grid_cell_size,
                    ),
                )

        enforce_pressure_boundary_condition(sim_grid_next, domain_config)

        simulation_grid[Layer.PRESSURE.value] = sim_grid_next[
            Layer.PRESSURE.value
        ].copy()

        pressure_residual = calculate_pressure_residual(
            simulation_grid, density, delta_t, grid_cell_size
        )

        if pressure_residual <= 1e-5:
            print("Pressure residual Small enough")
            break

    # print(iteration)


def calculate_velocity_residual(
    simulation_grid: np.ndarray, grid_cell_size: float
) -> float:
    residual = 0.0
    for x_coord in range(1, simulation_grid.shape[2] - 1):
        for y_coord in range(1, simulation_grid.shape[1] - 1):
            residual += (
                central_difference_value(
                    simulation_grid,
                    Layer.VELOCITY_X,
                    x_coord,
                    y_coord,
                    Direction.X,
                    grid_cell_size,
                )
                + central_difference_value(
                    simulation_grid,
                    Layer.VELOCITY_Y,
                    x_coord,
                    y_coord,
                    Direction.Y,
                    grid_cell_size,
                )
            ) ** 2

    return np.sqrt(residual)


def velocity_continuity(
    simulation_grid: np.ndarray, domain_config: DomainConfig
) -> np.ndarray:
    divergence = np.zeros_like(simulation_grid)

    # Ignore outer cells next to boundary
    for x_coord in range(2, simulation_grid.shape[2] - 2):
        for y_coord in range(2, simulation_grid.shape[1] - 2):
            set_simulation_grid_value(
                divergence,
                Layer.PRESSURE,
                x_coord,
                y_coord,
                central_difference_value(
                    simulation_grid,
                    Layer.VELOCITY_X,
                    x_coord,
                    y_coord,
                    Direction.X,
                    domain_config.grid_cell_size,
                )
                + central_difference_value(
                    simulation_grid,
                    Layer.VELOCITY_Y,
                    x_coord,
                    y_coord,
                    Direction.Y,
                    domain_config.grid_cell_size,
                ),
            )

    return divergence
