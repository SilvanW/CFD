from pathlib import Path

import numpy as np
import pytest

from app.modules.configuration import AppConfig, read_config
from app.modules.domain import (
    Layer,
    generate_simulation_grid,
    get_simulation_grid_value,
    set_simulation_grid_value,
)
from app.modules.simulation import Direction, central_difference, laplace_operator


@pytest.fixture
def test_config() -> AppConfig:
    config = read_config(Path("tests/test_config.json"))
    config.domain.grid_points_x = 3
    config.domain.grid_points_y = 3

    return config


@pytest.fixture
def simulation_grid(test_config: AppConfig) -> np.ndarray:
    return generate_simulation_grid(test_config.domain)


# Central Difference
def test_central_difference_u_x(simulation_grid):
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_X, 1, 2, 2)
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_X, 3, 2, 4)

    differenced_grid = central_difference(
        simulation_grid, Layer.VELOCITY_X, 2, 2, 1, Direction.X
    )
    assert get_simulation_grid_value(differenced_grid, Layer.VELOCITY_X, 2, 2) == 1


def test_central_difference_u_y(simulation_grid):
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_X, 2, 1, 2)
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_X, 2, 3, 4)

    differenced_grid = central_difference(
        simulation_grid, Layer.VELOCITY_X, 2, 2, 1, Direction.Y
    )

    assert get_simulation_grid_value(differenced_grid, Layer.VELOCITY_X, 2, 2) == 1


def test_central_difference_v_x(simulation_grid):
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_Y, 1, 2, 2)
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_Y, 3, 2, 4)

    differenced_grid = central_difference(
        simulation_grid, Layer.VELOCITY_Y, 2, 2, 1, Direction.X
    )

    assert get_simulation_grid_value(differenced_grid, Layer.VELOCITY_Y, 2, 2) == 1


def test_central_difference_v_y(simulation_grid):
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_Y, 2, 1, 2)
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_Y, 2, 3, 4)

    differenced_grid = central_difference(
        simulation_grid, Layer.VELOCITY_Y, 2, 2, 1, Direction.Y
    )

    assert get_simulation_grid_value(differenced_grid, Layer.VELOCITY_Y, 2, 2) == 1


# Laplace Operator
def test_laplace_operator(simulation_grid):
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_X, 2, 1, 4)
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_X, 2, 3, 4)
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_X, 1, 2, 4)
    set_simulation_grid_value(simulation_grid, Layer.VELOCITY_X, 3, 2, 4)

    differenced_grid = laplace_operator(simulation_grid, Layer.VELOCITY_X, 2, 2, 1)

    assert get_simulation_grid_value(differenced_grid, Layer.VELOCITY_X, 2, 2) == 16
