from pathlib import Path

import pytest

from app.modules.configuration import AppConfig, read_config
from app.modules.domain import Layer, generate_simulation_grid


@pytest.fixture
def test_config():
    config = read_config(Path("tests/test_config.json"))
    config.domain.grid_points_x = 2
    config.domain.grid_points_y = 2

    return config


def test_generate_simulation_grid_layer_count(test_config: AppConfig):
    simulation_grid = generate_simulation_grid(test_config.domain)

    assert simulation_grid.shape[0] == len(Layer)


def test_generate_simulation_grid_boundary_layers(test_config: AppConfig):
    simulation_grid = generate_simulation_grid(test_config.domain)

    assert simulation_grid.shape[1] == test_config.domain.grid_points_x + 2
    assert simulation_grid.shape[2] == test_config.domain.grid_points_y + 2
