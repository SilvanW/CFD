from pathlib import Path

import pytest

from app.modules.configuration import AppConfig, read_config
from app.modules.dimensionless_quantities import reynolds_number


@pytest.fixture
def test_config() -> AppConfig:
    config = read_config(Path("tests/test_config.json"))
    config.fluid.kinematic_viscosity = 0.01

    return config


def test_reynolds_number(test_config: AppConfig):
    assert reynolds_number(1, 1, test_config.fluid.kinematic_viscosity) == 100


def test_reynolds_number_invalid_flow_speed(test_config: AppConfig):
    with pytest.raises(ValueError):
        reynolds_number(-1, 1, test_config.fluid.kinematic_viscosity)


def test_reynolds_number_invalid_characteristic_length(test_config: AppConfig):
    with pytest.raises(ValueError):
        reynolds_number(1, -1, test_config.fluid.kinematic_viscosity)
