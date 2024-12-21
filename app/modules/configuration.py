"""
This Module is used to load a config json File and provide the Values in a Pydantic Class.
"""

import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

# from dotenv import load_dotenv
# from flatten_dict import flatten, unflatten
from pydantic import (
    BaseModel,
    ConfigDict,
    FilePath,
    PositiveFloat,
    PositiveInt,
    computed_field,
    validate_call,
)


class SolverConfig(BaseModel):

    max_iterations: PositiveInt
    target_residual: PositiveFloat
    time_step: PositiveFloat


class BoundaryConditionType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


class ScalarBoundaryCondition(BaseModel):
    type: BoundaryConditionType
    value: float


class VectorialBoundaryCondition(BaseModel):
    type: BoundaryConditionType
    x_direction: float
    y_direction: float


class BoundaryConditionConfig(BaseModel):

    # Velocity
    velocity_top: VectorialBoundaryCondition
    velocity_right: VectorialBoundaryCondition
    velocity_bottom: VectorialBoundaryCondition
    velocity_left: VectorialBoundaryCondition

    # Pressure
    pressure_top: ScalarBoundaryCondition
    pressure_right: ScalarBoundaryCondition
    pressure_bottom: ScalarBoundaryCondition
    pressure_left: ScalarBoundaryCondition


class DomainConfig(BaseModel):

    real_width: float
    real_height: float
    grid_points_x: int
    grid_points_y: int
    boundary_conditions: BoundaryConditionConfig

    @computed_field
    @property
    def grid_cell_size(self) -> float:
        return self.real_height / (self.grid_points_x - 1)


class FluidConfig(BaseModel):

    density: float
    kinematic_viscosity: float

    @computed_field
    @property
    def dynamic_viscosity(self) -> float:
        return self.kinematic_viscosity * self.density


class AppConfig(BaseModel):
    """
    This Class defines all the Values and the expected Type in the Config file.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    solver: SolverConfig
    domain: DomainConfig
    fluid: FluidConfig


@validate_call
def read_config(config_path: FilePath = Path("./config.json")) -> AppConfig:
    """Read config json file from specified path and return PydanticModel

    Args:
        config_path (FilePath, optional): Path to the config file. Defaults to "./config.json".

    Returns:
        AppConfig: Pydantic Representation of the config file
    """

    if config_path == "":
        path = os.path.join(os.getcwd(), config_path)
    else:
        path = config_path

    with open(path, "r", encoding="utf-8") as fh:
        json_config: dict = json.load(fh)

        return AppConfig(**json_config)
