"""
This Module holds Dimensionless quantities required to classify flows
"""


def reynolds_number(
    flow_speed: float, characteristic_lenth: float, kinematic_viscosity: float
) -> float:
    """Calculates the Reynolds Number of the Fluid based on the kinematic viscosity

    Args:
        flow_speed [m/s] (float): Speed of the Fluid
        characteristic_lenth [m] (float): Scale of the Physical System
        kinematic_viscosity [m^2/s] (float): Viscosity of the Fluid

    Raises:
        ValueError: Flow Speed cannot be negative
        ValueError: Characteristic Length cannot be negative

    Returns:
        float: Reynolds Number calculated from the passed arguments
    """

    if flow_speed < 0.0:
        raise ValueError("Flow Speed cannot be negative")

    if characteristic_lenth < 0.0:
        raise ValueError("Characteristice Length cannot be negative")

    return (flow_speed * characteristic_lenth) / kinematic_viscosity


def courant_number(flow_speed: float, time_step: float, grid_cell_size: float) -> float:
    """Calculates the Courant Number

    Args:
        flow_speed [m/s] (float): Speed of the Fluid
        time_step (float): Time step of the simulation
        grid_cell_size (float): Size of one grid Zell

    Raises:
        ValueError: Flow Speed cannot be negative

    Returns:
        float: _description_
    """
    if flow_speed < 0.0:
        raise ValueError("Flow Speed cannot be negative")
    return flow_speed * time_step / grid_cell_size
