"""
This Module is responsible for Domain Visualisation
"""

import numpy as np
from matplotlib import figure
import matplotlib.pyplot as plt

from modules.domain import Layer


def plot_pressure_heatmap(simulation_grid: np.ndarray) -> figure:
    """Plot the Pressure Field from the Simulation Grid as a Heatmap

    Args:
        simulation_grid (np.ndarray): Simulation Grid

    Returns:
        figure: plt.imshow Visualisation
    """
    plt.imshow(simulation_grid[Layer.PRESSURE.value])

    plt.title("Druckfeld")


def plot_velocity_quiver_plot(simulation_grid: np.ndarray) -> figure:
    """Plot the Velocity Field from the Simulation Grid as Quiver Plot

    Args:
        simulation_grid (np.ndarray): Simulation Grid

    Returns:
        figure: plt.quiver Visualisation
    """
    velocity_x = simulation_grid[Layer.VELOCITY_X.value]
    velocity_y = simulation_grid[Layer.VELOCITY_Y.value]

    # Convert to angles
    velocity_x = velocity_x / np.sqrt(velocity_x**2 + velocity_y**2)
    velocity_y = velocity_y / np.sqrt(velocity_x**2 + velocity_y**2)

    plt.quiver(velocity_x, velocity_y, scale=25)

    plt.title("Geschwindigkeitsfeld")
