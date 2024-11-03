"""
This Module is responsible for Domain Visualisation
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
from matplotlib.lines import Line2D
from modules.domain import Layer


def plot_pressure_heatmap(simulation_grid: np.ndarray) -> figure:
    """Plot the Pressure Field from the Simulation Grid as a Heatmap

    Args:
        simulation_grid (np.ndarray): Simulation Grid

    Returns:
        figure: plt.imshow Visualisation
    """
    plt.imshow(simulation_grid[Layer.PRESSURE.value], origin="lower")

    plt.title("Druckfeld")


def plot_velocity_quiver_plot(
    simulation_grid: np.ndarray, show_boundary_box: bool = False
) -> figure:
    """Plot the Velocity Field from the Simulation Grid as Quiver Plot

    Args:
        simulation_grid (np.ndarray): Simulation Grid
        show_boundary_box (bool, optional): Show Boundary Box. Defaults to False.

    Returns:
        figure: plt.quiver Visualisation
    """
    velocity_x = simulation_grid[Layer.VELOCITY_X.value]
    velocity_y = simulation_grid[Layer.VELOCITY_Y.value]

    # Convert to angles
    velocity_x = velocity_x / np.sqrt(velocity_x**2 + velocity_y**2)
    velocity_y = velocity_y / np.sqrt(velocity_x**2 + velocity_y**2)

    plt.quiver(velocity_x, velocity_y, headwidth=2)

    plt.title("Geschwindigkeitsfeld")

    if show_boundary_box:
        y_start = 0.5
        y_end = simulation_grid.shape[1] - 1.5

        x_start = 0.5
        x_end = simulation_grid.shape[2] - 1.5

        top_line = Line2D([x_start, x_end], [y_end, y_end], color="blue")
        left_line = Line2D([x_start, x_start], [y_start, y_end], color="blue")
        bottom_line = Line2D([x_start, x_end], [y_start, y_start], color="blue")
        right_line = Line2D([x_end, x_end], [y_start, y_end], color="blue")

        plt.gca().add_line(top_line)
        plt.gca().add_line(left_line)
        plt.gca().add_line(bottom_line)
        plt.gca().add_line(right_line)
