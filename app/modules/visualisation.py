"""
This Module is responsible for Domain Visualisation
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
from matplotlib.lines import Line2D
from modules.domain import Layer


def plot_pressure_heatmap(
    simulation_grid: np.ndarray, show_boundary_box: bool = False
) -> figure:
    """Plot the Pressure Field from the Simulation Grid as a Heatmap

    Args:
        simulation_grid (np.ndarray): Simulation Grid
        show_boundary_box (bool, optional): Show Boundary Box. Defaults to False.

    Returns:
        figure: plt.imshow Visualisation
    """
    imshow = plt.imshow(simulation_grid[Layer.PRESSURE.value], origin="lower")

    plt.colorbar(imshow)

    plt.title("Druckfeld")

    if show_boundary_box:
        y_start = 0.5
        y_end = simulation_grid.shape[1] - 1.5

        x_start = 0.5
        x_end = simulation_grid.shape[2] - 1.5

        top_line = Line2D([x_start, x_end], [y_end, y_end], color="gray")
        left_line = Line2D([x_start, x_start], [y_start, y_end], color="gray")
        bottom_line = Line2D([x_start, x_end], [y_start, y_start], color="gray")
        right_line = Line2D([x_end, x_end], [y_start, y_end], color="gray")

        plt.gca().add_line(top_line)
        plt.gca().add_line(left_line)
        plt.gca().add_line(bottom_line)
        plt.gca().add_line(right_line)

    return imshow


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

    magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

    # Convert to angles
    velocity_x = velocity_x / np.sqrt(velocity_x**2 + velocity_y**2)
    velocity_y = velocity_y / np.sqrt(velocity_x**2 + velocity_y**2)

    quiver = plt.quiver(velocity_x, velocity_y, magnitude, headwidth=2)

    plt.colorbar(quiver)

    plt.title("Geschwindigkeitsfeld")

    if show_boundary_box:
        y_start = 0.5
        y_end = simulation_grid.shape[1] - 1.5

        x_start = 0.5
        x_end = simulation_grid.shape[2] - 1.5

        top_line = Line2D([x_start, x_end], [y_end, y_end], color="gray")
        left_line = Line2D([x_start, x_start], [y_start, y_end], color="gray")
        bottom_line = Line2D([x_start, x_end], [y_start, y_start], color="gray")
        right_line = Line2D([x_end, x_end], [y_start, y_end], color="gray")

        plt.gca().add_line(top_line)
        plt.gca().add_line(left_line)
        plt.gca().add_line(bottom_line)
        plt.gca().add_line(right_line)

    return quiver


def plot_velocity_streamlines(
    simulation_grid: np.ndarray,
    include_boundary_conditions: bool = False,
) -> figure:
    """Plot the Velocity Field from the Simulation Grid as Streamline Plot

    Args:
        simulation_grid (np.ndarray): Simulation Grid

    Returns:
        figure: plt.quiver Visualisation
    """
    if include_boundary_conditions:
        velocity_x = simulation_grid[Layer.VELOCITY_X.value]
        velocity_y = simulation_grid[Layer.VELOCITY_Y.value]

        magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        x = np.linspace(0, simulation_grid.shape[2], simulation_grid.shape[2])
        y = np.linspace(0, simulation_grid.shape[1], simulation_grid.shape[1])
    else:
        velocity_x = simulation_grid[Layer.VELOCITY_X.value, 1:-1, 1:-1]
        velocity_y = simulation_grid[Layer.VELOCITY_Y.value, 1:-1, 1:-1]

        magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        x = np.linspace(1, simulation_grid.shape[2] - 1, simulation_grid.shape[2] - 2)
        y = np.linspace(1, simulation_grid.shape[1] - 1, simulation_grid.shape[1] - 2)

    streamline = plt.streamplot(x, y, velocity_x, velocity_y, color=magnitude)

    plt.colorbar(streamline.lines)

    plt.title("Geschwindigkeitslinien")

    return streamline
