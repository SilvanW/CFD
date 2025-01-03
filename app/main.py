import matplotlib.pyplot as plt
import numpy as np
from modules.configuration import AppConfig, read_config
from modules.dimensionless_quantities import courant_number, reynolds_number
from modules.domain import (
    Layer,
    enforce_pressure_boundary_condition,
    enforce_velocity_boundary_conditions,
    generate_simulation_grid,
    get_simulation_grid_value,
    set_simulation_grid_value,
)
from modules.simulation import (
    Direction,
    calculate_pressure_residual,
    central_difference_value,
    get_intermediate_velocity,
    right_hand_side_value,
    velocity_continuity,
)
from modules.visualisation import (
    plot_pressure_contour_velocity_streamlines,
    plot_pressure_heatmap,
    plot_velocity_quiver_plot,
    plot_velocity_streamlines,
)


def simulate(
    simulation_grid: np.ndarray, app_config: AppConfig, delta_t: float
) -> None:
    plt.ion()
    residuals: list[float] = []
    for iteration in range(app_config.solver.max_iterations):
        simulation_grid_one = np.zeros_like(simulation_grid)
        # 1. Impuls ohne Druckgradient
        for x_coord in range(1, simulation_grid.shape[2] - 1):
            for y_coord in range(1, simulation_grid.shape[1] - 1):
                # X Velocity
                set_simulation_grid_value(
                    simulation_grid_one,
                    Layer.VELOCITY_X,
                    x_coord,
                    y_coord,
                    get_intermediate_velocity(
                        simulation_grid,
                        Layer.VELOCITY_X,
                        x_coord,
                        y_coord,
                        app_config.fluid.dynamic_viscosity,
                        delta_t,
                        app_config.domain.grid_cell_size,
                    ),
                )

                # Y Velocity
                set_simulation_grid_value(
                    simulation_grid_one,
                    Layer.VELOCITY_Y,
                    x_coord,
                    y_coord,
                    get_intermediate_velocity(
                        simulation_grid,
                        Layer.VELOCITY_Y,
                        x_coord,
                        y_coord,
                        app_config.fluid.dynamic_viscosity,
                        delta_t,
                        app_config.domain.grid_cell_size,
                    ),
                )

        # 1.1 Geschwindigkeits Randbedingungen Erzwingen
        enforce_velocity_boundary_conditions(simulation_grid_one, app_config.domain)

        # 2.1 RHS der Druck Poisson Gleichung
        for _ in range(app_config.solver.max_pressure_iterations):
            # Use temp pressure grid
            for x_coord in range(1, simulation_grid_one.shape[2] - 1):
                for y_coord in range(1, simulation_grid_one.shape[1] - 1):
                    set_simulation_grid_value(
                        simulation_grid_one,  # store pressure value calculated form original grid in temp
                        Layer.PRESSURE,
                        x_coord,
                        y_coord,
                        (
                            1
                            / 4
                            * (
                                get_simulation_grid_value(
                                    simulation_grid,
                                    Layer.PRESSURE,
                                    x_coord + 1,
                                    y_coord,
                                )
                                + get_simulation_grid_value(
                                    simulation_grid,
                                    Layer.PRESSURE,
                                    x_coord - 1,
                                    y_coord,
                                )
                                + get_simulation_grid_value(
                                    simulation_grid,
                                    Layer.PRESSURE,
                                    x_coord,
                                    y_coord + 1,
                                )
                                + get_simulation_grid_value(
                                    simulation_grid,
                                    Layer.PRESSURE,
                                    x_coord,
                                    y_coord - 1,
                                )
                                - app_config.domain.grid_cell_size**2
                                * right_hand_side_value(
                                    simulation_grid_one,
                                    app_config.fluid.density,
                                    delta_t,
                                    x_coord,
                                    y_coord,
                                    app_config.domain.grid_cell_size,
                                )
                            )
                        ),
                    )

            enforce_pressure_boundary_condition(simulation_grid_one, app_config.domain)

            simulation_grid[Layer.PRESSURE.value] = simulation_grid_one[
                Layer.PRESSURE.value
            ]

            pressure_residual = calculate_pressure_residual(
                simulation_grid_one,
                app_config.fluid.density,
                delta_t,
                app_config.domain.grid_cell_size,
            )

            if pressure_residual <= 1e-5:
                print("Pressure residual Small enough")
                # break

        # get old velocity
        sim_grid_old = simulation_grid.copy()

        # 3.1 Geschwindigkeitskorrektur
        for x_coord in range(1, simulation_grid_one.shape[2] - 1):
            for y_coord in range(1, simulation_grid_one.shape[1] - 1):
                # Velocity X
                set_simulation_grid_value(
                    simulation_grid_one,
                    Layer.VELOCITY_X,
                    x_coord,
                    y_coord,
                    get_simulation_grid_value(
                        simulation_grid_one, Layer.VELOCITY_X, x_coord, y_coord
                    )
                    - delta_t
                    / app_config.fluid.density
                    * central_difference_value(
                        simulation_grid_one,
                        Layer.PRESSURE,
                        x_coord,
                        y_coord,
                        Direction.X,
                        app_config.domain.grid_cell_size,
                    ),
                )

                # Velocity Y
                set_simulation_grid_value(
                    simulation_grid_one,
                    Layer.VELOCITY_Y,
                    x_coord,
                    y_coord,
                    get_simulation_grid_value(
                        simulation_grid_one, Layer.VELOCITY_Y, x_coord, y_coord
                    )
                    - delta_t
                    / app_config.fluid.density
                    * central_difference_value(
                        simulation_grid_one,
                        Layer.PRESSURE,
                        x_coord,
                        y_coord,
                        Direction.Y,
                        app_config.domain.grid_cell_size,
                    ),
                )

        # 3.2 Geschw. RB erzwingen
        enforce_velocity_boundary_conditions(simulation_grid_one, app_config.domain)

        # Copy pressure form grid one to normal grid to ensure all information is in one place
        simulation_grid[Layer.PRESSURE.value] = simulation_grid_one[
            Layer.PRESSURE.value
        ]
        simulation_grid[Layer.VELOCITY_X.value] = simulation_grid_one[
            Layer.VELOCITY_X.value
        ]
        simulation_grid[Layer.VELOCITY_Y.value] = simulation_grid_one[
            Layer.VELOCITY_Y.value
        ]

        plt.clf()
        plot_pressure_contour_velocity_streamlines(
            simulation_grid_one, app_config.domain
        )
        plt.show()

        # Änderung klein
        velocity_norm_old = np.linalg.norm(
            sim_grid_old[Layer.VELOCITY_X.value] ** 2
            + sim_grid_old[Layer.VELOCITY_Y.value] ** 2
        )

        velocity_norm_new = np.linalg.norm(
            simulation_grid[Layer.VELOCITY_X.value] ** 2
            + simulation_grid[Layer.VELOCITY_Y.value] ** 2
        )

        velocity_residual = np.abs(velocity_norm_new - velocity_norm_old)

        print(f"Velocity Residual: {velocity_residual}")

        residuals.append(velocity_residual)

        if np.isnan(velocity_residual):
            print("Velocity Residual Exploded")
            break

        if velocity_residual <= app_config.solver.target_residual:
            print("Velocity Residual Small enough")
            # break

        plt.pause(0.00001)

    plt.ioff()
    print(iteration)

    # Velocity_Quiver
    plt.clf()
    plot_velocity_quiver_plot(simulation_grid_one)
    plt.savefig("../images/velocity_quiver.png")

    # Velocity Streamlines
    plt.clf()
    plot_velocity_streamlines(simulation_grid_one)
    plt.savefig("../images/velocity_streamlines.png")

    # Pressure Heatmap
    plt.clf()
    plot_pressure_heatmap(simulation_grid_one)
    plt.savefig("../images/pressure.png")

    # Residuals
    plt.clf()
    plt.plot(residuals)
    plt.title("Residuals")
    plt.xlabel("Iterations")
    plt.ylabel("Velocity Residual")
    plt.savefig("../images/residuals.png")


if __name__ == "__main__":
    app_config = read_config()

    simulation_grid = generate_simulation_grid(app_config.domain)

    # Check Config
    courant = courant_number(
        1, app_config.solver.time_step, app_config.domain.grid_cell_size
    )
    print(f"Courant Number: {courant}")

    reynolds = reynolds_number(
        1, app_config.domain.real_width, app_config.fluid.kinematic_viscosity
    )

    print(f"Reynolds Number: {reynolds}")

    # source for that
    if courant > 0.5:
        pass
        # raise ValueError("Courant number cannot be greater than 0.5")

    if courant <= 0.1:
        pass
        # raise ValueError("Courant number cannot be smaller than 0.1")

    simulate(simulation_grid, app_config, app_config.solver.time_step)

    plt.clf()
    plot_pressure_contour_velocity_streamlines(simulation_grid, app_config.domain)
    plt.savefig("../images/pressure_and_velocity.png")

    # Check Continuity
    divergence = velocity_continuity(simulation_grid, app_config.domain)
    print(f"Final Divergence: {np.max(np.abs(divergence[Layer.PRESSURE.value]))}")

    plt.clf()
    plt.imshow(divergence[Layer.PRESSURE.value], origin="lower")
    plt.colorbar()
    plt.savefig("../images/continuity.png")
