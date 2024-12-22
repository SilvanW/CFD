import matplotlib.pyplot as plt
import numpy as np
from modules.configuration import AppConfig, read_config
from modules.domain import (
    Layer,
    enforce_velocity_boundary_conditions,
    generate_simulation_grid,
    get_simulation_grid_value,
    set_simulation_grid_value,
)
from modules.simulation import (
    Direction,
    calculate_velocity_residual,
    central_difference_value,
    get_intermediate_velocity,
    update_pressure_grid,
)
from modules.visualisation import (
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
        simulation_grid_one = simulation_grid.copy()

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
                        Direction.X,
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
                        Direction.Y,
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
        update_pressure_grid(
            simulation_grid_one,
            app_config.fluid.density,
            delta_t,
            app_config.domain,
            app_config.domain.grid_cell_size,
        )

        # plot_pressure_heatmap(simulation_grid_one, True)

        # get old velocity
        sim_grid_old = simulation_grid.copy()

        # 3.1 Geschwindigkeitskorrektur
        for x_coord in range(1, simulation_grid_one.shape[2] - 1):
            for y_coord in range(1, simulation_grid_one.shape[1] - 1):
                # Velocity X
                set_simulation_grid_value(
                    simulation_grid,
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
                    simulation_grid,
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

        # Copy pressure form grid one to normal grid to ensure all information is in one place
        simulation_grid[Layer.PRESSURE.value] = simulation_grid_one[
            Layer.PRESSURE.value
        ].copy()

        # 3.2 Geschw. RB erzwingen
        enforce_velocity_boundary_conditions(simulation_grid, app_config.domain)

        plt.clf()
        plot_velocity_quiver_plot(simulation_grid, True)
        # plot_pressure_heatmap(simulation_grid, True)
        plt.show()

        # Änderung klein
        velocity_norm_old = np.linalg.norm(
            sim_grid_old[Layer.VELOCITY_X.value] - sim_grid_old[Layer.VELOCITY_Y.value]
        )

        velocity_norm_new = np.linalg.norm(
            simulation_grid[Layer.VELOCITY_X.value]
            - simulation_grid[Layer.VELOCITY_Y.value]
        )

        velocity_residual = np.abs(velocity_norm_new - velocity_norm_old)

        print(f"Velocity Residual: {velocity_residual}")

        residuals.append(velocity_residual)

        if velocity_residual <= app_config.solver.target_residual:
            print("Velocity Residual Small enough")
            break

        plt.pause(0.00001)

    plt.ioff()
    print(iteration)

    # Velocity_Quiver
    plt.clf()
    plot_velocity_quiver_plot(simulation_grid)
    plt.savefig("../images/velocity_quiver.png")

    # Velocity Streamlines
    plt.clf()
    plot_velocity_streamlines(simulation_grid)
    plt.savefig("../images/velocity_streamlines.png")

    # Pressure Heatmap
    plt.clf()
    plot_pressure_heatmap(simulation_grid)
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

    simulate(simulation_grid, app_config, app_config.solver.time_step)
