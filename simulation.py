# Connect two identical linear springs symmetrically to a mass in a “V” shape, and apply an adjustable force to the mass. When this force is varied, the resulting motion of the mass depends on the history of changes in the applied force under certain conditions. Investigate this phenomenon.

# Packages
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp

# Fundamentals
import os
import time
import argparse

# Modules
import Graphing

# Default Values
timestep = 1 / 240
p1 = np.array([-0.8, 0])
p2 = np.array([0.8, 0])

relaxed = 1.6
k = 10
mass = 1
force = 0
damping = 0.5

# Flags
animate = False
save_anim = False

# Plotting
graphs = [Graphing.Coordinates, Graphing.Energy]


@njit(cache=True)
def magn(vec: np.ndarray):
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


@njit(cache=True)
def unit(vec: np.ndarray):
    return vec / magn(vec)


@njit(cache=True)
def step(t, state: np.ndarray):
    p, v = state[0:2], state[2:4]
    f = -k * (2 * p - p1 - p2 - relaxed * (unit(p - p1) + unit(p - p2)))
    f_d = v * damping

    return np.array([v[0], v[1], (f[0] - f_d[0]) / mass, (f[1] - f_d[1]) / mass])


def solve(y0):
    t0 = time.perf_counter()

    time_array = np.linspace(0, runtime, int(runtime / timestep))

    returned = solve_ivp(step, [0, runtime], y0, t_eval=time_array)
    solve = returned.y.T

    if __name__ == "__main__":
        print("Solved ODE:", int(1e3 * (time.perf_counter() - t0)), "ms\n")

        if len(graphs) > 0:
            fig, axis = plt.subplots(len(graphs))
            if len(graphs) == 1:
                axis = [axis]
            axis[0].set_title("Spring Hysteresis [MODEL]")

            for i, func in enumerate(graphs):
                func(
                    axis[i],
                    solve=solve,
                    timestep=timestep,
                    time=time_array,
                    mass=mass,
                    k=k,
                    relaxed=relaxed,
                    p1=p1,
                    p2=p2,
                )

            plt.tight_layout()
            plt.show()

        if animate:
            Graphing.animate_simulation(solve, time_array, p1, p2, save_anim=save_anim)
    else:
        return solve, time_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[Spring Hysteresis Simulation IYPT 2025]"
    )

    parser.add_argument(
        "-l",
        "--length",
        help="Runtime of the simulation",
        action="store",
        type=float,
        default=8,
    )
    parser.add_argument(
        "-a", "--animate", help="Show the animation", action="store_true"
    )
    parser.add_argument("-s", "--save", help="Save the animation", action="store_true")
    parser.add_argument(
        "-v", "--version", action="version", version="[Spring Hysteresis] v0.1a"
    )
    os.environ["QT_LOGGING_RULES"] = "qt.qpa.wayland.textinput=false"

    args = parser.parse_args()
    save_anim = args.save
    animate = args.animate
    runtime = args.length

    print(
        "\n\033[1mSpring Hysteresis Numerical Solution\033[0m\nKindly wait while the simulation is run!\n"
    )

    while True:
        print("\n\033[1mInput initial conditions (x, y, x',y') or [x] to exit:\033[0m")
        solve(np.array(input().split(), dtype="float"))
