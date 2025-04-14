import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy.linalg as la
import matplotlib.patches as patches


def Coordinates(ax, **args):
    solve = args["solve"]
    time = args["time"]
    ax.plot(time, solve[:, 0], label="X")
    ax.plot(time, solve[:, 1], label="Y")
    ax.legend()


def Energy(ax, **args):
    solve = args["solve"]
    time = args["time"]
    timestep = args["timestep"]

    mass, k, relaxed, p1, p2 = (
        args["mass"],
        args["k"],
        args["relaxed"],
        args["p1"],
        args["p2"],
    )

    ek = 0.5 * mass * la.norm(solve[:, 2:4], axis=1) ** 2

    d1 = la.norm(solve[:, 0:2] - p1, axis=1) - relaxed
    d2 = la.norm(solve[:, 0:2] - p2, axis=1) - relaxed
    es = 0.5 * k * (d1**2 + d2**2)

    ax.plot(time, ek + es, label="Energy")
    ax.legend()


def animate_simulation(solve, time_array, p1, p2, save_anim=False):
    # Extract position data
    positions = solve[:, 0:2]  # x, y coordinates of the mass

    # Create figure and axis
    fig, ax = plt.subplots()

    # Set limits to accommodate the spring system
    margin = 0.1
    max_y = max(abs(positions[:, 1])) + margin

    ax.set_xlim(p1[0] - 0.1, p2[0] + 0.1)
    ax.set_ylim(-max_y, max_y)

    # Fixed points for the springs

    # Create plot elements
    springs = []
    springs.append(ax.plot([], [], "-", lw=2)[0])  # left spring
    springs.append(ax.plot([], [], "-", lw=2)[0])  # right spring

    # Create mass (as a circle)
    mass_point = patches.Circle((0, 0), 0.05, fc="r", ec="k", zorder=3)
    ax.add_patch(mass_point)

    # Fixed points (as squares)
    ax.add_patch(patches.Rectangle(p1 - 0.03, 0.06, 0.06, fc="gray", ec="k", zorder=2))
    ax.add_patch(patches.Rectangle(p2 - 0.03, 0.06, 0.06, fc="gray", ec="k", zorder=2))

    # Trajectory line
    (trajectory,) = ax.plot([], [], "g-", alpha=0.5, lw=1)

    # Set up the plot
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Spring Hysteresis Simulation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Initialize zigzag lines for springs
    def spring_points(start, end, segments=20):
        direction = end - start
        length = np.linalg.norm(direction)
        unit_direction = direction / length

        # Create perpendicular vector for zigzag
        perp = np.array([-unit_direction[1], unit_direction[0]])

        # Create zigzag points
        points = []
        points.append(start)

        zigzag_amplitude = 0.03  # Scale with spring length

        for i in range(1, segments):
            t = i / segments
            pos = start + direction * t

            # Add zigzag pattern (alternating perpendicular displacement)
            if i % 2:
                pos = pos + perp * zigzag_amplitude
            else:
                pos = pos - perp * zigzag_amplitude

            points.append(pos)

        points.append(end)
        return np.array(points)

    # Animation function
    def animate(i):
        # Get current position
        pos = positions[i]

        # Update mass position
        mass_point.center = pos

        # Update springs
        spring1_points = spring_points(p1, pos)
        spring2_points = spring_points(p2, pos)

        springs[0].set_data(spring1_points[:, 0], spring1_points[:, 1])
        springs[1].set_data(spring2_points[:, 0], spring2_points[:, 1])

        start_idx = max(0, i - 200)
        trajectory.set_data(
            positions[start_idx : i + 1, 0], positions[start_idx : i + 1, 1]
        )

        return springs + [mass_point, trajectory]

    # Create animation
    frames = min(500, len(time_array))  # Limit frames for performance
    step = len(time_array) // frames if len(time_array) > frames else 1

    anim = FuncAnimation(
        fig, animate, frames=range(0, len(time_array), step), interval=30, blit=True
    )

    # Save animation if requested
    if save_anim:
        anim.save("spring_hysteresis.mp4", writer="ffmpeg", fps=30, dpi=100)

    plt.tight_layout()
    plt.show()

    return anim
