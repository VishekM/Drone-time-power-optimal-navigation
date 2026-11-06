"""Plot the initial state of the drone navigation problem.

The drone starts in the bottom right corner and has to reach the top left
corner of a 2D workspace, with obstacles in between. No path is drawn.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

# Workspace bounds and start/goal positions (metres)
X_MAX = 100.0
Y_MAX = 100.0
START = (92.0, 8.0)
GOAL = (8.0, 92.0)

INK = "#1f2933"
MUTED = "#6b7280"
OBSTACLE_FILL = "#9aa5b1"
START_COLOR = "#1d4ed8"
GOAL_COLOR = "#b45309"


def make_obstacles():
    """Return the obstacles as matplotlib patches."""
    return [
        Circle((70, 28), 9),
        Rectangle((38, 12), 16, 20),
        Polygon([(78, 48), (94, 54), (88, 70), (72, 64)]),
        Ellipse((50, 52), 26, 13, angle=-35),
        Polygon([(18, 34), (32, 40), (28, 56), (12, 52)]),
        Rectangle((56, 76), 22, 10, angle=15),
        Circle((28, 76), 7),
    ]


def draw_drone(ax, center, arm=3.2, rotor=1.5):
    """Draw a top-down quadrotor: an X frame, four rotors and a body."""
    cx, cy = center
    for dx, dy in [(-arm, -arm), (-arm, arm), (arm, -arm), (arm, arm)]:
        ax.plot([cx, cx + dx], [cy, cy + dy], color=INK, lw=2, zorder=5)
        ax.add_patch(
            Circle((cx + dx, cy + dy), rotor, fc="white", ec=INK, lw=1.5, zorder=6)
        )
    ax.add_patch(Circle(center, 1.3, fc=START_COLOR, ec=INK, lw=1.5, zorder=7))


def main():
    fig, ax = plt.subplots(figsize=(8, 8))

    for obstacle in make_obstacles():
        obstacle.set(fc=OBSTACLE_FILL, ec=INK, lw=1, zorder=2)
        ax.add_patch(obstacle)

    # Start: a zone around the drone. Goal: a target with a star at its centre.
    ax.add_patch(
        Circle(START, 7, fc=START_COLOR, alpha=0.15, ec="none", zorder=3)
    )
    ax.add_patch(Circle(START, 7, fill=False, ec=START_COLOR, lw=2, zorder=3))
    draw_drone(ax, START)

    ax.add_patch(Circle(GOAL, 7, fc=GOAL_COLOR, alpha=0.15, ec="none", zorder=3))
    ax.add_patch(Circle(GOAL, 7, fill=False, ec=GOAL_COLOR, lw=2, zorder=3))
    ax.plot(*GOAL, marker="*", ms=20, mfc=GOAL_COLOR, mec=INK, mew=1, zorder=4)

    ax.annotate(
        "START",
        xy=START,
        xytext=(START[0] - 9, START[1]),
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    ax.annotate(
        "GOAL",
        xy=GOAL,
        xytext=(GOAL[0] + 9, GOAL[1]),
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )

    # Obstacles get a proxy entry so the grey shapes are identified on the plot
    ax.legend(
        handles=[Rectangle((0, 0), 1, 1, fc=OBSTACLE_FILL, ec=INK, lw=1, label="Obstacle")],
        loc="upper right",
        frameon=False,
    )

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=MUTED)
    ax.set_ylabel("y (m)", color=MUTED)
    ax.set_title("Drone initial state", color=INK, fontsize=14, loc="left")
    ax.tick_params(colors=MUTED)
    ax.grid(color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig("drone_initial_state.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
