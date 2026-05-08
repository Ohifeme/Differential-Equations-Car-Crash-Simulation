"""
Car Collision Simulation - Damped Spring System
Differential Equations Project

Models a car collision using the second-order ODE:
    m * x''(t) + c * x'(t) + k * x(t) = 0

Where:
    m = mass of the vehicle (kg)
    c = damping coefficient (represents energy loss from crumpling)
    k = stiffness coefficient (spring-like resistance during impact)
    x(t) = displacement (compression) of the car during collision (m)
    x'(t) = velocity during collision (m/s)
    x''(t) = acceleration during collision (m/s^2)

Rewritten as a system of first-order ODEs for Euler's method:
    Let v = x'(t)
    x'(t) = v
    v'(t) = -(c/m)*v - (k/m)*x
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec


# ──────────────────────────────────────────────
#  USER INPUT
# ──────────────────────────────────────────────

def get_user_input():
    print("=" * 55)
    print("   CAR COLLISION SIMULATION - Damped Spring System")
    print("=" * 55)
    print("\nEnter simulation parameters (or press Enter for defaults):\n")

    def prompt(msg, default):
        val = input(f"  {msg} [default: {default}]: ").strip()
        return float(val) if val else default

    m = prompt("Vehicle mass (kg)", 1500.0)
    k = prompt("Collision stiffness k (N/m)", 50000.0)
    c = prompt("Damping coefficient c (N·s/m)", 4000.0)
    x0 = prompt("Initial compression x0 (m)", 0.5)
    v0 = prompt("Initial velocity into collision v0 (m/s)", -15.0)
    t_end = prompt("Simulation duration (seconds)", 2.0)
    dt = prompt("Time step dt (seconds)", 0.001)

    print("\n  Parameters accepted. Running simulation...\n")
    return m, k, c, x0, v0, t_end, dt


# ──────────────────────────────────────────────
#  EULER'S METHOD SOLVER
# ──────────────────────────────────────────────

def eulers_method(m, k, c, x0, v0, t_end, dt):
    """
    Numerically solves the damped spring ODE using Euler's method.

    System:
        x'(t) = v(t)
        v'(t) = -(c/m)*v(t) - (k/m)*x(t)

    Returns arrays of time, position, velocity, and acceleration.
    """
    t_values = np.arange(0, t_end + dt, dt)
    n = len(t_values)

    x = np.zeros(n)   # position (compression)
    v = np.zeros(n)   # velocity
    a = np.zeros(n)   # acceleration

    # Initial conditions
    x[0] = x0
    v[0] = v0
    a[0] = -(c / m) * v0 - (k / m) * x0

    # Euler's method: step forward in time
    for i in range(1, n):
        a[i - 1] = -(c / m) * v[i - 1] - (k / m) * x[i - 1]
        v[i] = v[i - 1] + a[i - 1] * dt
        x[i] = x[i - 1] + v[i - 1] * dt

    # Compute final acceleration value
    a[-1] = -(c / m) * v[-1] - (k / m) * x[-1]

    return t_values, x, v, a


# ──────────────────────────────────────────────
#  GRAPHS: Position, Velocity, Acceleration
# ──────────────────────────────────────────────

def plot_graphs(t, x, v, a, m, k, c):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"Car Collision Simulation — Damped Spring System\n"
        f"m={m} kg  |  k={k} N/m  |  c={c} N·s/m",
        fontsize=13, fontweight='bold'
    )

    # Position
    axes[0].plot(t, x, color='steelblue', linewidth=2)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel("Position x(t)  [m]", fontsize=11)
    axes[0].set_title("Displacement (Compression) Over Time", fontsize=10)
    axes[0].fill_between(t, x, 0, alpha=0.12, color='steelblue')
    axes[0].grid(True, alpha=0.3)

    # Velocity
    axes[1].plot(t, v, color='tomato', linewidth=2)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel("Velocity v(t)  [m/s]", fontsize=11)
    axes[1].set_title("Velocity Over Time", fontsize=10)
    axes[1].fill_between(t, v, 0, alpha=0.12, color='tomato')
    axes[1].grid(True, alpha=0.3)

    # Acceleration
    axes[2].plot(t, a, color='mediumseagreen', linewidth=2)
    axes[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[2].set_ylabel("Acceleration a(t)  [m/s²]", fontsize=11)
    axes[2].set_title("Acceleration Over Time", fontsize=10)
    axes[2].fill_between(t, a, 0, alpha=0.12, color='mediumseagreen')
    axes[2].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time  [s]", fontsize=11)

    plt.tight_layout()
    plt.savefig("collision_graphs.png", dpi=150, bbox_inches='tight')
    print("  [Saved] collision_graphs.png")
    plt.show()


# ──────────────────────────────────────────────
#  ANIMATION
# ──────────────────────────────────────────────

def animate_collision(t, x, v):
    """
    Animates the car collision as two rectangles.
    The moving car's position is driven by the simulated displacement x(t).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Static car (wall / target car) — fixed on right
    static_car = Rectangle((7, 0.5), 2.5, 1.2, color='#e94560', zorder=3)
    ax.add_patch(static_car)

    # Moving car — starts on left, moves right
    moving_car = Rectangle((1, 0.5), 2.5, 1.2, color='#0f3460', zorder=3,
                            edgecolor='#16213e', linewidth=2)
    ax.add_patch(moving_car)

    # Wheels
    wheel_color = '#888'
    wheels = []
    for cx in [1.5, 3.0]:
        w = plt.Circle((cx, 0.5), 0.25, color=wheel_color, zorder=4)
        ax.add_patch(w)
        wheels.append(w)
    static_wheels = []
    for cx in [7.5, 9.0]:
        w = plt.Circle((cx, 0.5), 0.25, color=wheel_color, zorder=4)
        ax.add_patch(w)
        static_wheels.append(w)

    # Spring (collision zone)
    spring_line, = ax.plot([], [], color='#f5a623', linewidth=2, zorder=5)

    # Info text
    time_text = ax.text(0.02, 0.92, '', transform=ax.transAxes,
                        color='white', fontsize=10)
    vel_text = ax.text(0.02, 0.82, '', transform=ax.transAxes,
                       color='#aaaaff', fontsize=10)

    ax.set_title("Car Collision Animation — Damped Spring Model",
                 color='white', fontsize=12, pad=10)

    # Normalize position: map x(t) to screen coords
    # Car starts at x=1, moves right. Collision begins when car front (x+2.5) hits static car left (7)
    # So base position of moving car = 1 + (max_x - x(t)) scaled
    x_norm = x - x.min()
    x_norm = x_norm / (x_norm.max() + 1e-9)  # 0 to 1

    # Subsample for animation speed
    step = max(1, len(t) // 300)
    frames = range(0, len(t), step)

    def draw_spring(x1, x2, y=1.1, n_coils=8):
        """Draw a zigzag spring between x1 and x2."""
        xs = np.linspace(x1, x2, n_coils * 2 + 2)
        ys = np.zeros_like(xs)
        ys[1:-1:2] = y + 0.15
        ys[2:-1:2] = y - 0.15
        ys[0] = y
        ys[-1] = y
        return xs, ys

    def update(frame):
        # Moving car x position: approaches from left, compresses, bounces back
        progress = x_norm[frame]
        car_x = 1 + progress * 3.5  # moves from x=1 to ~x=4.5

        moving_car.set_x(car_x)
        for i, w in enumerate(wheels):
            w.center = (car_x + 0.5 + i * 1.5, 0.5)

        # Spring between front of moving car and back of static car
        front = car_x + 2.5
        back = 7.0
        if front < back:
            spring_line.set_data([], [])
        else:
            sx, sy = draw_spring(back, front)
            spring_line.set_data(sx, sy)

        time_text.set_text(f"Time: {t[frame]:.3f} s")
        vel_text.set_text(f"Velocity: {v[frame]:.2f} m/s")
        return moving_car, spring_line, time_text, vel_text, *wheels

    ani = animation.FuncAnimation(
        fig, update, frames=frames,
        interval=20, blit=True, repeat=False
    )

    try:
        ani.save("collision_animation.gif", writer='pillow', fps=30)
        print("  [Saved] collision_animation.gif")
    except Exception:
        print("  [Note] Could not save GIF (pillow not installed). Showing live animation instead.")

    plt.tight_layout()
    plt.show()
    return ani


# ──────────────────────────────────────────────
#  SUMMARY STATS
# ──────────────────────────────────────────────

def print_summary(t, x, v, a, m, k, c, dt):
    print("\n" + "=" * 55)
    print("  SIMULATION SUMMARY")
    print("=" * 55)
    print(f"  Vehicle Mass:          {m} kg")
    print(f"  Stiffness (k):         {k} N/m")
    print(f"  Damping (c):           {c} N·s/m")
    print(f"  Time Step (dt):        {dt} s")
    print(f"  Total Steps:           {len(t)}")
    print("-" * 55)
    print(f"  Max Compression:       {x.max():.4f} m")
    print(f"  Min Compression:       {x.min():.4f} m")
    print(f"  Max Velocity:          {v.max():.4f} m/s")
    print(f"  Min Velocity:          {v.min():.4f} m/s")
    print(f"  Peak Acceleration:     {a.max():.4f} m/s²")
    print(f"  Peak Deceleration:     {a.min():.4f} m/s²")
    print(f"  Peak Force on Car:     {abs(a.min()) * m:.2f} N  ({abs(a.min()) * m / 9.81:.1f} g-force)")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    # Get parameters from user
    m, k, c, x0, v0, t_end, dt = get_user_input()

    # Solve using Euler's method
    t, x, v, a = eulers_method(m, k, c, x0, v0, t_end, dt)

    # Print summary statistics
    print_summary(t, x, v, a, m, k, c, dt)

    # Plot position, velocity, acceleration graphs
    print("  Generating graphs...")
    plot_graphs(t, x, v, a, m, k, c)

    # Run animation
    print("  Running animation...")
    animate_collision(t, x, v)

    print("  Done! All outputs saved.\n")


if __name__ == "__main__":
    main()
