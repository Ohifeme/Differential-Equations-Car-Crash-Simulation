# Car Collision Simulation — Damped Spring System

A differential equations project that models a car collision as a **damped spring system**, solved numerically using **Euler's method** and visualized through graphs and animation.

---

## The Math Model

The collision is represented by the second-order ODE:

```
m·x''(t) + c·x'(t) + k·x(t) = 0
```

| Variable | Description |
|----------|-------------|
| `m` | Mass of the vehicle (kg) |
| `c` | Damping coefficient — energy loss from crumpling (N·s/m) |
| `k` | Stiffness coefficient — spring-like resistance during impact (N/m) |
| `x(t)` | Displacement / compression of the car during collision (m) |
| `x'(t)` | Velocity during collision (m/s) |
| `x''(t)` | Acceleration during collision (m/s²) |

This is rewritten as a system of first-order ODEs and solved step-by-step using **Euler's method**:

```
x'(t) = v(t)
v'(t) = -(c/m)·v(t) - (k/m)·x(t)
```

---

## Dependencies

Make sure you have **Python 3.7+** installed, then install the required libraries:

```bash
pip install numpy matplotlib pillow
```

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical arrays and math |
| `matplotlib` | Graphs and animation |
| `pillow` | Saving the animation as a GIF |

---

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**

```bash
pip install numpy matplotlib pillow
```

**3. Run the simulation**

```bash
python car_collision_simulation.py
```

---

## User Input

When you run the program, it will prompt you to enter simulation parameters. You can type your own values or **press Enter to use the defaults**:

```
  Vehicle mass (kg)                  [default: 1500.0]:
  Collision stiffness k (N/m)        [default: 50000.0]:
  Damping coefficient c (N·s/m)      [default: 4000.0]:
  Initial compression x0 (m)        [default: 0.5]:
  Initial velocity into collision    [default: -15.0]:
  Simulation duration (seconds)      [default: 2.0]:
  Time step dt (seconds)             [default: 0.001]:
```

---

## Outputs

After running, the simulation produces:

### 1. Terminal Summary
Prints key statistics including peak compression, max/min velocity, peak force, and g-force experienced during the collision.

### 2. `collision_graphs.png`
A 3-panel graph saved to your current directory showing:
- **Position** x(t) — how much the car compresses over time
- **Velocity** v(t) — how the car slows during impact
- **Acceleration** a(t) — the forces experienced during the crash

### 3. `collision_animation.gif`
An animation of two cars colliding, with a spring visualizing the compression between them. Saved to your current directory.

---

## File Structure

```
your-repo/
│
├── car_collision_simulation.py   # Main simulation file
├── README.md                     # This file
├── collision_graphs.png          # Generated after running (graphs)
└── collision_animation.gif       # Generated after running (animation)
```

---

## Troubleshooting

**GIF not saving?**
Make sure `pillow` is installed:
```bash
pip install pillow
```

**Graphs not displaying?**
If you're on a headless server or WSL without a display, add this near the top of the script:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

**Wrong Python version?**
Check your version with:
```bash
python --version
```
This project requires Python 3.7 or higher.

---

## Project Context

This simulation is part of a group differential equations project analyzing car collisions from multiple perspectives — computer science, actuarial science, robotics, and biomedical engineering. The CS contribution focuses on numerical simulation and data visualization using programming to model real-world physics.
