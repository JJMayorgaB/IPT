import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def solve_wave_equation(Lx, Lt, v, a, nx, nt, bc_type='zero'):
    """
    Solve the wave equation: ∂2u/∂x² - (1/v²)∂2u/∂t² = -ax²

    Parameters:
    Lx (float): Length of spatial domain
    Lt (float): Total time of simulation
    v (float): Wave velocity
    a (float): Coefficient of the source term
    nx (int): Number of spatial grid points
    nt (int): Number of time steps
    bc_type (str): Boundary condition type ('zero' or 'periodic')

    Returns:
    tuple: (x, t, u) - spatial grid, time grid, and solution matrix
    """
    # Grid spacing
    dx = Lx / (nx - 1)
    dt = Lt / (nt - 1)

    # Create grids
    x = np.linspace(0, Lx, nx)
    t = np.linspace(0, Lt, nt)

    # Initialize solution array
    u = np.zeros((nt, nx))

    # Source term
    source = -a - (3/8)*a*(x-1)**2

    # Stability condition
    stability_condition = (v * dt / dx)**2
    print(f"Stability condition (should be ≤ 1): {stability_condition}")

    # Initial conditions (assuming zero initial displacement and velocity)
    u[0, :] = 0

    # First time step (using forward difference)
    u[1, :] = u[0, :] + dt * np.zeros_like(x)

    # Finite difference method
    for n in range(1, nt - 1):
        for i in range(1, nx - 1):
            # Central difference approximations
            d2u_dx2 = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2

            # Wave equation discretization
            u[n+1, i] = (2*u[n, i] - u[n-1, i] + 
                         (v*dt)**2 * d2u_dx2 + 
                         dt**2 * source[i])

        # Boundary conditions
        if bc_type == 'zero':
            # Dirichlet boundary conditions (zero at boundaries)
            u[n+1, 0] = 0
            u[n+1, -1] = 0
        elif bc_type == 'periodic':
            # Periodic boundary conditions
            u[n+1, 0] = u[n+1, -2]
            u[n+1, -1] = u[n+1, 1]

    return x, t, u

Lx = 10.0    # Length of spatial domain
Lt = 20.0     # Total simulation time
v = 1.0      # Wave velocity
a = 1e-14    # Coefficient of source term
nx = 100     # Spatial grid points
nt = 500     # Time steps

# Solve the wave equation
x, t, u = solve_wave_equation(Lx, Lt, v, a, nx, nt)

# Create animation
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, Lx)
ax.set_ylim(np.min(u), np.max(u))
ax.set_title('Wave Equation Animation')
ax.set_xlabel('Spatial Position')
ax.set_ylabel('Amplitude')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x, u[frame, :])
    ax.set_title(f"Wave Profile at t = {t[frame]:.2f}")
    return line,

# Adjust frames per second (fps) here
fps = 150
ani = FuncAnimation(fig, update, frames=range(0, nt, 2), init_func=init, blit=True)

# Save the animation as a GIF
ani.save('wave_equation_animation.gif', fps=fps, writer='imagemagick')

# Display the plot
plt.show()
