import numpy as np
import dolfinx
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

# Problem Parameters
D = 1.0        # Bending stiffness
rho = 1.0      # Density
h = 0.01       # Plate thickness
a, b = 1.0, 1.0  # Domain dimensions
T = 2.0        # Total simulation time
dt = 0.01      # Time step

# Mesh and Function Space
comm = MPI.COMM_WORLD
msh = mesh.create_rectangle(comm, [[0, 0], [a, b]], [32, 32], 
                            cell_type=mesh.CellType.triangle)
V = fem.FunctionSpace(msh, ("Lagrange", 2))

# Trial and Test Functions
w = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Spatial Coordinate
x = ufl.SpatialCoordinate(msh)

# External Load 
q = ScalarType(4.0 * ufl.pi**4 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))

# Boundary Conditions
def boundary(x):
    return np.logical_or(
        np.logical_or(
            np.isclose(x[0], 0.0),
            np.isclose(x[0], a)
        ),
        np.logical_or(
            np.isclose(x[1], 0.0),
            np.isclose(x[1], b)
        )
    )

# Create boundary condition
bc = fem.dirichletbc(ScalarType(0), 
                     fem.locate_dofs_geometrical(V, boundary))

# Variational Problem Setup
a_form = D * ufl.inner(ufl.grad(ufl.grad(w)), ufl.grad(ufl.grad(v))) * ufl.dx
m_form = rho * h * ufl.inner(w, v) * ufl.dx
L_form = ufl.inner(q, v) * ufl.dx

# Prepare result storage
x_points = np.linspace(0, a, 100)  # x coordinates
y_points = np.linspace(0, b, 100)  # y coordinates
results = []

# Time-stepping
uh = fem.Function(V)
time_steps = int(T / dt)

for step in range(time_steps):
    t = step * dt
    
    # Solve linear problem
    problem = LinearProblem(a_form + m_form / dt**2, L_form, bcs=[bc])
    uh = problem.solve()
    
    # Interpolate solution at specific points
    w_values = []
    for x_val in x_points:
        for y_val in y_points:
            # Use interpolation method
            point_coords = np.array([[x_val, y_val, 0.0]])
            
            # Find cells containing the point
            cells = dolfinx.geometry.compute_collisions(msh, point_coords)
            candidate_cells = dolfinx.geometry.compute_colliding_cells(msh, cells, point_coords)
            
            # Interpolate if point is found in mesh
            if len(candidate_cells) > 0:
                w_at_point = uh.eval(point_coords)[0]
                w_values.append([w_at_point, t, x_val, y_val])
    
    # Append results for this time step
    if w_values:
        results.extend(w_values)

# Convert to numpy array
results_array = np.array(results)
np.save("w_values.npy", results_array)
print("Resultados guardados en 'w_values.npy'")
print("Resultados shape:", results_array.shape)

