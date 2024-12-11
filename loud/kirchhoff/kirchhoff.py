import numpy as np
import dolfinx
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import div, grad, dx, inner, pi, sin, SpatialCoordinate

D = 1.0  # Rigidez de flexión
rho = 1.0  # Densidad de la placa
h = 0.01  # Espesor de la placa
a, b = 1.0, 1.0  # Dimensiones del dominio
T = 2.0  # Tiempo total de simulación
dt = 0.01  # Paso temporal

msh = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [a, b]], [32, 32], cell_type=mesh.CellType.triangle)
V = fem.FunctionSpace(msh, ("Lagrange", 2))

w = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

x = SpatialCoordinate(msh)
q = ScalarType(4.0 * pi**4 * sin(pi * x[0]) * sin(pi * x[1]))

a = D * inner(grad(grad(w)), grad(grad(v))) * dx  # Término de flexión
m = rho * h * inner(w, v) * dx  # Término de masa
L = inner(q, v) * dx  # Carga externa

def boundary(x):
    return np.full_like(x[0], 0.0)

bc = fem.dirichletbc(value=ScalarType(0), dofs=fem.locate_dofs_geometrical(V, boundary))

uh = fem.Function(V)

x_points = np.linspace(0, a, 100)  # Coordenadas x
y_points = np.linspace(0, b, 100)  # Coordenadas y

results = []

time_steps = int(T / dt)

for step in range(time_steps):
    t = step * dt
    
    problem = LinearProblem(a + m / dt**2, L, bcs=[bc])
    uh = problem.solve()
    
    for x_val in x_points:
        for y_val in y_points:
            
            w_at_point = uh.eval([x_val, y_val])
            
            results.append((w_at_point, t, x_val, y_val))

results_array = np.array(results)

np.save("w_values.npy", results_array)
print("Resultados guardados en 'w_values.npy'")
