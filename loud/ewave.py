import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos y del problema
L = 10.0       # Longitud del dominio (en unidades de longitud)
T = 2.0        # Tiempo total (en unidades de tiempo)
v = 1.0        # Velocidad de propagación de la onda
A = 1.0        # Amplitud máxima de la fuente

# Parámetros de discretización
nx = 100       # Número de puntos espaciales
nt = 1000      # Número de pasos temporales
dx = L / (nx - 1)  # Espaciado espacial
dt = T / nt    # Paso temporal

# Estabilidad del esquema
c = v * dt / dx
if c >= 1:
    raise ValueError("El esquema no es estable, reduce dt o aumenta dx")

# Inicialización de las matrices
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
u = np.zeros((nt, nx))  # Solución
u_new = np.zeros(nx)    # Nueva solución temporal
u_old = np.zeros(nx)    # Solución en el paso anterior

source_array = np.zeros((nt, nx))  # Inicialización de la fuente
for n in range(nt):
    source_array[n, :] = A * np.sin(2 * np.pi * x / L) * np.cos(2 * np.pi * t[n] / T)  # Ejemplo de patrón fuente

# Condiciones iniciales
u[0, :] = np.zeros(nx)  # u(x, 0) = 0
u[1, :] = u[0, :]  # Derivada temporal inicial nula (onda en reposo)

# Bucle en el tiempo
for n in range(1, nt - 1):
    for i in range(1, nx - 1):
        # EDP discretizada con la fuente como array
        u[n+1, i] = (2 * u[n, i] - u[n-1, i]
                     + c**2 * (u[n, i+1] - 2 * u[n, i] + u[n, i-1])
                     + dt**2 * source_array[n, i])

    # Condiciones de frontera
    u[n+1, 0] = 0  # Frontera izquierda
    u[n+1, -1] = 0  # Frontera derecha

# Graficar los resultados
plt.figure(figsize=(8, 5))
for n in range(0, nt, nt // 10):
    plt.plot(x, u[n, :], label=f"t = {t[n]:.2f}")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Solución de la ecuación de onda con fuente discreta")
plt.legend()
plt.show()
