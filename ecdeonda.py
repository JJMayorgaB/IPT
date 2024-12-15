import numpy as np
import imageio
import matplotlib.pyplot as plt

# Cargar los datos
deformacion = np.load("deformacion.npy")
x_vals = np.load("x_vals.npy")
t_vals = np.load("t_vals.npy")
y_vals = np.linspace(0, 5, 250)  # Definir el dominio en y

print(deformacion.shape)  # Esto imprimirá la forma de deformacion
print(x_vals.shape)       # Esto imprimirá la forma de x_vals
print(t_vals.shape)       # Esto imprimirá la forma de t_vals



# Definir parámetros del problema
L, a = 10, 5  # Tamaño del dominio en x e y
Nx, Ny = 50, 50  # Número de puntos en x e y
dx, dy = L / Nx, a / Ny  # Espaciado
dt = 0.001  # Paso temporal
v = 343000  # Velocidad de la onda de presión de sonido (cm/s)
rho = 0.000887  # Densidad del aire (g/cm^3)
v0 = 300  # Velocidad inicial (cm/s)
T_max = 60  # Tiempo total de simulación (s)
Nt = int(T_max / dt)  # Número de pasos de tiempo

# Definir el dominio espacial
x = np.linspace(0, L, Nx)
y = np.linspace(0, a, Ny)
t = np.linspace(0, T_max, Nt)

# Inicializar el campo de presión
p = np.zeros((Nx, Ny, Nt))

# Condiciones iniciales
p[:, :, 0] = 0  # p(x, y, 0) = 0
dp_dt = np.zeros((Nx, Ny))  # Derivada temporal inicial
dp_dt[Nx // 2, Ny // 2] = v0 / (dx * dy)  # Fuente puntual en el centro
p[:, :, 1] = p[:, :, 0] + dt * dp_dt  # Primer paso temporal

# Resolución por diferencias finitas explícitas
for n in range(1, Nt - 1):
    # Si deformacion es solo un campo espacial, no se usa en la derivada temporal.
    # Entonces, se elimina el cálculo de d2w_dt2 relacionado con deformacion.

    # Operador Laplaciano de p (campo de presión)
    laplacian = (
        np.roll(p[:, :, n], 1, axis=0) + np.roll(p[:, :, n], -1, axis=0) +
        np.roll(p[:, :, n], 1, axis=1) + np.roll(p[:, :, n], -1, axis=1) - 4 * p[:, :, n]
    ) / dx**2

    # Actualización de la ecuación de onda para el campo de presión
    p[:, :, n + 1] = (
        2 * p[:, :, n] - p[:, :, n - 1] + dt**2 * (v**2 * laplacian - rho * 0)  # No uso deformacion
    )

# Continuar con el resto del código para generar el GIF


# Parámetros para el GIF
num_frames = p.shape[2]  # Número total de frames (basado en la dimensión t)
intervalo = 10  # Intervalo de frames para el GIF

# Crear una lista para almacenar los frames
frames = []

# Crear el GIF
for t_index in range(0, num_frames, intervalo):
    p_at_t = p[:, :, t_index]  # Campo de presión en el tiempo t_index
    
    # Graficar el campo de presión en el plano x-y
    plt.figure(figsize=(10, 6))
    plt.contourf(x, y, p_at_t, 50, cmap='viridis')
    plt.colorbar(label="p(x, y, t)")
    plt.title(f'Campo de presión en el plano x-y, t={t_vals[t_index]:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Guardar la imagen en el frame
    plt.close()  # Cerrar la figura para evitar múltiples ventanas
    frames.append(plt.gcf())  # Agregar la figura al lista de frames

# Guardar el GIF usando imageio
gif_path = "evolucion_onda.gif"
imageio.mimsave(gif_path, frames, duration=0.1)  # Puedes ajustar 'duration' para cambiar la velocidad

print(f"GIF generado y guardado como {gif_path}")
