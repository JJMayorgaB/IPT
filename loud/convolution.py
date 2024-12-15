import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Definir el rango de x y t
x = np.linspace(0, 10, 200)  # Espacio
t = np.linspace(0, 10, 1200)  # Tiempo
x0 = 9
alpha = 1
t_0 = 1

# Crear un grid para combinar x y t
X, T = np.meshgrid(x, t)

# Definir las funciones f1 y f2
f1 = np.sin((T - t_0) * alpha * (X - x0)**2)  # Función de x y t
f2 = (X - x0)**(-2)  # Solo depende de x

# Realizar la convolución en cada línea temporal
convolucion = np.array([convolve(f1_t, f2[0], mode='same') for f1_t in f1])

idx_x0 = np.abs(x - 0).argmin()  # Índice más cercano

# Extraer la función f(9, t) (columna correspondiente a x = 9)
f_5_t = convolucion[:, idx_x0]

# Graficar f(9, t) como función de t
#plt.figure(figsize=(10, 6))
#plt.plot(t, f_5_t, label=r'$f(0, t)$')
#plt.title("Convolución en x = 0 como función de t")
#plt.xlabel("Tiempo (t)")
#plt.ylabel("Amplitud")
#plt.legend()
#plt.grid()
#plt.show()

# Graficar el resultado como función de x y t
plt.figure(figsize=(12, 6))
plt.contourf(x, t, convolucion, levels=50, cmap='viridis')
plt.colorbar(label="Amplitud")
plt.title("Convolución como función de x y t")
plt.xlabel("x")
plt.ylabel("t")
plt.grid()
plt.show()
