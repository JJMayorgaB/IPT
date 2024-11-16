import numpy as np
import matplotlib.pyplot as plt

# Crear dos arrays de ejemplo
x = np.linspace(0, 10, 100)  # Array con 100 puntos entre 0 y 10
y = np.sin(x)  # Array con los valores de la función seno para cada valor en x

# Crear la figura y el eje
plt.figure(figsize=(8, 6))

# Graficar los arrays
plt.plot(x, y, label='sin(x)', color='b', linewidth=2)

# Añadir etiquetas y título
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gráfica de sin(x)')

# Mostrar leyenda
plt.legend()

# Mostrar la gráfica
plt.grid(True)
plt.show()
