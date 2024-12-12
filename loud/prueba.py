import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# Definir el integrando
def integrand(x):
    return np.exp(-x**2)

# Rango de integración (por ser impropia, elegimos un rango suficientemente grande)
x_vals = np.linspace(-10000, 10000, 200000)  # Malla en el dominio de integración

# Evaluar el integrando
y_vals = integrand(x_vals)

integral_result = simpson(y=y_vals, x=x_vals)

print(f"El valor de la integral es aproximadamente: {integral_result}")
