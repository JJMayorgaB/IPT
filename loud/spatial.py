import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Definir los parámetros
a = 0.1  # Ajusta según el problema
c = 1.0  # Ajusta según el problema
d = 1.0  # Ajusta según el problema
omega = 1.0  # Ajusta según el problema
b = 9.0
L = 10.0 #Longitud de la placa

# Función integrando
def integrand(k, x, a, c, d, omega, b):
    numerator = np.exp(-(2 / a**2) * k**2) * np.exp(1j * k * (x - b))
    denominator = c * k**4 - d * omega**2
    return numerator / denominator

# Límite para aproximar infinito
k_max = 150

# Función para calcular la integral numérica
def compute_w(x, a, c, d, omega, b):
    # Separar en partes real e imaginaria
    result_real, _ = quad(lambda k: np.real(integrand(k, x, a, c, d, omega, b)), -k_max, k_max)
    result_imag, _ = quad(lambda k: np.imag(integrand(k, x, a, c, d, omega, b)), -k_max, k_max)
    return result_real + 1j * result_imag

# Rango de x para graficar
x_values = np.linspace(0,L+10,1000)
 
w_values = [compute_w(x, a, c, d, omega, b) for x in x_values]

# Graficar la función obtenida
plt.figure(figsize=(10, 6))
plt.plot(x_values, np.real(w_values), label="Parte real de w(x)", color="blue")
plt.plot(x_values, np.imag(w_values), label="Parte imaginaria de w(x)", color="red")
plt.xlabel("x")
plt.ylabel("w(x)")
plt.title("Gráfica de la función w(x)")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.grid()
plt.show()
