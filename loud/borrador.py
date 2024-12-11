import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Parámetros físicos
L = 10.0  # Longitud espacial
T = 5.0   # Tiempo máximo
N = 5     # Número de términos de la suma
delta_t = 0.5  # Escala temporal
rho = 1.0  # Densidad lineal
A = 1.0  # Área transversal
EI = 1.0  # Rigidez flexional
x_0 = 5.0  # Posición de referencia espacial
t_i = np.linspace(0, T, N)  # Tiempos discretos
r_0 = 1.0  # Parámetro de decaimiento en el dominio k

# Función para integrar sobre k
def k_integrand(k, x, x_0, EI, rho, A, r_0, omega):
    numerator = np.exp(-2 / r_0**2 * k**2) * np.exp(1j * k * (x - x_0))
    denominator = (EI / (rho * A)) * k**4 - omega**2
    return numerator / denominator

# Función para integrar sobre ω
def omega_integrand(omega, t, x, x_0, EI, rho, A, r_0, delta_t, t_i):
    k_integral, _ = quad(
        lambda k: k_integrand(k, x, x_0, EI, rho, A, r_0, omega),
        -np.inf, np.inf
    )
    return np.exp(-delta_t**2 * omega**2 / 2) * np.exp(-1j * omega * (t - t_i)) * k_integral

# Función total w(x, t)
def compute_w(x, t, delta_t, rho, A, EI, r_0, x_0, t_i):
    w_total = 0
    for ti in t_i:
        omega_integral, _ = quad(
            lambda omega: omega_integrand(omega, t, x, x_0, EI, rho, A, r_0, delta_t, ti),
            -np.inf, np.inf
        )
        w_total += omega_integral
    return (2 * np.pi * delta_t / (rho * A)) * w_total

# Crear malla de puntos y evaluar w(x, t)
x_values = np.linspace(0, L, 100)
t_values = np.linspace(0, T, 100)

# Calcular w(x, t) en L para todos los valores de t
w_temporal = np.array([compute_w(0.9*L, t, delta_t, rho, A, EI, r_0, x_0, t_i) for t in t_values])

# Calcular w(x, t) en tres valores específicos de t (0, t_i[1], 2.5 * t_i[1])
t_specific = [0, t_i[1], 2.5 * t_i[1]]
w_spatial = {t: np.array([compute_w(x, t, delta_t, rho, A, EI, r_0, x_0, t_i) for x in x_values]) for t in t_specific}

# Graficar
plt.figure(figsize=(15, 8))

# Parte temporal en x = L
plt.subplot(2, 1, 1)
plt.plot(t_values, np.real(w_temporal), label="Parte real")
plt.plot(t_values, np.imag(w_temporal), label="Parte imaginaria")
plt.title(f"Parte temporal de w(x, t) en x = {L}")
plt.xlabel("t")
plt.ylabel("w(x, t)")
plt.legend()
plt.grid()

# Parte espacial en t = 0, t = t_i[1], y t = 2.5t_i[1]
plt.subplot(2, 1, 2)
for t, w in w_spatial.items():
    plt.plot(x_values, np.real(w), label=f"t = {t:.2f} (Parte real)")
    plt.plot(x_values, np.imag(w), linestyle='--', label=f"t = {t:.2f} (Parte imaginaria)")
plt.title("Parte espacial de w(x, t)")
plt.xlabel("x")
plt.ylabel("w(x, t)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
