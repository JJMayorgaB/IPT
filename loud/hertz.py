import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros
r = 0.002 #m
dy = 0.05 #m
h = 0.0005 #m
E1 = 210e9
E2 = 2.8e9
dx = 0.142
m1 = 7850*0.195*np.pi*r**2
m2 = 1400*dy*h*dx
mu = (m1+m2)/(m1*m2)  # Valor de mu
print(mu)
k = (E1+E2)/(E1*E2)  # Valor de k
print(k)
print((k/mu)*0.887*0.0316)
v = 3   # Valor inicial de velocidad m/s

# Definir el sistema de ecuaciones diferenciales
def system(t, y):
    delta, v1 = y
    if delta < 0:  # Ajustar para evitar que delta sea negativo
        delta = 0  # Para asegurar que no tome valores negativos
    ddelta_dt = v1
    dv1_dt = -k * delta**(3/2) / mu
    return [ddelta_dt, dv1_dt]

# Condiciones iniciales
y0 = [0, v]  # [delta(0), v(0)]

# Resolver la ecuación diferencial
t_span = (0, 100)  # Intervalo de tiempo
t_eval = np.linspace(0, 100, 1000)  # Puntos en los que evaluar la solución
solution = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Extraer la solución
delta = solution.y[0]

# Graficar la solución
plt.figure(figsize=(8, 6))
plt.plot(solution.t, delta, label=r'$\delta(t)$', color='b')
plt.title(r'Solución de $\mu \frac{\partial^2 \delta}{\partial t^2} + k \delta^{3/2}(t) = 0$')
plt.xlabel('Tiempo t')
plt.ylabel(r'$\delta(t)$')
plt.grid(True)
plt.legend()
plt.show()
