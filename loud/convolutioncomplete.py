import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Definir el rango de x y t
x = np.linspace(0, 10, 500)  # Espacio
t = np.linspace(0, 60, 500)  # Tiempo
x0 = 9.0
t0 = 1.0

EI = 1.0
rho = 1.0
A = 1.0
deltaT = 0.1
r0 = 0.2
N = 0
F0 = 1


alpha = -0.5*(deltaT*(EI/(rho*A)))**2
beta = -2*(r0**2)

prefactor = -2*(np.pi)**2*F0*(deltaT/EI)

# Crear un grid para combinar x y t
X, T = np.meshgrid(x, t)

tau = X - x0

epsilon = 1e-6  # Pequeño valor para evitar división por cero
safe_tau = np.where(np.abs(tau) < epsilon, epsilon, tau)

f2 = np.exp(alpha * safe_tau**4) * np.exp(beta * safe_tau**2)

convolucion = np.zeros_like(T)

for i in range(N+1):

    f1 = np.sin((T - i*t0) * alpha * safe_tau**2) / (safe_tau**2)
    convolucion += np.array([convolve(f1_t, f2[:, 0], mode='same') for f1_t in f1])

convolucion *= prefactor

x_eval = 9.0
idx_x0 = np.abs(x - x_eval).argmin()  # Índice más cercano

# Extraer la función f(9, t) (columna correspondiente a x = 9)
f_x_t = convolucion[:, idx_x0]

# Graficar f(9, t) como función de t
plt.figure(figsize=(10, 6))
plt.plot(t, f_x_t, label=rf'$f({x_eval}, t)$')
plt.title(f"Convolución en x = {x_eval} como función de t")
plt.xlabel("Tiempo (t)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()
plt.show()

# Graficar el resultado como función de x y t
#plt.figure(figsize=(12, 6))
#plt.contourf(x, t, convolucion, levels=50, cmap='viridis')
#plt.colorbar(label="Amplitud")
#plt.title("Convolución como función de x y t")
#plt.xlabel("x")
#plt.ylabel("t")
#plt.grid()
#plt.show()

plt.imshow(convolucion, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', origin='lower', cmap='viridis')
plt.contourf(x, t, convolucion, levels=50, cmap='viridis')
plt.colorbar(label='Convolución')
plt.xlabel('Espacio (x)')
plt.ylabel('Tiempo (t)')
plt.title('Mapa de calor de la convolución')
plt.show()
