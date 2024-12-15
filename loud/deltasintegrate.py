import numpy as np
from scipy.integrate import simpson  
import matplotlib.pyplot as plt 

def integrand(k_vals, x, t, EI, rho, A, x_o, t_i):

    K, X, T = np.meshgrid(k_vals, x, t)

    with np.errstate(divide='ignore', invalid='ignore'):

        term1 = np.sin(np.sqrt(EI / (rho * A)) * K**2 * (T - t_i)) / K**2
        term4 = np.exp(1j * K * (X - x_o))
    
    return term1 * term4


def w_function(x_vals, t_vals, EI, rho, A, x_o, N, t_1):

    k_vals_positive = np.logspace(-2, 3, 10000)  # Escala logarítmica
    k_vals_negative = -k_vals_positive
    k_vals_full = np.concatenate((k_vals_negative, k_vals_positive))

    prefactor = (-np.pi) / EI

    w_results = np.zeros((len(x_vals), len(t_vals)), dtype=np.complex128) 
    
    for b in range(N+1):
        
        t_i_val = t_1 * b
        
        integrand_vals = integrand(k_vals_full, x_vals, t_vals, 
                                   EI, rho, A, x_o, t_i_val)
        
        for i, x_val in enumerate(x_vals):
            for j, t_val in enumerate(t_vals):
                w_results[i, j] = simpson(integrand_vals[:, i, j], k_vals_full)  # Integración en k
    
    w_results *= prefactor
    
    return w_results

EI = 1.0
rho = 1.0
A = 1.0
x_o = 9.0
t_1 = 3.0
N = 0  
    
x_vals = np.linspace(0, 10, 200)  
t_vals = np.linspace(0, 10, 200)  
    
w_results = w_function(x_vals, t_vals, EI, rho, A, x_o, N, t_1)
    
#x_eval_index = np.abs(x_vals - 9.0).argmin()
#w_at_x_eval = w_results[x_eval_index, :]

# Graficar el resultado
plt.figure(figsize=(10, 6))
plt.contourf(x_vals, t_vals, w_results.real.T, levels=50, cmap="viridis")
plt.colorbar(label="w(x, t) (real)")
plt.title("Solución w(x, t)")
plt.xlabel("x")
plt.ylabel("t")
plt.grid()
plt.show()

