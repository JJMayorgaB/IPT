import numpy as np
from scipy.integrate import simpson  
import matplotlib.pyplot as plt 

def integrand(k_vals, x, t, EI, rho, A, Delta_t, x_o, r_o, t_i):

    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.where(k_vals != 0, np.sin((EI / (rho * A)) * k_vals**2 * (t - t_i)) / k_vals**2, (EI / (rho * A)) * (t - t_i))
        term2 = np.exp(-0.5 * ((Delta_t * EI / (rho * A))**2) * k_vals**4)
        term3 = np.exp(-2 * (k_vals / r_o)**2)
        term4 = np.exp(1j * k_vals * (x - x_o))
    
    return term1 * term2 * term3 * term4


def w_function(x_vals, t_vals, EI, rho, A, Delta_t, x_o, r_o, t_0, N):

    k_vals_positive = np.logspace(-2, 1, 500)  # Escala logarítmica
    k_vals_negative = -k_vals_positive
    k_vals_full = np.concatenate((k_vals_negative, k_vals_positive))
    
    prefactor = (-2 * np.pi**2 * Delta_t) / EI
    
    x_mesh, t_mesh = np.meshgrid(x_vals, t_vals, indexing='ij')
    
    w_results = np.zeros_like(x_mesh, dtype=complex)
    
    for b in range(N + 1):
        
        t_i_val = t_0 * b
        
        # Broadcast k_vals to match x_mesh and t_mesh shape
        k_vals_broadcast = k_vals_full[:, np.newaxis, np.newaxis]
        
        integrand_vals = integrand(k_vals_broadcast, x_mesh, t_mesh, 
                                   EI, rho, A, Delta_t, x_o, r_o, t_i_val)
        
        # Integrate along the k_vals axis (axis=0)
        integral_result = simpson(y=integrand_vals, x=k_vals_full, axis=0)
        w_results += integral_result
    
    w_results *= prefactor
    
    return w_results



EI = 1.0
rho = 1.0
A = 1.0
r_o = 0.1
Delta_t = 1.0
x_o = 9.0
t_1 = 2.0
N = 10  
    
x_vals = np.linspace(0, 10, 200)  
t_vals = np.linspace(0, 50, 1000)  
    
w_results = w_function(x_vals, t_vals, EI, rho, A, Delta_t, x_o, r_o, t_1, N)
    
x_eval_index = np.abs(x_vals - 9.0).argmin()
w_at_x_eval = w_results[x_eval_index, :]

#grafica de w en (x,t), comportamiento completo
plt.figure(figsize=(12, 7))
plt.contourf(t_vals, x_vals, np.abs(w_results), levels=20, cmap='viridis')
plt.colorbar(label='Magnitud de w')
plt.xlabel('Tiempo')
plt.ylabel('Posición x')
plt.title('Distribución de w')
plt.tight_layout()
plt.show()
    
#grafica de w en un punto x especifico x util para analizar por ejemplo el extremo de la placa, tensiones, etc.
plt.figure(figsize=(10, 6))
plt.plot(t_vals, np.abs(w_at_x_eval))
plt.xlabel('Tiempo')
plt.ylabel('Magnitud de w en x = 9')
plt.title('Evolución temporal de w en x = 9')
plt.tight_layout()
plt.show()
