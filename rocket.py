import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#ipt flying rocket

juanjose = 1

# Parámetros 
a_b = 25472.3537562965e-6 #+ 136.073682289469e-6 # Área de la sección transversal de la botella
a_e = 59.1737825859559e-6 #- 0.192821119516073e-6 # Área de la sección transversal de la boquilla de la botella
c_d = 0.75 # Coeficiente de cohete a escala
g = 9.78 # Valor de la gravedad en Bogotá
m_b = 89.9e-3 #+ 0.1e-3 # Masa del cohete sin agua

p_aire = 172368.93233 #- 6894.7572931783 # Presión interna (25psi) está en pascales
v_aire = 100e-6 #+ 0.0001 # Volumen ocupado por el aire en la botella.
R = 8.31446261815324 # Constante de los gases ideales
T = 291.15  # Temperatura ambiente
n = p_aire*v_aire/(R*T) # Cantidad de moles de aire en la botella
m_m_a = 28.9656e-3 # Masa molar del aire
m_aire = n*m_m_a # Masa del aire


v_agua = 1500e-6 - v_aire #- 0.0001 # Volumen de agua en la botella
rho_agua = 997 # Densidad del agua en kg/m^3
m_agua = v_agua*rho_agua # Masa de agua en la botella

p_agua = p_aire # Presión interna del agua
p_atm = 73000 # Presión atmosférica de Bogotá
h = 14e-2 # Altura desde la superficie del agua en el tanque hasta la boquilla
a_b_0 = v_agua/h

rho_aire = m_m_a*p_atm/(R*T)

pasos = 1000

t_agua = np.linspace(0, 0.4847, pasos)  # tiempo en segundos, 1000 pasos
num_pasos = len(t_agua)

p_agua_arr = np.zeros(num_pasos)
v_aire_arr = np.zeros(num_pasos)
v_agua_arr = np.zeros(num_pasos)
m_agua_arr = np.zeros(num_pasos)
u_agua_arr = np.zeros(num_pasos)
beta_arr = np.zeros(num_pasos)
h_arr = np.zeros(num_pasos)

p_agua_t = p_aire
h_t = h

for i in range(num_pasos):
    # Actualización de variables
    u_agua_t = (1/np.sqrt(1-(a_e/a_b)**2))*np.sqrt((2*(p_agua_t-p_atm))/(rho_agua)+2*g*h)
    beta = rho_agua*a_e*u_agua_t
    m_agua_t = m_agua - beta*t_agua[i]
    v_agua_t = m_agua_t / rho_agua
    v_aire_t = v_aire + v_agua - v_agua_t
    p_agua_t = p_agua * (v_aire / v_aire_t)**1.4
    h_t = v_agua_t/a_b_0
    
    
    p_agua_arr[i]=p_agua_t
    v_aire_arr[i]=v_aire_t
    v_agua_arr[i]=v_agua_t
    m_agua_arr[i]=m_agua_t   
    u_agua_arr[i]=u_agua_t
    beta_arr[i]=beta
    h_arr[i] = h_t
    
###### Promedios Agua #####
p_agua_prom = np.mean(p_agua_arr)
v_aire_prom = np.mean(v_aire_arr)
v_agua_prom = np.mean(v_agua_arr)
m_agua_prom = np.mean(m_agua_arr)
u_agua_prom = np.mean(u_agua_arr)
beta_prom = np.mean(beta_arr)
print(p_agua_prom)
print(v_aire_prom)
print(v_agua_prom)
print(m_agua_prom)
print(u_agua_prom)
print(beta_prom)

p_agua_func = interp1d(t_agua, p_agua_arr, kind='cubic', fill_value="extrapolate")
u_agua_func = interp1d(t_agua, u_agua_arr, kind='cubic', fill_value="extrapolate")
m_agua_func = interp1d(t_agua, m_agua_arr, kind='cubic', fill_value="extrapolate")
beta_func = interp1d(t_agua, beta_arr, kind='cubic', fill_value="extrapolate")
h_func = interp1d(t_agua, h_arr, kind='cubic', fill_value="extrapolate")

def solve_agua(t, y_agua):
    y, y_punto, x, x_punto, theta = y_agua
    p_agua_f = p_agua_func(t)
    m_agua_f = m_agua_func(t)
    h_f = h_func(t)
    dy_dt = y_punto
    dx_dt = x_punto
    dy_dos_puntos_dt = 2*a_e/(1-(a_e/a_b_0)**2)*(p_agua_f-p_atm + rho_agua*g*h_f)*np.sin(theta)/(m_b + m_aire + m_agua_f)-(rho_aire*(y_punto**2)*c_d*a_b)/(2*(m_b + m_aire + m_agua_f))*np.sin(theta)- g
    dx_dos_puntos_dt = 2*a_e/(1-(a_e/a_b_0)**2)*(p_agua_f-p_atm + rho_agua*g*h_f)*np.cos(theta)/(m_b + m_aire + m_agua_f)-(rho_aire*(y_punto**2)*c_d*a_b)/(2*(m_b + m_aire + m_agua_f))*np.cos(theta)
    theta = np.arctan(y_punto/x_punto)
    return [dy_dt, dy_dos_puntos_dt, dx_dt, dx_dos_puntos_dt, theta]


sol = solve_ivp(solve_agua, [0,0.4847] , [0.0,1e-10,0,1e-10,84.5*np.pi/180], t_eval=np.linspace(0, 0.4847,pasos))

plt.figure(figsize=(8, 5)) 

plt.plot(sol.t, sol.y[0],  linewidth=2, label = "altitude") 

plt.title("Rocket Motion", fontsize=18) 
plt.xlabel("tiempo", fontsize=14)  
plt.ylabel("m", fontsize=14)  
plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")  
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")  
plt.legend()

plt.tight_layout() 
plt.show()