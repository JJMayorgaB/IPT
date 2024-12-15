import numpy as np
from scipy.optimize import root_scalar

num_intervals = 20  
L=10
f1 = lambda x: np.cos(x) * np.cosh(x) + 1

intervals = [(i, i+2) for i in range(0, 2*num_intervals, 2)]

def find_roots(equation, intervals):
    roots = []
    for interval in intervals:
        try:
            sol = root_scalar(equation, bracket=interval, method='brentq')
            if sol.converged:
                root = sol.root
                # Evitar duplicados debido a la periodicidad de cos(x)
                if not any(np.isclose(root, r, atol=1e-6) for r in roots):
                    roots.append(root)
        except ValueError:
            continue
    return np.array(roots)  

beta_values = find_roots(f1, intervals) / L


print(f"Intervalos generados: {intervals}")
print(len(intervals))
print(beta_values)
print(len(beta_values))