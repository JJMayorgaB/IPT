import numpy as np

results_array = np.load("w_values.npy")

# Reshape for easier processing
# Assumes results are sorted by time, then x, then y
w_data = results_array[:, 0]  # Displacement values
t_data = results_array[:, 1]  # Time values
x_data = results_array[:, 2]  # X coordinates
y_data = results_array[:, 3]  # Y coordinates

# Calculate temporal derivative
def calculate_temporal_derivative(w_data, t_data):
    # Central difference method
    dw_dt = np.gradient(w_data, t_data)
    return dw_dt

# Calculate spatial derivatives
def calculate_spatial_derivatives(w_data, x_data, y_data):
    # Reshape data if needed
    w_grid = w_data.reshape(len(np.unique(t_data)), 
                            len(np.unique(x_data)), 
                            len(np.unique(y_data)))
    
    # Partial derivatives using numpy gradient
    dw_dx = np.gradient(w_grid, x_data, axis=1)
    dw_dy = np.gradient(w_grid, y_data, axis=2)
    
    return dw_dx, dw_dy

# Example usage
temporal_derivative = calculate_temporal_derivative(w_data, t_data)
spatial_dx, spatial_dy = calculate_spatial_derivatives(w_data, x_data, y_data)
