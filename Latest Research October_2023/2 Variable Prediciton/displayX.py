import numpy as np
np.set_printoptions(linewidth=200, threshold=np.inf)

# Define the function
def f(x1, x2):
    return 3 * np.cos(2 * np.pi * (x1**2 - x2**2))

# Set the parameters
lower_bound = -1
upper_bound = 1
n_samples = 1000

# Generate the data
x1_values = np.linspace(lower_bound, upper_bound, n_samples).reshape(n_samples, 1)
x2_values = np.linspace(lower_bound, upper_bound, n_samples).reshape(n_samples, 1)

# Get a meshgrid for x1 and x2 values
X1, X2 = np.meshgrid(x1_values, x2_values)

# Calculate y values using the function
y_values = f(X1, X2)

# Reshape the data for training
X = np.column_stack((X1.ravel(), X2.ravel()))
y = y_values.ravel()

print(X)
