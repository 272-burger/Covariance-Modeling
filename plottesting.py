import numpy as np
import matplotlib.pyplot as plt

# Parameters for the normal distribution
mean = [0, 0]  # Mean of the distribution
cov = [[1, 0], [0, 1]]  # Covariance matrix of the distribution

# Generate grid of x and y values
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate the probability density function (PDF) values for each point in the grid
Z = np.exp(-0.5 * np.einsum('ijk,kl,ijl->ij', pos - mean, np.linalg.inv(cov), pos - mean)) \
    / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

# Plot the 2D normal distribution
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Normal Distribution')
plt.show()



