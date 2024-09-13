import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Define the model and loss function
def model(x, theta_0, theta_1):
    return theta_0 + theta_1 * x

def mse_loss(y_pred, y):
    return F.mse_loss(y_pred, y)

# Gradient Descent and Momentum Implementations
def gradient_descent(x, y, method='full-batch', lr=0.01, batch_size=5, max_epochs=15, epsilon=0.001, momentum=0.9):
    theta_0 = torch.tensor(0.0, requires_grad=True)
    theta_1 = torch.tensor(0.0, requires_grad=True)
    
    losses = []
    iterations = []
    theta_0_vals = []
    theta_1_vals = []
    n = len(x)
    total_iterations = 0
    e = 0

    # Initialize momentum terms
    v_theta_0 = torch.zeros_like(theta_0)
    v_theta_1 = torch.zeros_like(theta_1)

    for epoch in range(max_epochs):
        if method == 'full-batch':
            # Full-batch gradient descent
            total_iterations += 1
            theta_0.grad = None
            theta_1.grad = None
            y_pred = model(x, theta_0, theta_1)
            loss = mse_loss(y_pred, y)
            loss.backward()
            
            # Update parameters with momentum
            with torch.no_grad():
                v_theta_0 = momentum * v_theta_0 + lr * theta_0.grad
                v_theta_1 = momentum * v_theta_1 + lr * theta_1.grad
                theta_0 -= v_theta_0
                theta_1 -= v_theta_1
        elif method == 'stochastic':
            # Stochastic gradient descent (SGD)
            indices = np.random.permutation(n)
            for i in indices:
                total_iterations += 1
                x_single, y_single = x[i:i+1], y[i:i+1]
                theta_0.grad = None
                theta_1.grad = None
                y_pred = model(x_single, theta_0, theta_1)
                loss = mse_loss(y_pred, y_single)
                loss.backward()
                
                # Update parameters with momentum
                with torch.no_grad():
                    v_theta_0 = momentum * v_theta_0 + lr * theta_0.grad
                    v_theta_1 = momentum * v_theta_1 + lr * theta_1.grad
                    theta_0 -= v_theta_0
                    theta_1 -= v_theta_1

        e += 1
        losses.append(loss.item())
        iterations.append(epoch)
        theta_0_vals.append(theta_0.item())
        theta_1_vals.append(theta_1.item())

        if loss.item() < epsilon:
            break

    return theta_0_vals, theta_1_vals, losses, iterations, total_iterations, e

# Generate synthetic data with new dataset
np.random.seed(45)
num_samples = 40

# Generate data
x1 = np.random.uniform(-1, 1, num_samples)
f_x = 3*x1 + 4
eps = np.random.randn(num_samples)
y = f_x + eps

# Convert data to tensors
x = torch.tensor(x1, dtype=torch.float32).view(-1, 1)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Parameters
learning_rate = 0.01
momentum = 0.9
max_epochs = 15
batch_size = 10
epsilon = 0.001

# Run gradient descent with different methods
theta_0_vals_full, theta_1_vals_full, losses_full, iterations_full, total_iterations_full, e_full = gradient_descent(x, y, method='full-batch', lr=learning_rate, momentum=momentum, max_epochs=max_epochs, epsilon=epsilon)
theta_0_vals_sgd, theta_1_vals_sgd, losses_sgd, iterations_sgd, total_iterations_sgd, e_sgd = gradient_descent(x, y, method='stochastic', lr=learning_rate, momentum=momentum, max_epochs=max_epochs, epsilon=epsilon)
print(f"Steps with Full-Batch Gradient Descent with Momentum: {total_iterations_full / len(iterations_full)}")
print(f"Steps with Stochastic Gradient Descent with Momentum: {total_iterations_sgd / len(iterations_sgd)}")
# Function to plot contour plots
def plot_contours(x, y, theta_0_vals, theta_1_vals, method_name, epochs_to_plot):
    theta_0_range = np.linspace(-10, 10, 100)
    theta_1_range = np.linspace(-10, 10, 100)
    theta_0_grid, theta_1_grid = np.meshgrid(theta_0_range, theta_1_range)
    
    loss_grid = np.zeros(theta_0_grid.shape)

    for i in range(theta_0_grid.shape[0]):
        for j in range(theta_0_grid.shape[1]):
            theta_0_val = torch.tensor(theta_0_grid[i, j], dtype=torch.float32)
            theta_1_val = torch.tensor(theta_1_grid[i, j], dtype=torch.float32)
            y_pred = model(x, theta_0_val, theta_1_val)
            loss_grid[i, j] = mse_loss(y_pred, y).item()

    for epoch in epochs_to_plot:
        if epoch < len(theta_0_vals):
            plt.figure(figsize=(12, 6))
            plt.contourf(theta_0_range, theta_1_range, loss_grid, levels=50, cmap='viridis')
            plt.colorbar()
            plt.scatter(theta_0_vals[epoch], theta_1_vals[epoch], color='red', label=f'Epoch {epoch}')
            plt.xlabel('Theta 0')
            plt.ylabel('Theta 1')
            plt.title(f'Contour Plot of Loss Function ({method_name})')
            plt.legend()
            plt.grid()
            plt.show()
def plot_results(iterations, theta_vals, method_name):
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, theta_vals, label=f'{method_name}')
    plt.xlabel('Iterations')
    plt.ylabel('Theta Values')
    plt.title(f'{method_name} - Theta Values over Iterations')
    plt.legend()
    plt.grid()
    plt.show()

# Plot contours for every epoch
plot_contours(x, y, theta_0_vals_full, theta_1_vals_full, 'Full-Batch with Momentum', range(len(theta_0_vals_full)))
plot_contours(x, y, theta_0_vals_sgd, theta_1_vals_sgd, 'Stochastic with Momentum', range(len(theta_0_vals_sgd)))

# Vanilla Gradient Descent for comparison
def vanilla_gradient_descent(x, y, lr=0.01, batch_size=5, max_epochs=10000, epsilon=0.001):
    theta_0 = torch.tensor(0.0, requires_grad=True)
    theta_1 = torch.tensor(0.0, requires_grad=True)
    
    losses = []
    iterations = []
    theta_0_vals = []
    theta_1_vals = []
    n = len(x)
    total_iterations = 0
    e=0

    for epoch in range(max_epochs):
        if batch_size == n:
            # Full-batch gradient descent
            total_iterations += 1
            theta_0.grad = None
            theta_1.grad = None
            y_pred = model(x, theta_0, theta_1)
            loss = mse_loss(y_pred, y)
            loss.backward()
            
            # Update parameters
            with torch.no_grad():
                theta_0 -= lr * theta_0.grad
                theta_1 -= lr * theta_1.grad

        else:
            # Mini-batch gradient descent
            indices = np.random.permutation(n)
            for i in range(0, n, batch_size):
                total_iterations += 1
                idx = indices[i:i+batch_size]
                x_batch, y_batch = x[idx], y[idx]
                theta_0.grad = None
                theta_1.grad = None
                y_pred = model(x_batch, theta_0, theta_1)
                loss = mse_loss(y_pred, y_batch)
                loss.backward()
                
                # Update parameters
                with torch.no_grad():
                    theta_0 -= lr * theta_0.grad
                    theta_1 -= lr * theta_1.grad

        e += 1
        losses.append(loss.item())
        iterations.append(epoch)
        theta_0_vals.append(theta_0.item())
        theta_1_vals.append(theta_1.item())

        if loss.item() < epsilon:
            break

    return theta_0_vals, theta_1_vals, losses, iterations, total_iterations, e

# Run vanilla gradient descent
theta_0_vals_vanilla, theta_1_vals_vanilla, losses_vanilla, iterations_vanilla, total_iterations_vanilla, e_vanilla = vanilla_gradient_descent(x, y, lr=learning_rate, batch_size=batch_size, max_epochs=max_epochs, epsilon=epsilon)

# Plot results for vanilla gradient descent
plot_results(iterations_vanilla, theta_0_vals_vanilla, 'Vanilla Gradient Descent')
