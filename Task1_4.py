import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Define the model and loss function
def model(x, theta_0, theta_1):
    return theta_0 + theta_1 * x

def mse_loss(y_pred, y):
    return F.mse_loss(y_pred, y)

def compute_optimal_theta(x, y):
    n = len(x)
    x_sum = torch.sum(x)
    y_sum = torch.sum(y)
    xy_sum = torch.sum(x * y)
    x2_sum = torch.sum(x ** 2)
    
    # Compute theta_1 and theta_0 using the normal equations
    theta_1 = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum ** 2)
    theta_0 = (y_sum - theta_1 * x_sum) / n
    
    return theta_0.item(), theta_1.item()

def compute_min_loss(x, y, theta_0_opt, theta_1_opt):
    y_pred_opt = theta_1_opt * x + theta_0_opt
    return mse_loss(y_pred_opt, y).item()

# Gradient Descent with Momentum Implementations
def gradient_descent(x, y, min_loss, method='full-batch', lr=0.01, batch_size=5, max_epochs=10000, epsilon=0.001, momentum=0.9):
    # Initialize parameters
    theta_0 = torch.tensor(0.0, requires_grad=True)
    theta_1 = torch.tensor(0.0, requires_grad=True)
    
    # Initialize momentum terms
    v_theta_0 = torch.zeros_like(theta_0)
    v_theta_1 = torch.zeros_like(theta_1)
    
    losses = []
    iterations = []
    theta_0_vals = []
    theta_1_vals = []
    n = len(x)
    total_iterations = 0  # Track total iterations
    e = 0
    
    for epoch in range(max_epochs):
        if method == 'full-batch':
            # Full-batch gradient descent
            total_iterations += 1  # One iteration per epoch
            theta_0.grad = None
            theta_1.grad = None
            y_pred = model(x, theta_0, theta_1)
            loss = mse_loss(y_pred, y)
            loss.backward()  # Compute gradients
            
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
                total_iterations += 1  # One iteration per sample
                x_single, y_single = x[i:i+1], y[i:i+1]
                theta_0.grad = None
                theta_1.grad = None
                y_pred = model(x_single, theta_0, theta_1)
                loss = mse_loss(y_pred, y_single)
                loss.backward()  # Compute gradients
                
                # Update parameters for each individual sample with momentum
                with torch.no_grad():
                    v_theta_0 = momentum * v_theta_0 + lr * theta_0.grad
                    v_theta_1 = momentum * v_theta_1 + lr * theta_1.grad
                    theta_0 -= v_theta_0
                    theta_1 -= v_theta_1

        # Track losses and parameter values after each epoch
        e += 1
        current_loss = loss.item()
        losses.append(current_loss)
        iterations.append(epoch)
        theta_0_vals.append(theta_0.item())
        theta_1_vals.append(theta_1.item())

        # Early stopping: Check if the current loss is within epsilon neighborhood of the minimum loss
        if abs(current_loss - min_loss) < epsilon:
            break

    return theta_0_vals, theta_1_vals, losses, iterations, total_iterations, e

# Generate synthetic data
np.random.seed(45)
num_samples = 40

# Generate data
x1 = np.random.uniform(-1, 1, num_samples)
f_x = 3 * x1 + 4
eps = np.random.randn(num_samples)
y_np = f_x + eps

# Convert data to tensors
x = torch.tensor(x1, dtype=torch.float32).view(-1, 1)
y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

# Parameters
learning_rate = 0.01
momentum = 0.9
max_epochs = 10000
batch_size = 10
epsilon = 0.001

# Compute the optimal theta values using normal equations
theta_0_opt, theta_1_opt = compute_optimal_theta(x, y)

# Compute the minimum loss based on optimal thetas
min_loss = compute_min_loss(x, y, theta_0_opt, theta_1_opt)

# Run gradient descent methods
theta_0_vals_full, theta_1_vals_full, losses_full, iterations_full, total_iterations_full, e_full = gradient_descent(x, y, min_loss, method='full-batch', lr=learning_rate, momentum=momentum, max_epochs=max_epochs, epsilon=epsilon)
theta_0_vals_sgd, theta_1_vals_sgd, losses_sgd, iterations_sgd, total_iterations_sgd, e_sgd = gradient_descent(x, y, min_loss, method='stochastic', lr=learning_rate, momentum=momentum, max_epochs=max_epochs, epsilon=epsilon)

print(f"Steps with Full-Batch Gradient Descent with Momentum: {total_iterations_full / len(iterations_full)}",f" Number of epochs:{len(iterations_full)}")
print(f"Steps with Stochastic Gradient Descent with Momentum: {total_iterations_sgd / len(iterations_sgd)}",f" Number of epochs:{len(iterations_sgd)}")

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
# Gradient Descent without Momentum
def gradient_descent_vanilla(x, y, min_loss, method='full-batch', lr=0.01, batch_size=5, max_epochs=10000, epsilon=0.001):
    # Initialize parameters
    theta_0 = torch.tensor(0.0, requires_grad=True)
    theta_1 = torch.tensor(0.0, requires_grad=True)
    
    losses = []
    iterations = []
    theta_0_vals = []
    theta_1_vals = []
    n = len(x)
    total_iterations = 0
    e = 0
    
    for epoch in range(max_epochs):
        if method == 'full-batch':
            # Full-batch gradient descent
            total_iterations += 1
            theta_0.grad = None
            theta_1.grad = None
            y_pred = model(x, theta_0, theta_1)
            loss = mse_loss(y_pred, y)
            loss.backward()
            
            with torch.no_grad():
                theta_0 -= lr * theta_0.grad
                theta_1 -= lr * theta_1.grad
        elif method == 'stochastic':
            # Stochastic gradient descent
            indices = np.random.permutation(n)
            for i in indices:
                total_iterations += 1
                x_single, y_single = x[i:i+1], y[i:i+1]
                theta_0.grad = None
                theta_1.grad = None
                y_pred = model(x_single, theta_0, theta_1)
                loss = mse_loss(y_pred, y_single)
                loss.backward()

                with torch.no_grad():
                    theta_0 -= lr * theta_0.grad
                    theta_1 -= lr * theta_1.grad

        # Track losses and parameters
        e += 1
        current_loss = loss.item()
        losses.append(current_loss)
        iterations.append(epoch)
        theta_0_vals.append(theta_0.item())
        theta_1_vals.append(theta_1.item())

        if abs(current_loss - min_loss) < epsilon:
            break

    return theta_0_vals, theta_1_vals, losses, iterations, total_iterations, e
# Run vanilla gradient descent
theta_0_vals_vanilla_full, theta_1_vals_vanilla_full, losses_vanilla_full, iterations_vanilla_full, total_iterations_vanilla_full, e_vanilla_full = gradient_descent_vanilla(x, y, min_loss, method='full-batch', lr=learning_rate, max_epochs=max_epochs, epsilon=epsilon)
theta_0_vals_vanilla_sgd, theta_1_vals_vanilla_sgd, losses_vanilla_sgd, iterations_vanilla_sgd, total_iterations_vanilla_sgd, e_vanilla_sgd = gradient_descent_vanilla(x, y, min_loss, method='stochastic', lr=learning_rate, max_epochs=max_epochs, epsilon=epsilon)

# Print results
print(f"Vanilla Full-Batch GD Steps: {total_iterations_vanilla_full / len(iterations_vanilla_full)}",f" Number of epochs:{len(iterations_vanilla_full)}")
print(f"Vanilla Stochastic GD Steps: {total_iterations_vanilla_sgd / len(iterations_vanilla_sgd)}",f" Number of epochs:{len(iterations_vanilla_sgd)}")

# Plot contours for every epoch
plot_contours(x, y, theta_0_vals_full, theta_1_vals_full, 'Full-Batch with Momentum', range(15))
plot_contours(x, y, theta_0_vals_sgd, theta_1_vals_sgd, 'Stochastic with Momentum', range(15))

# Plot results for full-batch and stochastic gradient descent
plot_results(iterations_full, losses_full, 'Full-Batch Gradient Descent with Momentum')
plot_results(iterations_sgd, losses_sgd, 'Stochastic Gradient Descent with Momentum')