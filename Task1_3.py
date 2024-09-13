import torch
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(45)
num_samples = 40

# Generate data
x1 = np.random.uniform(-1, 1, num_samples)
f_x = 3 * x1 + 4
eps = np.random.randn(num_samples)
y = f_x + eps

# Convert data to torch tensors
x1 = torch.tensor(x1, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define the linear regression model
def model(x, theta_0, theta_1):
    return theta_1 * x + theta_0

# Define the Mean Squared Error loss function
def mse_loss(predicted, target):
    return torch.mean((predicted - target) ** 2)

# Gradient Descent Implementation with early stopping for epsilon neighborhood
def gradient_descent(x, y, method='full-batch', lr=0.01, batch_size=5, max_epochs=10000, epsilon=0.001):
    # Initialize parameters
    theta_0 = torch.tensor(0.0, requires_grad=True)
    theta_1 = torch.tensor(0.0, requires_grad=True)
    
    losses = []
    iterations = []
    theta_0_vals = []
    theta_1_vals = []
    n = len(x)
    total_iterations = 0  # Track total iterations
    e=0
    
    for epoch in range(max_epochs):
        if method == 'full-batch':
            # Full-batch gradient descent
            total_iterations += 1  # One iteration per epoch
            theta_0.grad = None
            theta_1.grad = None
            y_pred = model(x, theta_0, theta_1)
            loss = mse_loss(y_pred, y)
            loss.backward()  # Compute gradients
            
            # Update parameters
            with torch.no_grad():
                theta_0 -= lr * theta_0.grad
                theta_1 -= lr * theta_1.grad

        elif method == 'mini-batch':
            # Mini-batch gradient descent
            indices = np.random.permutation(n)
            for i in range(0, n, batch_size):
                total_iterations += 1  # One iteration per mini-batch
                idx = indices[i:i+batch_size]
                x_batch, y_batch = x[idx], y[idx]
                theta_0.grad = None
                theta_1.grad = None
                y_pred = model(x_batch, theta_0, theta_1)
                loss = mse_loss(y_pred, y_batch)
                loss.backward()  # Compute gradients
                
                # Update parameters for each mini-batch
                with torch.no_grad():
                    theta_0 -= lr * theta_0.grad
                    theta_1 -= lr * theta_1.grad

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
                
                # Update parameters for each individual sample
                with torch.no_grad():
                    theta_0 -= lr * theta_0.grad
                    theta_1 -= lr * theta_1.grad

        # Track losses and parameter values after each epoch
        e=e+1
        losses.append(loss.item())
        iterations.append(epoch)
        theta_0_vals.append(theta_0.item())
        theta_1_vals.append(theta_1.item())

        # Early stopping: Check if the loss is within epsilon neighborhood
        if loss.item() < epsilon:
            break

    return theta_0_vals, theta_1_vals, losses, iterations, total_iterations,e  # Return total iterations


# Visualization function for loss convergence
def plot_loss_convergence(full_batch_losses, mini_batch_losses, sgd_losses):
    plt.plot(full_batch_losses, label='Full-batch')
    plt.plot(mini_batch_losses, label='Mini-batch')
    plt.plot(sgd_losses, label='SGD')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs for Different Methods')
    plt.legend()
    plt.show()

# Contour plot for visualization per epoch (up to 15 epochs for visualization)
def plot_contour_per_epoch(theta_0_vals, theta_1_vals, x, y, method):
    T0, T1 = np.meshgrid(np.linspace(-1, 7, 100), np.linspace(-1, 7, 100))
    Z = np.zeros_like(T0)

    # Convert torch tensors to NumPy arrays for contour computation
    x_np = x.detach().numpy()
    y_np = y.detach().numpy()

    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            theta_0 = T0[i, j]
            theta_1 = T1[i, j]
            y_pred = theta_1 * x_np + theta_0  # Predict using numpy
            Z[i, j] = np.mean((y_pred - y_np)**2)  # Compute MSE manually for each pair of (theta_0, theta_1)

    for epoch in range(min(15, len(theta_0_vals))):  # Plot for a maximum of 15 epochs
        plt.contour(T0, T1, Z, levels=30)
        plt.scatter(theta_0_vals[epoch], theta_1_vals[epoch], color='red', marker='x', label=f'Epoch {epoch}')
        plt.title(f'Contour Plot for {method} Method at Epoch {epoch}')
        plt.xlabel('theta_0')
        plt.ylabel('theta_1')
        plt.legend()
        plt.show()

# Hyperparameters
learning_rate = 0.01
max_epochs = 10000  # Allow up to 100 epochs
batch_size = 5
epsilon = 0.001

# Full-batch Gradient Descent
theta_0_full, theta_1_full, full_batch_losses, full_batch_iters, full_batch_total_iter,f_epochh = gradient_descent(x1, y, method='full-batch', lr=learning_rate, max_epochs=max_epochs, epsilon=epsilon)

# Mini-batch Gradient Descent
theta_0_mini, theta_1_mini, mini_batch_losses, mini_batch_iters, mini_batch_total_iter,m_epochh = gradient_descent(x1, y, method='mini-batch', lr=learning_rate, batch_size=batch_size, max_epochs=max_epochs, epsilon=epsilon)

# Stochastic Gradient Descent
theta_0_sgd, theta_1_sgd, sgd_losses, sgd_iters, sgd_total_iter,s_epochh = gradient_descent(x1, y, method='stochastic', lr=learning_rate, max_epochs=max_epochs, epsilon=epsilon)
#print(m_epochh)
# Plot loss convergence
plot_loss_convergence(full_batch_losses, mini_batch_losses, sgd_losses)

# Plot contour plots for each method across first 15 epochs for visualization
plot_contour_per_epoch(theta_0_full, theta_1_full, x1, y, method='Full-batch')
plot_contour_per_epoch(theta_0_mini, theta_1_mini, x1, y, method='Mini-batch')
plot_contour_per_epoch(theta_0_sgd, theta_1_sgd, x1, y, method='SGD')

# Calculate and print average iterations
print(f"Average iterations for full-batch gradient descent: {full_batch_total_iter / f_epochh:.2f}")
print(f"Average iterations for mini-batch gradient descent: {mini_batch_total_iter / m_epochh:.2f}")
print(f"Average iterations for stochastic gradient descent: {sgd_total_iter / s_epochh:.2f}")
