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

# Compute the optimal theta_0 and theta_1 using normal equations
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

# Compute the minimum loss value for the optimal thetas
def compute_min_loss(x, y, theta_0_opt, theta_1_opt):
    y_pred_opt = theta_1_opt * x + theta_0_opt
    return mse_loss(y_pred_opt, y).item()

# Gradient Descent Implementation with epsilon neighborhood from minimum loss
def gradient_descent(x, y, min_loss, method='full-batch', lr=0.01, batch_size=5, max_epochs=10000, epsilon=0.001):
    # Initialize parameters
    theta_0 = torch.tensor(0.0, requires_grad=True)
    theta_1 = torch.tensor(0.0, requires_grad=True)
    e=0
    losses = []
    iterations = []
    theta_0_vals = []
    theta_1_vals = []
    n = len(x)
    total_iterations = 0  # Track total iterations

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
        e+=1
        current_loss = loss.item()
        losses.append(current_loss)
        iterations.append(epoch)
        theta_0_vals.append(theta_0.item())
        theta_1_vals.append(theta_1.item())

        # Early stopping: Check if the current loss is within epsilon neighborhood of the minimum loss
        if abs(current_loss - min_loss) < epsilon:
            break

    # Ensure we have 15 entries by padding with the last values if necessary
    while len(theta_0_vals) < 15:
        for i in range(0, n, batch_size):
                #total_iterations += 1  # One iteration per mini-batch
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
        theta_0_vals.append(theta_0.item())
        theta_1_vals.append(theta_1.item())
    
    return theta_0_vals, theta_1_vals, losses, iterations, total_iterations,e


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
            Z[i, j] = np.mean((y_pred - y_np) ** 2)  # Compute MSE manually for each pair of (theta_0, theta_1)

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

# Compute the optimal theta values using normal equations
theta_0_opt, theta_1_opt = compute_optimal_theta(x1, y)

# Compute the minimum loss based on optimal thetas
min_loss = compute_min_loss(x1, y, theta_0_opt, theta_1_opt)

# Full-batch Gradient Descent
theta_0_full, theta_1_full, full_batch_losses, full_batch_iters, full_batch_total_iter, f_epochh = gradient_descent(
    x1, y, min_loss, method='full-batch', lr=learning_rate, max_epochs=max_epochs, epsilon=epsilon)

# Mini-batch Gradient Descent
theta_0_mini, theta_1_mini, mini_batch_losses, mini_batch_iters, mini_batch_total_iter, m_epochh = gradient_descent(
    x1, y, min_loss, method='mini-batch', lr=learning_rate, batch_size=batch_size, max_epochs=max_epochs, epsilon=epsilon)

# Stochastic Gradient Descent (SGD)
theta_0_sgd, theta_1_sgd, sgd_losses, sgd_iters, sgd_total_iter, s_epochh = gradient_descent(
    x1, y, min_loss, method='stochastic', lr=learning_rate, max_epochs=max_epochs, epsilon=epsilon)

# Print results
print("Optimal values: theta_0 =", theta_0_opt, "theta_1 =", theta_1_opt)
print("min_loss =", min_loss)
print("Average iterations required for Full-batch method to reach minimum loss are:", full_batch_total_iter / f_epochh)
print("Average iterations required for Mini-batch method to reach minimum loss are:", mini_batch_total_iter / m_epochh)
print("Average iterations required for SGD method to reach minimum loss are:", sgd_total_iter / s_epochh)

# Plot the loss convergence for each method
plot_loss_convergence(full_batch_losses, mini_batch_losses, sgd_losses)

# Plot contour for the three methods with tracked theta values
plot_contour_per_epoch(theta_0_full, theta_1_full, x1, y, method='Full-batch')
plot_contour_per_epoch(theta_0_mini, theta_1_mini, x1, y, method='Mini-batch')
plot_contour_per_epoch(theta_0_sgd, theta_1_sgd, x1, y, method='SGD')
