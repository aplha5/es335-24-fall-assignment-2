import torch
import numpy as np

np.random.seed(45)
num_samples = 40

x1 = np.random.uniform(-1, 1, num_samples)
f_x = 3 * x1 + 4
eps = np.random.randn(num_samples)
y = f_x + eps

# Convert data to torch tensors
x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=False)
y = torch.tensor(y, dtype=torch.float32, requires_grad=False)

# Initialize parameters θ0 and θ1 (requires_grad=True for autograd to track them)
theta_0 = torch.tensor(0.0, requires_grad=True)
theta_1 = torch.tensor(0.0, requires_grad=True)

# Define the linear regression model
def model(x, theta_0, theta_1):
    return theta_1 * x + theta_0

# Define the Mean Squared Error loss for a single data point
def mse_loss(predicted, target):
    return (predicted - target) ** 2

# Store stochastic gradients for each data point
grad_theta_0_list = []
grad_theta_1_list = []

learning_rate = 0.001  # Define a learning rate

for i in range(num_samples):
    # Zero the gradients from previous steps
    theta_0.grad = None
    theta_1.grad = None

    # Get the i-th data point
    x_i = x1[i]
    y_i = y[i]

    # Forward pass: compute the prediction and loss for this data point
    y_pred_i = model(x_i,theta_0, theta_1)
    loss_i = mse_loss(y_pred_i, y_i)

    # Backward pass: compute gradients
    loss_i.backward()

    # Update parameters using gradient descent
    with torch.no_grad():
        theta_0 -= learning_rate * theta_0.grad
        theta_1 -= learning_rate * theta_1.grad

    # Store the gradients for θ0 and θ1
    grad_theta_0_list.append(theta_0.grad.item())
    grad_theta_1_list.append(theta_1.grad.item())

# Compute the average of the stochastic gradients
avg_grad_theta_0 = np.mean(grad_theta_0_list)
avg_grad_theta_1 = np.mean(grad_theta_1_list)

# Print the average stochastic gradients
print(f"Average Stochastic Gradient for θ0: {avg_grad_theta_0}")
print(f"Average Stochastic Gradient for θ1: {avg_grad_theta_1}")

# Now, compute the true gradient using the entire dataset
# Zero the gradients first
theta_0.grad = None
theta_1.grad = None
theta_0 = torch.tensor(0.0, requires_grad=True)
theta_1 = torch.tensor(0.0, requires_grad=True)
# Forward pass: compute the predictions and the total loss
y_pred = model(x1,theta_0, theta_1)
total_loss = torch.mean((y_pred - y) ** 2)

# Backward pass: compute the true gradients
total_loss.backward()

# Print the true gradients
print(f"True Gradient for θ0: {theta_0.grad.item()}")
print(f"True Gradient for θ1: {theta_1.grad.item()}")

# Compare the average stochastic gradient with the true gradient
print(f"Difference in θ0 gradient: {abs(avg_grad_theta_0 - theta_0.grad.item())}")
print(f"Difference in θ1 gradient: {abs(avg_grad_theta_1 - theta_1.grad.item())}")
