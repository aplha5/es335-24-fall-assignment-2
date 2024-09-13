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
theta_0 = torch.tensor(4.0, requires_grad=True)
theta_1 = torch.tensor(3.0, requires_grad=True)

# linear regression model
def model(x):
    return theta_1 * x + theta_0

# loss function
def mse_loss(predicted, target):
    return torch.mean((predicted - target) ** 2)

# computing predictions and loss
y_pred = model(x1)
loss = mse_loss(y_pred, y)

# Backward pass: compute gradients
loss.backward()
print(f"Gradient with respect to θ0 (theta_0): {theta_0.grad}")
print(f"Gradient with respect to θ1 (theta_1): {theta_1.grad}")
