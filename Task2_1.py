import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import os
from einops import rearrange
file_path = os.path.join(os.path.dirname(__file__), 'cat.jpeg') 
# Remove all the warnings
import warnings
warnings.filterwarnings('ignore')

# Set env CUDA_LAUNCH_BLOCKING=1
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Read in an image
img = torchvision.io.read_image(file_path)
print(img.shape)
plt.imshow(rearrange(img, 'c h w -> h w c').numpy())
plt.title("Original Image")
plt.axis("off")
plt.show()
def create_coordinate_map(img):
    """
    Create a coordinate map from the image tensor.
    img: torch.Tensor of shape (num_channels, height, width)
    return: tuple of torch.Tensor of shape (height * width, 2) and torch.Tensor of shape (height * width, num_channels)
    """
    num_channels, height, width = img.shape

    # Create a 2D grid of (x,y) coordinates
    w_coords = torch.arange(width).repeat(height, 1)
    h_coords = torch.arange(height).repeat(width, 1).t()
    w_coords = w_coords.reshape(-1)
    h_coords = h_coords.reshape(-1)

    # Combine the x and y coordinates into a single tensor
    X = torch.stack([h_coords, w_coords], dim=1).float()

    # Move X to GPU if available
    X = X.to(device)

    # Reshape the image to (h * w, num_channels)
    Y = rearrange(img, 'c h w -> (h w) c').float()
    return X, Y

# Create coordinate map
dog_X, dog_Y = create_coordinate_map(img)
from sklearn.kernel_approximation import RBFSampler

def apply_rff(X, gamma=1, n_components=100):
    rff = RBFSampler(gamma=gamma, n_components=n_components)
    X_rff = rff.fit_transform(X.cpu())
    return torch.tensor(X_rff, dtype=torch.float32).to(device)

# Apply RFF
dog_X_rff = apply_rff(dog_X)

class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.linear(x)
    
net = LinearModel(dog_X_rff.shape[1], 3)
net.to(device)

def train(net, lr, X, Y, epochs, verbose=True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch} loss: {loss.item():.6f}")
    return loss.item()

# Train the model
train(net, 0.01, dog_X_rff, dog_Y, 1000)
def poly_features(X, degree):
    """
    Apply polynomial features transformation to the input tensor.
    X: torch.Tensor of shape (num_samples, 2)
    degree: int
    return: torch.Tensor of shape (num_samples, degree * (degree + 1) / 2)
    """
    X1 = X[:, 0].unsqueeze(1)
    X2 = X[:, 1].unsqueeze(1)
    X = torch.cat([X1, X2], dim=1)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X.cpu())
    return torch.tensor(X_poly, dtype=torch.float32).to(device)

# Apply polynomial features
dog_X_poly = poly_features(dog_X, 2)

# Redefine and train the model
net = LinearModel(dog_X_poly.shape[1], 3)
net.to(device)
train(net, 0.005, dog_X_poly, dog_Y, 1500)
def calculate_rmse(original, reconstructed):
    return torch.sqrt(F.mse_loss(original, reconstructed))

def calculate_psnr(original, reconstructed):
    mse = calculate_rmse(original, reconstructed).item()
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def plot_reconstructed_and_original_image(original_img, net, X, title=""):
    """
    Plot the original and reconstructed images.
    """
    num_channels, height, width = original_img.shape
    net.eval()
    with torch.no_grad():
        outputs = net(X)
        outputs = outputs.reshape(height, width, num_channels)
        outputs = rearrange(outputs, 'h w c -> c h w')
        
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.imshow(rearrange(outputs, 'c h w -> h w c').cpu().numpy())
    ax0.set_title("Reconstructed Image")

    ax1.imshow(rearrange(original_img, 'c h w -> h w c').cpu().numpy())
    ax1.set_title("Original Image")

    for a in [ax0, ax1]:
        a.axis("off")

    fig.suptitle(title, y=0.9)
    plt.tight_layout()
    plt.show()

# Plot results
plot_reconstructed_and_original_image(img, net, dog_X_poly, title="Reconstructed Image with Polynomial Features")

# Calculate metrics
original_img = rearrange(img, 'c h w -> (h w) c').float() / 255.0
reconstructed_img = net(dog_X_poly).reshape(num_channels, height, width).float() / 255.0
rmse = calculate_rmse(original_img, reconstructed_img)
psnr = calculate_psnr(original_img, reconstructed_img)

print(f'Root Mean Squared Error (RMSE): {rmse.item()}')
print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr} dB')
