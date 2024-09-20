import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from math import log10
import os
from einops import rearrange

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load and resize the image
file_path = os.path.join(os.path.dirname(__file__), 'cat.jpeg') 
img = torchvision.io.read_image(file_path)
img = img / 255.0  # Normalize to [0,1]

# Resize the image to reduce memory usage
transform = transforms.Resize((500, 500))  # Resize to 500x500 to reduce memory
img = transform(img)
print(f"Resized Image shape: {img.shape}")

# Visualize the resized image
plt.imshow(rearrange(img, 'c h w -> h w c').numpy())
plt.title("Resized Image")
plt.axis("off")
plt.show()

# Step 2: Prepare the coordinate map (X, Y) -> (R, G, B)
def create_coordinate_map(img):
    """
    img: torch.Tensor of shape (num_channels, height, width)
    
    return: tuple of torch.Tensor of shape (height * width, 2) and torch.Tensor of shape (height * width, num_channels)
    """
    
    num_channels, height, width = img.shape
    
    # Create a 2D grid of (x,y) coordinates (h, w)
    w_coords = torch.arange(width).repeat(height, 1)
    h_coords = torch.arange(height).repeat(width, 1).t()
    w_coords = w_coords.reshape(-1)
    h_coords = h_coords.reshape(-1)

    # Combine the x and y coordinates into a single tensor
    X = torch.stack([h_coords, w_coords], dim=1).float()

    # Reshape the image to (h * w, num_channels)
    Y = rearrange(img, 'c h w -> (h w) c').float()
    
    return X, Y

X, Y = create_coordinate_map(img)

# Step 3: Create Random Fourier Features (RFF) using minibatching
def create_rff_features_batch(X, num_features, sigma, batch_size=100000):
    from sklearn.kernel_approximation import RBFSampler
    rff = RBFSampler(n_components=num_features, gamma=1/(2 * sigma**2))
    
    X_rff_list = []
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size].cpu().numpy()
        X_rff_batch = rff.fit_transform(X_batch)
        X_rff_list.append(torch.tensor(X_rff_batch, dtype=torch.float32).to(device))
    
    return torch.cat(X_rff_list, dim=0)

# Normalize the coordinates (X)
from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler(feature_range=(-1, 1)).fit(X)
X_scaled = scaler_X.transform(X)
X_scaled = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# Create RFF features using minibatching
num_rff_features = 7000  # Increase the number of features for better approximation
sigma = 0.008
X_rff = create_rff_features_batch(X_scaled, num_rff_features, sigma)

# Step 4: Define the Linear Regression model
class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid to keep output in range [0, 1]

# Step 5: Train the model
def train(net, X, Y, lr=0.001, epochs=5000):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")
    
    return net

# Initialize model
net = LinearModel(X_rff.shape[1], 3).to(device)
Y = Y.to(device)

# Train the model
net = train(net, X_rff, Y, lr=0.001, epochs=5000)

# Step 6: Reconstruct the image
def reconstruct_image(net, X_rff, original_img):
    num_channels, height, width = original_img.shape
    net.eval()
    
    with torch.no_grad():
        outputs = net(X_rff)
        reconstructed_img = outputs.reshape(height, width, num_channels)
    
    return reconstructed_img

reconstructed_img = reconstruct_image(net, X_rff, img)

# Step 7: Plot the original and reconstructed images
def plot_images(original_img, reconstructed_img):
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.imshow(rearrange(original_img, 'c h w -> h w c').cpu().numpy())
    ax0.set_title("Original Image")
    ax0.axis("off")

    ax1.imshow(reconstructed_img.cpu().numpy())
    ax1.set_title("Reconstructed Image (RFF)")
    ax1.axis("off")

    plt.tight_layout()
    plt.show()

plot_images(img, reconstructed_img)

# Step 8: Calculate MSE and PSNR
def calculate_metrics(original_img, reconstructed_img):
    original_img_np = rearrange(original_img, 'c h w -> (h w) c').cpu().numpy()
    reconstructed_img_np = reconstructed_img.cpu().numpy().reshape(-1, 3)  
    mse = mean_squared_error(original_img_np, reconstructed_img_np)
    psnr = 10 * log10(1 / mse)
    
    return mse, psnr

mse, psnr = calculate_metrics(img, reconstructed_img)
print(f"MSE: {mse:.6f}")
print(f"PSNR: {psnr:.2f} dB")