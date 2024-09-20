import requests
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchaudio
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression
import soundfile as sf
import os

# Define the URL and local file path for the WAV file

wav_file_path = os.path.join(os.path.dirname(__file__), 'audio.wav')


# Load your audio sample
audio, sr = torchaudio.load(wav_file_path)
audio = audio[0]  # Use first channel if stereo
tm = audio.shape[0] / sr
print(f"Audio length: {tm} seconds")

# Trim to 5 seconds if necessary
if tm > 5:
    audio = audio[:5 * sr]

# Prepare time variable
X = torch.arange(0, len(audio)).unsqueeze(1).float()
X = X / X.max() * 200 - 100  # Scale to [-100, 100]

# Create RFF features
def create_rff_features(X, num_features, sigma):
    rff = RBFSampler(n_components=num_features, gamma=1 / (2 * sigma**2), random_state=13)
    X = X.cpu().numpy()
    X = rff.fit_transform(X)
    return torch.tensor(X, dtype=torch.float32)

num_features = 5000
sigma = 0.008
X_rff = create_rff_features(X, num_features, sigma)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_rff.numpy(), audio.numpy())

# Predict audio
pred_audio = model.predict(X_rff.numpy())
pred_audio_tensor = torch.tensor(pred_audio, dtype=torch.float32)

# Save reconstructed audio
sf.write('pred_audio.wav', pred_audio, sr)

# Calculate RMSE
rmse = torch.sqrt(torch.mean((pred_audio_tensor - audio) ** 2))
print(f"RMSE: {rmse.item():.6f}")

# Calculate SNR
signal_power = torch.mean(audio ** 2)
noise_power = torch.mean((pred_audio_tensor - audio) ** 2)
snr = 10 * torch.log10(signal_power / noise_power)
print(f"SNR: {snr.item():.2f} dB")

# Play reconstructed audio
from IPython.display import Audio
Audio(pred_audio, rate=sr)

# Plot original and reconstructed audio
plt.figure(figsize=(15, 4))
plt.plot(audio.numpy(), color='blue', alpha=0.7, label='Original Audio')
plt.plot(pred_audio_tensor.numpy(), color='red', alpha=0.7, label='Reconstructed Audio')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Original vs Reconstructed Audio Waveform')
plt.grid()
plt.legend()
plt.show()
