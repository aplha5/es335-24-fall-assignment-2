import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import soundfile as sf
from IPython.display import Audio, display

# Load audio file (you can replace this with your actual file path)
file_path = os.path.join(os.path.dirname(__file__), 'audio.mp3')
audio, sr = torchaudio.load(file_path)

# Trim the audio to the first 5 seconds
duration = 5
audio = audio[0, :sr * duration]  # Trim to 5 seconds
t = torch.linspace(0, duration, steps=audio.shape[0])  # Time points (t)

# Plot the original audio waveform
def audio_plot(audio, sr, title):
    plt.figure(figsize=(12, 4))
    plt.plot(audio, color='blue', alpha=0.6)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid()
    plt.show()

audio_plot(audio, sr, 'Original Audio Waveform')

# RFF feature transformation
def create_rff_features(X, num_features, sigma):
    rff = RBFSampler(n_components=num_features, gamma=1 / (2 * sigma**2), random_state=42)
    X_rff = rff.fit_transform(X.cpu().numpy().reshape(-1, 1))
    return torch.tensor(X_rff, dtype=torch.float32)

num_features = 5000
sigma = 0.1
X_rff = create_rff_features(t, num_features, sigma)

# Apply Linear Regression to learn mapping from time (t) to amplitude (A)
model = LinearRegression()
model.fit(X_rff.numpy(), audio.numpy())

# Predict the reconstructed audio using the learned model
pred_audio = model.predict(X_rff.numpy())

# Save the reconstructed audio
sf.write('reconstructed_audio.wav', pred_audio, sr)

# Plot the reconstructed audio waveform
audio_plot(pred_audio, sr, 'Reconstructed Audio Waveform')

# Play the original and reconstructed audio for comparison
print("Original Audio:")
display(Audio(audio.numpy(), rate=sr))  # Play original audio

print("Reconstructed Audio:")
display(Audio(pred_audio, rate=sr))  # Play reconstructed audio

# Evaluate the reconstruction using RMSE and SNR
def calculate_rmse(original, reconstructed):
    return np.sqrt(mean_squared_error(original, reconstructed))

def calculate_snr(original, reconstructed):
    signal_power = np.mean(np.square(original))
    noise_power = np.mean(np.square(original - reconstructed))
    return 10 * np.log10(signal_power / noise_power)

rmse = calculate_rmse(audio.numpy(), pred_audio)
snr = calculate_snr(audio.numpy(), pred_audio)

print(f"RMSE: {rmse}")
print(f"SNR: {snr} dB")

# Plot the original vs reconstructed audio for visual comparison
plt.figure(figsize=(12, 4))
plt.plot(audio.numpy(), label='Original Audio', color='blue', alpha=0.6)
plt.plot(pred_audio, label='Reconstructed Audio', color='red', alpha=0.6)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original vs Reconstructed Audio')
plt.grid()
plt.legend()
plt.show()
