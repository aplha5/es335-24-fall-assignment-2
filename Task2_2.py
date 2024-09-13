import numpy as np
import librosa
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sounddevice as sd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

# Load the audio sample
def load_audio(file_path, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

# Play audio
def play_audio(audio, sr=22050):
    sd.play(audio, sr)
    sd.wait()

# Save audio
def save_audio(audio, file_path, sr=22050):
    wav.write(file_path, sr, (audio * 32767).astype(np.int16))  # Convert to 16-bit PCM format

# Reconstruction function
def reconstruct_audio(audio, sr=22050):
    t = np.arange(len(audio))

    # Apply Random Fourier Features
    rff = RBFSampler(gamma=0.1, n_components=100)  # Adjust gamma
    X_features = rff.fit_transform(t[:, np.newaxis])

    # Fit Linear Regression
    model = LinearRegression()
    model.fit(X_features, audio)

    # Predict using the model
    X_features_test = rff.transform(t[:, np.newaxis])
    audio_reconstructed = model.predict(X_features_test)

    return audio_reconstructed

# Calculate RMSE
def calculate_rmse(original, reconstructed):
    return np.sqrt(mean_squared_error(original, reconstructed))

# Calculate SNR
def calculate_snr(original, noise):
    return 10 * np.log10(np.mean(original ** 2) / np.mean(noise ** 2))

# Plot audio signals
def plot_audio_signals(original, reconstructed):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(original)
    plt.title('Original Audio')
    plt.subplot(2, 1, 2)
    plt.plot(reconstructed)
    plt.title('Reconstructed Audio')
    plt.tight_layout()
    plt.show()

# Main function
def main():
    file_path = os.path.join(os.path.dirname(__file__), 'audio.mp3')  # Ensure correct file path
    audio = load_audio(file_path)
    
    # Reconstruct audio
    audio_reconstructed = reconstruct_audio(audio)

    # Save and play the reconstructed audio
    save_audio(audio_reconstructed, 'reconstructed_audio.wav')
    play_audio(audio)
    play_audio(audio_reconstructed)

    # Calculate metrics
    rmse = calculate_rmse(audio, audio_reconstructed)
    noise = audio - audio_reconstructed
    snr = calculate_snr(audio, noise)

    # Output metrics
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Signal-to-Noise Ratio (SNR): {snr} dB')

    # Plot audio signals
    plot_audio_signals(audio, audio_reconstructed)

if __name__ == "__main__":
    main()
