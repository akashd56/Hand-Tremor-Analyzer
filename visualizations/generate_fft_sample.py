import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters for a typical Parkinson's tremor (4-6 Hz)
fs = 100 # Hz
duration = 2.0 # seconds
t = np.arange(0, duration, 1/fs)

# Signal = 5Hz Tremor + Random Noise
tremor_freq = 5.2
signal = 0.5 * np.sin(2 * np.pi * tremor_freq * t) + 0.2 * np.random.normal(size=len(t))

# Calculate FFT
n = len(t)
yf = fft(signal)
xf = fftfreq(n, 1/fs)[:n//2]
psd = 2.0/n * np.abs(yf[0:n//2])

# Plotting
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal, color='#1f77b4', lw=1.5)
plt.title("Sample Tremor Time-Series (Input to 1D-CNN)", fontsize=14)
plt.ylabel("Amplitude")
plt.grid(alpha=0.3)

plt.subplot(2, 1, 2)
plt.fill_between(xf, psd, color='#ff7f0e', alpha=0.5)
plt.plot(xf, psd, color='#d62728', lw=2)
plt.title("FFT Power Spectrum (ALAMEDA Statistical Branch)", fontsize=14)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0, 15) # Parkinson's tremors are usually 3-12 Hz
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("visualizations/sample_tremor_fft_ppt.png", dpi=300)
print("FFT Visualization saved to: visualizations/sample_tremor_fft_ppt.png")
