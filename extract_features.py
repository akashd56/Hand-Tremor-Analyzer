import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import os
import glob

def calculate_fft_features(signal, fps=10):
    n = len(signal)
    if n < 2:
        return [0] * 24
    
    # Remove DC component
    signal = signal - np.mean(signal)
    
    fft_values = np.abs(fft(signal))
    freqs = fftfreq(n, d=1/fps)
    
    # Keep only positive frequencies
    pos_mask = freqs > 0
    fft_values = fft_values[pos_mask]
    freqs = freqs[pos_mask]
    
    if len(fft_values) == 0:
        return [0] * 24

    mean = np.mean(fft_values)
    std = np.std(fft_values)
    var = np.var(fft_values)
    median = np.median(fft_values)
    
    # Dom freq
    dom_idx = np.argmax(fft_values)
    dom_freq = freqs[dom_idx]
    
    return [
        mean, std, var, 0, 0, # Placeholder for avg_diff_mean, above_mean_rt
        median, 0, 0, # med_dev, iqr
        skew(fft_values) if len(fft_values) > 2 else 0,
        kurtosis(fft_values) if len(fft_values) > 2 else 0,
        np.min(fft_values), np.max(fft_values),
        np.max(fft_values) - np.min(fft_values),
        0, 0, 0, # peaks_rt, rest_rt, ssc_rt
        np.sqrt(np.mean(fft_values**2)), # rms
        np.sum(fft_values**2), # tot_power
        0, # dom_freq_rt
        np.sum(fft_values**2), # energy
        dom_freq,
        0, 0, 0 # pw_ar, entropy, flatness
    ]

def extract_features_from_data(data_points):
    # data_points: (N, 63)
    # Use centroid of [0, 4, 8, 12, 16, 20]
    indices = [0, 4, 8, 12, 16, 20]
    coords = data_points.reshape(-1, 21, 3)[:, indices, :] # (N, 6, 3)
    centroid = np.mean(coords, axis=1) # (N, 3)
    
    # Magnitude of movement (delta from previous frame)
    deltas = np.diff(centroid, axis=0) # (N-1, 3)
    magnitude = np.sqrt(np.sum(deltas**2, axis=1))
    
    if len(magnitude) == 0:
        return [0] * 95
    
    # Time-domain features
    m_mean = np.mean(magnitude)
    m_std = np.std(magnitude)
    m_var = np.var(magnitude)
    m_median = np.median(magnitude)
    m_skew = skew(magnitude) if len(magnitude) > 2 else 0
    m_kurt = kurtosis(magnitude) if len(magnitude) > 2 else 0
    m_min = np.min(magnitude)
    m_max = np.max(magnitude)
    
    time_feats = [
        m_mean, m_std, m_var, 0, 0, # avg_diff, above_mean
        m_median, 0, 0, # med_dev, iqr
        m_skew, m_kurt, m_min, m_max, m_max - m_min,
        0, 0, 0, # peaks, rest, ssc
        np.sqrt(np.mean(magnitude**2)), # rms
        np.sum(magnitude**2), # energy
        0, 0 # sampen, dfa
    ]
    
    # PC1 placeholder (use Y-axis as it's most relevant for tremors in this app)
    y_signal = centroid[:, 1]
    y_mean = np.mean(y_signal)
    y_feats = [y_mean, np.mean(np.abs(y_signal)), np.std(y_signal), np.var(y_signal)] + [0]*20
    
    # FFT features
    fft_feats = calculate_fft_features(magnitude)
    y_fft_feats = calculate_fft_features(y_signal)
    
    return time_feats + y_feats + fft_feats + y_fft_feats

# Load our data
TRAIN_CSV = "assests/dataset/train.csv"
DATA_DIR = "assests/dataset/data/"
meta_data = pd.read_csv(TRAIN_CSV)

extracted_features = []
print("Extracting ALAMEDA-like features from our dataset...")
for i in tqdm(range(meta_data.shape[0])):
    data_file = os.path.join(DATA_DIR, meta_data["data_file_name"][i])
    if not os.path.exists(data_file):
        extracted_features.append([0] * 92) # Approx 92 features
        continue

    temporary_data = pd.read_csv(data_file)
    data_points = temporary_data.drop("TIME", axis=1).values
    
    features = extract_features_from_data(data_points)
    extracted_features.append(features)

df_features = pd.DataFrame(extracted_features)
df_features['label'] = meta_data['folder_path']
df_features.to_csv("assests/dataset/our_features.csv", index=False)
print("Features saved to assests/dataset/our_features.csv")
