import numpy as np 
import matplotlib.pyplot as plt 
import utils 
import core
import scipy.fft as fft

# Data augmentation: 
# Add delays: {90°, 180°}
# Add color noise: white and pink 

# Source elevation order: -10°, ..., -40°, 0°, 10°, ..., 90°
hrir_left = "dataset/hrirs/hrir_left.npy"
hrir_left = np.load(hrir_left)
hrir_right = "dataset/hrirs/hrir_right.npy"
hrir_right = np.load(hrir_right)

"""
The main idea here is to augment each elevation data by a combination of delays and 
noise superposition. 
"""

def pink_noise(N, d=0.5): 
    X_white = np.fft.rfft(np.random.randn(N))
    S = np.fft.rfftfreq(N)
    S_pink = np.zeros_like(S)
    for i in range(0, len(S)):
        f = S[i]
        S_pink[i] = 1/np.where(f == 0, float('inf'), f**d)
    S_pink = S_pink / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S_pink
    noise = np.fft.irfft(X_shaped)
    noise = (np.max(noise) - noise) / (np.max(noise) - np.min(noise))
    return noise

# sample rate: 
fs = 44100
# samples: 
N = 512
t = np.linspace(0,N/fs,N)
print(t[-1])

# Noise: 
# white: 
mu = 0.0
sigma = 1.0
noise = np.random.normal(loc=mu, scale=np.sqrt(sigma), size=N)
pnoise = pink_noise(N=N)
SNR = 40 # [dB]

# By clipping both noises to [-1,1], and using noise amplitude as 0.000625 we get a SNR = 40.9

def zero_pad(signal, length=1024):
    pad_length = length - signal.shape[0]
    zeros = np.zeros(shape=(pad_length,))
    x_padded = np.concatenate([signal, zeros], axis=-1)
    return x_padded

def delay(signal, 
          delay_secs, 
          sample_rate=44100,
          samples=1024): 
    delay_samples = round(delay_secs * sample_rate)
    delay_fir = np.zeros(shape=(delay_samples,))
    delay_fir[-1] = 1
    delay_signal = np.convolve(signal, delay_fir)

    if delay_signal.shape[0] <= samples: 
        delay_signal = zero_pad(delay_signal)

    return delay_signal

DELAY_SECS90 = N / fs * 0.25 
DELAY_SECS180 = N / fs * 0.5
DELAY_SECS270 = N /fs * 0.75
DELAY_SECS360 = N / fs * 1.0
FINAL_SAMPLES = 1024


def data_upsample(): 
    hrir_left1024 = []
    for hLeft in hrir_left: 
        hLeft = zero_pad(hLeft)
        hrir_left1024.append(hLeft)

    hrir_left1024 = np.array(hrir_left1024)
    print(f"Zero padded dataset shape: {hrir_left1024.shape}")
    filename = "dataset/hrirs/hrir_left_1024.npy"
    np.save(filename, hrir_left1024)

    hrir_right1024 = []
    for hRight in hrir_right: 
        hRight = zero_pad(hRight)
        hrir_right1024.append(hRight)

    hrir_right1024 = np.array(hrir_right1024)
    print(f"Zero padded dataset shape: {hrir_right1024.shape}")
    filename = "dataset/hrirs/hrir_right_1024.npy"
    np.save(filename, hrir_right1024)

def data_delay(delay_secs, 
               delay_deg, 
               noise=True, 
               amplitude=0.000625): 
    hrir_left1024 = []
    for hLeft in hrir_left: 
        hLeft = delay(hLeft, delay_secs)
        if noise:
            hLeft += amplitude * np.clip(np.random.normal(size=hLeft.shape[0]), -1 , 1)
        hrir_left1024.append(hLeft)

    hrir_left1024 = np.array(hrir_left1024)
    print(f"Zero padded dataset shape: {hrir_left1024.shape}")
    filename = f"dataset/hrirs/hrir_left_delay_{delay_deg}_1024.npy"
    if noise: 
        filename = f"dataset/hrirs/hrir_left_noisy_delay_{delay_deg}_1024.npy"
    np.save(filename, hrir_left1024)

    hrir_right1024 = []
    for hRight in hrir_right: 
        hRight = delay(hRight, delay_secs)
        if noise:
            hRight += amplitude * np.clip(np.random.normal(size=hRight.shape[0]), -1 , 1)
        hrir_right1024.append(hRight)

    hrir_right1024 = np.array(hrir_right1024)
    print(f"Zero padded dataset shape: {hrir_right1024.shape}")
    filename = f"dataset/hrirs/hrir_right_delay_{delay_deg}_1024.npy"
    if noise: 
        filename = f"dataset/hrirs/hrir_right_noisy_delay_{delay_deg}_1024.npy"
    np.save(filename, hrir_right1024)

# data_delay(delay_secs=DELAY_SECS90, delay_deg="90")

# data_delay(delay_secs=DELAY_SECS180, delay_deg="180")

# data_delay(delay_secs=DELAY_SECS270, delay_deg="270")

# data_delay(delay_secs=DELAY_SECS360, delay_deg="360")


add_ons_left = ["left_1024", "left_delay_90_1024", "left_delay_180_1024", 
           "left_delay_270_1024", "left_delay_360_1024", "left_noisy_delay_90_1024",
           "left_noisy_delay_180_1024", "left_noisy_delay_270_1024", "left_noisy_delay_360_1024"]
add_ons_right = ["right_1024", "left_delay_90_1024", "right_delay_180_1024", 
           "right_delay_270_1024", "right_delay_360_1024", "right_noisy_delay_90_1024",
           "right_noisy_delay_180_1024", "right_noisy_delay_270_1024", "right_noisy_delay_360_1024"]

coordinates_path = "dataset/tags/cartesian_coordinates.npy"
coord = np.load(coordinates_path)
full_data = []
coordinates = []
i = 0 
for add_on in add_ons_right:
    hrir_path = f"dataset/hrirs/hrir_{add_on}.npy"
    hrir_dataset = np.load(hrir_path)
    if i == 0: 
        full_data = hrir_dataset
        coordinates = coord
    else: 
        full_data = np.concatenate([full_data, hrir_dataset], axis=0)
        coordinates = np.concatenate([coordinates, coord], axis=0)
    i += 1

np.save("cartesian_coordinates_augmented.npy", coordinates)
np.save("dataset/data_augmented__hrir_right.npy", full_data)













