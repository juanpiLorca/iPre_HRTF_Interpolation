import crepe 
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile

f0 = 440
f1 = 880 
f2 = 1720 
sr = 44100
secs = 1
t = np.linspace(0, secs, sr)
wave = 1.0 * np.sin(2*np.pi*f0*t) + 1.0 * np.sin(2*np.pi*f1*t) + 0.25 * np.sin(2*np.pi*f2*t)
sf.write(file="test-tone.wav", data=wave, samplerate=sr, format="WAV")


elev = "30"
azimuth = "030"
path = "hrtf_dataset/elev0/L0e000a.wav"

sr, audio = wavfile.read(path)
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sr} [Hz]")

samples = 1024
pad_width = samples - audio.shape[0]
audio = np.pad(audio, pad_width=pad_width, mode='constant', constant_values=0)
print(audio.shape)

time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
print(frequency)

print(confidence)


