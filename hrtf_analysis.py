import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from librosa import load

phi = ["-20", "-10", "0", "10", "20"]
theta = ["350", "355", "000", "005", "010"]
path_left = "hrtf_dataset/elev{}/L{}e{}a.wav"
pos_left = [path_left.format(phi[i], phi[i], theta[2]) for i in range(len(theta))]
path_right = "hrtf_dataset/elev{}/R{}e{}a.wav"

# sr, audioLeft = wavfile.read(filename=path_left, mmap=True)
# audioLeft = (audioLeft.max() - audioLeft) / (audioLeft.max() - audioLeft.min()) 
# audioLeft -= np.sum(audioLeft) / audioLeft.shape[0]
# sr, audioRight = wavfile.read(filename=path_right, mmap=True)
sr = 44100
audiosLeft = [load(path=pos_left[i], sr=sr)[0] for i in range(len(pos_left))]
# audioRight = load(path=path_right, sr=sr)[0]

N = audiosLeft[0].shape[0]
t = np.linspace(0, (N-1)/sr, N)
freqs = np.linspace(0, (N-1)*sr/N, N)

def extract_freq_info(signal): 
    S = np.abs(fft(signal))
    return S

def plot_log_spectrum(signal, S, phase, t=t, freqs=freqs):
    Slog = Slog = np.log10(S)
    phase = np.angle(fft(signal), deg=True)

    fig, axs = plt.subplots(ncols=3, nrows=1)
    axs[1].plot(t, signal, label="log(|X(f)|)")
    axs[1].grid(True)
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("|X(f)|dB")
    axs[1].legend()

    axs[1].semilogx(freqs, Slog, label="log(|X(f)|)")
    axs[1].grid(True)
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("|X(f)|dB")
    axs[1].legend()

    axs[2].semilogx(freqs, phase, label="phi(f)")
    axs[2].grid(True)
    axs[2].set_xlabel("Frequency [Hz]")
    axs[2].set_ylabel("Phi(f)")
    axs[2].legend()


def plot_spectrum(signals, Spectrums, t=t, freqs=freqs, 
                  phi=phi, theta=theta, constant=theta[2]): 
    fig, axs = plt.subplots(nrows=2, ncols=1)
    N_freq = (len(freqs) // 2) + 1

    for signal, S, p, th in zip(signals, Spectrums, phi, theta):
        th = constant
        axs[0].plot(t, signal, label=f"x(n) @ phi={p}°, theta={th}°")
        axs[0].grid(True)
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("x(n)")
        axs[0].legend()

        axs[1].plot(freqs[:N_freq], S[:N_freq], label=f"|X(f)| @ phi={p}°, theta={th}°")
        axs[1].grid(True)
        axs[1].set_xlabel("Frequency [Hz]")
        axs[1].set_ylabel("|X(f)|")
        axs[1].legend()

    plt.tight_layout()
    plt.show()

Spectrums = [extract_freq_info(signal=audiosLeft[i]) for i in range(len(audiosLeft))]
plot_spectrum(signals=audiosLeft, Spectrums=Spectrums)
