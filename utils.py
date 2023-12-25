import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE

# Spatial noise characteristics:
sample_rate = 44100
hrtf_length = 512
samples = hrtf_length

# Plotting aux: --------------------------------------------------------------------------------------------------------------------------
def plot_signal(x, N=samples, fs=sample_rate): 
    """
    Plots signal in both time domain and frequency domain.
    """
    t = np.linspace(0, (N-1) / fs, N)
    f = np.linspace(0, (N-1) * fs / N, N)
    X = np.fft.rfft(x)
    X_mag = np.abs(X)
    X_log = 20*np.log10(X_mag)
    N_freq = X_mag.shape[0]
    phi = np.angle(X)

    fig, axs = plt.subplots(nrows=3, figsize=(9,6))
    # Subplot 1: Time domain
    axs[0].plot(t, x)
    axs[0].grid(True)
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("x(n)")

    # Subplot 2: Logarithmic Scale - Magnitude
    axs[1].semilogx(f[:N_freq], X_log)
    axs[1].grid(True)
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("|X(w)|dB")

    # Subplot 3: Logarithmic Scale - Phase
    axs[2].semilogx(f[:N_freq], phi)
    axs[2].grid(True)
    axs[2].set_xlabel("Frequency [Hz]")
    axs[2].set_ylabel("Phi(w)")

    plt.tight_layout()
    plt.show()
    
def plot2validate(x, x_hat, N=samples, fs=sample_rate): 
    """
    Plots signal in both time domain and frequency domain.
    """
    t = np.linspace(0, (N-1) / fs, N)
    f = np.linspace(0, (N-1) * fs / N, N)

    X = np.fft.rfft(x)
    X_hat = np.fft.rfft(x_hat)
    X_mag = np.abs(X)
    X_hat_mag = np.abs(X_hat)
    X_log = 20*np.log10(X_mag)
    X_hat_log = 20*np.log10(X_hat_mag)
    phi = np.angle(X)
    phi_hat = np.angle(X_hat)

    N_freq = X_mag.shape[0]

    fig, axs = plt.subplots(nrows=3, figsize=(9,6))
    # Subplot 1: Time domain
    axs[0].plot(t, x, label="Original")
    axs[0].plot(t, x_hat, label="Reconstructed")
    axs[0].grid(True)
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("x(n)")
    axs[0].legend()

    # Subplot 2: Logarithmic Scale - Magnitude
    axs[1].semilogx(f[:N_freq], X_log, label="Original")
    axs[1].semilogx(f[:N_freq], X_hat_log, label="Reconstructed")
    axs[1].grid(True)
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("|X(w)|dB")
    axs[1].legend()

    # Subplot 3: Logarithmic Scale - Phase
    axs[2].semilogx(f[:N_freq], phi, label="Original")
    axs[2].semilogx(f[:N_freq], phi_hat, label="Reconstructed")
    axs[2].grid(True)
    axs[2].set_xlabel("Frequency [Hz]")
    axs[2].set_ylabel("Phi(w)")
    axs[2].legend()

    plt.tight_layout()
    plt.show()



def plot2signals(x_hat1, x_hat2, N=samples, fs=sample_rate): 
    t = np.linspace(0, (N-1) / fs, N)
    f = np.linspace(0, (N-1) * fs / N, N)

    X_hat1 = np.fft.rfft(x_hat1)
    X_hmag1 = np.abs(X_hat1)
    X_hlog1 = 20*np.log10(X_hmag1)
    phi1 = np.angle(X_hat1)
    N_freq = X_hmag1.shape[0]

    X_hat2 = np.fft.rfft(x_hat2)
    X_hmag2 = np.abs(X_hat2)
    X_hlog2 = 20*np.log10(X_hmag2)
    phi2 = np.angle(X_hat2)

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15,10))
    # signal time domain:
    axs[0, 0].plot(t, x_hat1)
    axs[0, 0].grid(True)
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("x(t)")
    # Signal frequency domain:
    # Magnitude:
    axs[1, 0].semilogx(f[:N_freq], X_hlog1)
    axs[1, 0].grid(True)
    axs[1, 0].set_xlabel("Frequency [Hz]")
    axs[1, 0].set_ylabel("|X(w)|dB")
    # Phase:
    axs[2, 0].semilogx(f[:N_freq], phi1)
    axs[2, 0].grid(True)
    axs[2, 0].set_xlabel("Frequency [Hz]")
    axs[2, 0].set_ylabel("Phi(w)")
    # signal time domain:
    axs[0, 1].plot(t, x_hat2)
    axs[0, 1].grid(True)
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("x(t)")
    # Signal frequency domain:
    # Magnitude:
    axs[1, 1].semilogx(f[:N_freq], X_hlog2)
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_ylabel("|X(w)|dB")
    # Phase:
    axs[2, 1].semilogx(f[:N_freq], phi2)
    axs[2, 1].grid(True)
    axs[2, 1].set_xlabel("Frequency [Hz]")
    axs[2, 1].set_ylabel("Phi(w)")
    plt.tight_layout()
    plt.show()

def plot_hrirs(x_left, xh_left, x_right, xh_right, 
               N=samples, fs=sample_rate): 
    """
    Plots two signals to compare.
    """
    t = np.linspace(0, (N-1) / fs, N)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].plot(t, x_left)
    axs[0, 0].set_title(f"Left IR time domain")
    axs[0, 0].grid(True)
    axs[0, 0].set_xlabel("time [s]")
    axs[0, 0].set_ylabel("x_left(t)")

    axs[1, 0].plot(t, x_right)
    axs[1, 0].set_title("Right IR time domain") 
    axs[1, 0].grid(True)
    axs[1, 0].set_xlabel("time [s]")
    axs[1, 0].set_ylabel("x_right(t)")

    axs[0, 1].plot(t, xh_left)
    axs[0, 1].set_title(f"Left hat IR time domain")
    axs[0, 1].grid(True)
    axs[0, 1].set_xlabel("time [s]")
    axs[0, 1].set_ylabel("xh_left(t)")

    axs[1, 1].plot(t, xh_right)
    axs[1, 1].set_title("Right hat IR time domain") 
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel("time [s]")
    axs[1, 1].set_ylabel("xh_right(t)")
    plt.tight_layout()
    plt.show()

# Latent Space: z
def tsne_plotting(z_left, z_right, n_components=2):
    tsne = TSNE(n_components=n_components)
    X_tsne_left = tsne.fit_transform(X=z_left)
    X_tsne_right = tsne.fit_transform(X=z_right)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    axs[0].scatter(x=X_tsne_left[:, 0], y=X_tsne_left[:, 1], c="blue")
    axs[0].set_xlabel('X-axis')
    axs[0].set_ylabel('Y-axis')
    axs[0].grid()
    axs[0].set_title('t-SNE latent Z left variables')
    axs[1].scatter(x=X_tsne_right[:, 0], y=X_tsne_right[:, 1], c="orange")
    axs[1].set_xlabel('X-axis')
    axs[1].set_ylabel('Y-axis')
    axs[1].grid()
    axs[1].set_title('t-SNE latent Z right variables')
    plt.tight_layout()
    plt.show()

# Sphere scatter plot: 
def plot3D(x, y, z): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="blue")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.set_title('MIT anechoic chamber')
    plt.show()


# Spatial Audio: ---------------------------------------------------------------------------------------------------------------------
def audio3D(x_left, x_right, filename, fs=sample_rate): 
    stereo_audio = np.column_stack([x_left, x_right])
    sf.write(f"{filename}.wav", stereo_audio, samplerate=fs, format='WAV')

def convAudio3D(x, hrir_left, hrir_right, filename):
    x_left = np.convolve(x, hrir_left, mode="same")
    x_right = np.convolve(x, hrir_right, mode="same")
    audio3D(x_left=x_left, x_right=x_right, filename=filename)
    return x_left, x_right

# Extraction from database: ----------------------------------------------------------------------------------------------------------
def extract_hrir(elev, azimuth, sr=sample_rate): 
    """
    For plotting signal Impulse Response. 
    Imports an specific hrtf from (elev, azimuth) given the KEMAR's dataset into a tensor.
    """
    if azimuth <= 99: 
        if azimuth < 10: 
            path_left_channel = "hrtf_dataset/elev{}/L{}e00{}a.wav".format(elev, elev, azimuth)
            path_right_channel = "hrtf_dataset/elev{}/R{}e00{}a.wav".format(elev, elev, azimuth)
        else:
            path_left_channel = "hrtf_dataset/elev{}/L{}e0{}a.wav".format(elev, elev, azimuth)
            path_right_channel = "hrtf_dataset/elev{}/R{}e0{}a.wav".format(elev, elev, azimuth)
    else: 
        path_left_channel = "hrtf_dataset/elev{}/L{}e{}a.wav".format(elev, elev, azimuth)
        path_right_channel = "hrtf_dataset/elev{}/R{}e{}a.wav".format(elev, elev, azimuth)
    x_right = librosa.load(path_right_channel, sr=sr)[0]
    x_left = librosa.load(path_left_channel, sr=sr)[0]
    return x_right, x_left