import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE

import utils

def tsne_plotting(X, label, color, n_components=2):
    tsne = TSNE(n_components=n_components)
    X_tsne = tsne.fit_transform(X)
    plt.grid()
    plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c=color)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f't-SNE KEMAR dataset: {label}')
    plt.show()


def plot3D(x, y, z): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="blue")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.set_title('MIT anechoic chamber')
    plt.show()

# Plotting coordinates: 
tags = "dataset/tags/cartesian_coordinates.npy"
tags = np.load(tags)
x = tags[:, 0]
y = tags[:, 1]
z = tags[:, 2]
plot3D(x=x, y=y, z=z)

# TSNE for the dataset: 
hrir_left = "dataset/hrirs/hrir_left.npy"
hrir_left = np.load(hrir_left)
hrir_right = "dataset/hrirs/hrir_right.npy"
hrir_right = np.load(hrir_right)

tsne_plotting(X=hrir_left, 
              label="HRIRs left", 
              color="blue")
tsne_plotting(X=hrir_right, 
              label="HRIRs right", 
              color="orange")

# Plotting specific HRIR pair: 
# sample rate: 
fs = 44100
# samples: 
N = 512
t = np.linspace(0,N/fs,N)

phi = 90
theta = 0
h_left, h_right = utils.extract_hrir(elev=phi, 
                                     azimuth=theta)

# fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,4))
# axs[0].plot(t, h_left)
# axs[0].set_title("HRIR left (90°, 0°)")
# axs[0].set_xlabel("Time [s]")
# axs[0].set_ylabel("Amplitude")
# axs[0].grid()
# axs[1].plot(t, h_right)
# axs[1].set_title("HRIR right (90°, 0°)")
# axs[1].set_xlabel("Time [s]")
# axs[1].set_ylabel("Amplitude")
# axs[1].grid()
# plt.tight_layout()
# plt.show()

# fig = plt.figure(figsize=(9,4))
# plt.plot(t, h_right)
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.grid()
# plt.show()

