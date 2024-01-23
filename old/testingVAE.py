import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils as utl
import core
from VAE import get_models

LATENT_SPACE = True
# Encoder extraction: 
def pull_module(model, module_name="encoder"): 
    for layer in model.layers:
        if layer.name == module_name:
            module = layer
    return module


# Model VAE Testing: --------------------------------------------------------------------------------------
VAE, _, _ = get_models()
weights = f"training data/weights/weights_cvaeVAE_on_batch128_epochs300.h5"
VAE.load_weights(weights)

# dataset:
path_data_left = "dataset/data_augmented__hrir_left.npy"
path_data_right = "dataset/data_augmented__hrir_right.npy"
data_left = np.load(path_data_left)
data_right = np.load(path_data_right)
dataset = np.concatenate([data_left[:, np.newaxis, :], 
                          data_right[:, np.newaxis, :]], axis=1)

# data validation: 
data_val = "dataset/vae/validation_dataset.npy"
data_val = np.load(data_val)


# Latent space: 
if LATENT_SPACE:
    encoder = pull_module(model=VAE)
    moments = encoder(dataset)
    z = moments[0]
    z_left, z_right = z[:, 0, :], z[:, 1, :]
    utl.tsne_plotting(z_left=z_left,
                    z_right=z_right)

# Forward Propagation: 
i = 10
x = dataset[i]
x_left = x[0,:]
x_right = x[1,:]
x = x[tf.newaxis, :, :]
xh_left, xh_right = VAE(x)
xh_left = xh_left[0]
xh_right = xh_right[0]


utl.plot2validate(x=x_left, 
                  x_hat=xh_left, 
                  N=1024)
utl.plot2validate(x=x_right, 
                  x_hat=xh_right, 
                  N=1024)
utl.plot_hrirs(x_left=x_left,
               x_right=x_right,
               xh_left=xh_left,
               xh_right=xh_right, 
               N=1024)
utl.plot2signals(x=x_right, x_hat=xh_right, N=1024)

# Audio spatialization: --------------------------------------------------------------------------------------
# 2 sec of a sine sweep:
sine_sweep = core.SineSweep()
# 2 sec of pink noise:
fs = 44100
secs = 2
hrir_length = 512
Ns = (int(secs * fs / hrir_length) + 1) * hrir_length
pink_noise, _ = core.gen_pink_noise(N=Ns)

utl.convAudio3D(x=sine_sweep,
            hrir_left=x_left,
            hrir_right=x_right,
            filename="sweep3D_original")
utl.convAudio3D(x=sine_sweep,
            hrir_left=xh_left,
            hrir_right=xh_right,
            filename="sweep3D_recon")
utl.convAudio3D(x=pink_noise,
            hrir_left=x_left,
            hrir_right=x_right,
            filename="pnoise3D_original")
utl.convAudio3D(x=pink_noise,
            hrir_left=xh_left,
            hrir_right=xh_right,
            filename="pnoise3D_recon")