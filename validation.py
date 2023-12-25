import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils as utl
import core
from Model import get_models

LATENT_SPACE = False

# Loss function values: (64-dim latent space)
num_epochs = 500
loss1 = "training data/losses per epoch/losses_cvae0_2_on_batch64_epochs500.npy"
loss1 = np.load(loss1)
print(f"Loss convergence model 1: {loss1[num_epochs-1]}")
loss2 = "training data/losses per epoch/losses_cvae1_2_on_batch64_epochs500.npy"
loss2 = np.load(loss2)
print(f"Loss convergence model 2: {loss2[num_epochs-1]}")
loss3 = "training data/losses per epoch/losses_cvae2_2_on_batch64_epochs500.npy"
loss3 = np.load(loss3)
print(f"Loss convergence model 3: {loss3[num_epochs-1]}")

# Plotting losses:
def plot_losses(loss1,
                loss2,
                tag1,
                tag2,
                num_epochs=num_epochs):
    epochs = np.arange(num_epochs)
    plt.plot(epochs, loss1, label=tag1)
    plt.plot(epochs, loss2, label=tag2)
    plt.xlabel('epochs')
    plt.ylabel('Multi-Spectral Loss')
    plt.legend()
    plt.show()

# Log Spectral Distortion metric: 
def metric(x, x_hat): 
    X = core.metrics_logmag(x)
    X_hat = core.metrics_logmag(x_hat)
    L = x.shape[0]
    K = X.shape[1] 
    LK = K * L 
    error = X - X_hat
    error = (1/LK) * error**2
    lsd = np.sum(error)
    return lsd

# Encoder extraction: 
def pull_module(model, 
                module_name="encoder"): 
    for layer in model.layers:
        if layer.name == module_name:
            module = layer
    return module


# Loss validation: --------------------------------------------------------------------------------------
num1 = "0"
num2 = "0_2"
batch = 64
loss1 = f"training data/losses per epoch/losses_cvae{num1}_on_batch{batch}_epochs{num_epochs}.npy"
loss1 = np.load(loss1)
loss2 = f"training data/losses per epoch/losses_cvae{num2}_on_batch{batch}_epochs{num_epochs}.npy"
loss2 = np.load(loss2)
plot_losses(loss1=loss1,
            tag1="latent dim = 32",
            loss2=loss2,
            tag2="latent dim = 64")


# Model validation: --------------------------------------------------------------------------------------
# num1 ==> latent dim = 32
# num2 ==> latent dim = 64 (BETTER PERFORMANCE)
num = num2
CVAE = get_models()
weights = f"training data/weights/weights_cvae{num}_on_batch{batch}_epochs{num_epochs}.h5"
CVAE.load_weights(weights)

# dataset:
hrir_left = "dataset/hrirs/hrir_left.npy"
hrir_left = core.tf_float32(np.load(hrir_left))
hrir_left = hrir_left[:, tf.newaxis, :]

hrir_right = "dataset/hrirs/hrir_right.npy"
hrir_right = core.tf_float32(np.load(hrir_right))
hrir_right = hrir_right[:, tf.newaxis, :]

spatial_info_path = "dataset/tags/radians_spherical.npy"
spatial_info = np.load(spatial_info_path)
spatial_info = core.tf_float32(spatial_info)
#spatial_info = tf.math.l2_normalize(spatial_info)
spatial_info = spatial_info[:, tf.newaxis, :]

dataset = tf.concat([hrir_left, hrir_right], axis=1)
spatial_info = tf.concat([spatial_info, spatial_info], axis=1)
_dataset_ = tf.concat([dataset, spatial_info], axis=-1)

# data validation: 
data_val = "dataset/shuffle to train/validation_spherical_data.npy"
data_val = np.load(data_val)

# data preprocess: 
num_coordinates = 3
elev_azimuth = 2
hrir_val = data_val[:, :, :-(num_coordinates + elev_azimuth)]
spatial_info = data_val[:, 0, -(num_coordinates+elev_azimuth):]
spatial_condition = core.tf_float32(spatial_info[:, :-elev_azimuth])
#spatial_condition = tf.math.l2_normalize(spatial_condition)
spatial_condition = tf.tile(spatial_condition[:, tf.newaxis, :], multiples=[1,2,1])
_hrir_val_ = tf.concat([hrir_val, spatial_condition], axis=-1)

# Latent space: 
if LATENT_SPACE:
    encoder = pull_module(model=CVAE)
    moments = encoder(_dataset_)
    z = moments[0]
    z_left, z_right = z[:, 0, :], z[:, 1, :]
    utl.tsne_plotting(z_left=z_left,
                    z_right=z_right)

# Forward Propagation: 
# sample extraction:
i = 10
phi = int(spatial_info[i, 3])
theta = int(spatial_info[i, 4])
print("======================================")
print(f"phi angle: {phi}°, theta angle: {theta}°")
print("======================================")

x = _hrir_val_[i]
x = x[tf.newaxis, :, :]
xh_left, xh_right = CVAE(x)
xh_left = xh_left[0]
xh_right = xh_right[0]

# HRIR extraction from MIT's KEMAR dataset:
x_left, x_right = utl.extract_hrir(elev=phi, azimuth=theta)

utl.plot2validate(x=x_left, 
                  x_hat=xh_left)
utl.plot2validate(x=x_right, 
                  x_hat=xh_right)
utl.plot_hrirs(x_left=x_left,
               x_right=x_right,
               xh_left=xh_left,
               xh_right=xh_right)
utl.plot2signals(x=x_right, x_hat=xh_right)


# Metrics computations: --------------------------------------------------------------------------------------
# Left & metrics: data validation (100) & dataset (710)
# 1.- data validation
xval_hat_left, xval_hat_right = CVAE(_hrir_val_)
xval_left = _hrir_val_[:, 0, :-num_coordinates]
xval_right = _hrir_val_[:, 1, :-num_coordinates]
metric_left_100 = metric(x=xval_left, 
                         x_hat=xval_hat_left)
metric_right_100 = metric(x=xval_right, 
                         x_hat=xval_hat_right)
print(f"Metrics left and right 100 validation samples: (1) {metric_left_100}, (2) {metric_right_100}")
# 2.- dataset
xval_hat_left, xval_hat_right = CVAE(_dataset_)
xval_left = _dataset_[:, 0, :-num_coordinates]
xval_right = _dataset_[:, 1, :-num_coordinates]
metric_left_710 = metric(x=xval_left, 
                         x_hat=xval_hat_left)
metric_right_710 = metric(x=xval_right, 
                         x_hat=xval_hat_right)
print(f"Metrics left and right dataset: (1) {metric_left_710}, (2) {metric_right_710}")


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







