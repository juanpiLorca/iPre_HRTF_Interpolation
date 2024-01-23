import numpy as np 
import tensorflow as tf
import core
import utils 
import matplotlib.pyplot as plt 
from Model import get_models, Sampling

num = "0_full"
batch = 64
num_epochs = 500
CVAE = get_models()
weights = f"training data/weights/weights_cvae{num}_on_batch{batch}_epochs{num_epochs}.h5"
CVAE.load_weights(weights)

# Encoder extraction: 
def pull_module(model, 
                module_name="encoder"): 
    for layer in model.layers:
        if layer.name == module_name:
            module = layer
    return module

encoder = pull_module(model=CVAE)
decoder = pull_module(model=CVAE, 
                      module_name="decoder_model")

# Data generation: ------------------------------------------------------------------------
# Desired location: (65°, 95°)
phi = 65
theta = 95
p_s = core.spherical2cartesian(phi=phi, 
                               theta=theta)
print(f"Desired location: {p_s}")

phi_c1 = 60
theta_c1 = 90
p_s1 = core.spherical2cartesian(phi=phi_c1, 
                                theta=theta_c1, 
                                tensor=True)
h_left1, h_right1 = utils.extract_hrir(elev=phi_c1, 
                                       azimuth=theta_c1)
h_left_c1 = core.tf_float32(h_left1)
h_right_c1 = core.tf_float32(h_right1)
h_left_c1 = tf.concat([h_left_c1, p_s1], axis=-1)[tf.newaxis, tf.newaxis, :]
h_right_c1 = tf.concat([h_right_c1, p_s1], axis=-1)[tf.newaxis, tf.newaxis, :]
x1 = tf.concat([h_left_c1, h_right_c1], axis=1)

phi_c2 = 70
theta_c2 = 105
p_s2 = core.spherical2cartesian(phi=phi_c2, 
                                theta=theta_c2, 
                                tensor=True)
h_left2, h_right2 = utils.extract_hrir(elev=phi_c2, 
                                       azimuth=theta_c2)
h_left_c2 = core.tf_float32(h_left2)
h_right_c2 = core.tf_float32(h_right2)
h_left_c2 = tf.concat([h_left_c2, p_s2], axis=-1)[tf.newaxis, tf.newaxis, :]
h_right_c2 = tf.concat([h_right_c2, p_s2], axis=-1)[tf.newaxis, tf.newaxis, :]
x2 = tf.concat([h_left_c2, h_right_c2], axis=1)

# Latent z extraction: 
z1, mu1, sigma1 = encoder(x1)
z2, mu2, sigma2 = encoder(x2)

z1_left = z1[:, 0, :][:, tf.newaxis, :]
z2_left = z2[:, 0, :][:, tf.newaxis, :]
z1_right = z2[:, 1, :][:, tf.newaxis, :]
z2_right = z2[:, 1, :][:, tf.newaxis, :]
p_s1 = p_s1[tf.newaxis, tf.newaxis, :]
p_s2 = p_s2[tf.newaxis, tf.newaxis, :]

t = 0
interpolations = []
while t <= 1: 
    # We lineary combine latent spaces and locations:
    p_s = t * p_s1 + (1 - t) * p_s2
    z_left = t * z1_left + (1 - t) * z2_left
    z_right = t * z1_right + (1 - t) * z2_right

    x_left, x_right = decoder([z_left, z_right, p_s])
    interpolations.append([x_left, x_right])

    t += 0.25

# channel: 0 ==> left ; 1 ==> right
ch = 1
x1 = interpolations[0][ch][0]
x2 = interpolations[1][ch][0]
x3 = interpolations[2][ch][0]
x4 = interpolations[3][ch][0]
x5 = interpolations[4][ch][0]



