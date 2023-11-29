import numpy as np 
import tensorflow as tf
import core
import utils 

from Model1 import get_models, Sampling

num = "0_2"
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

# Data generation: ------------------------------------------------------------------------
phi = 65
theta = 95
vec = core.deegres2radians(phi=phi, 
                           theta=theta)
print(f"source vector: {vec}")
r_s = vec[0]
theta_s = vec[1]
phi_s = vec[2]
p_s = core.tf_float32(np.array([r_s, theta_s, phi_s]))

phi_c1 = 60
theta_c1 = 90
h_left1, h_right1 = utils.extract_hrir(elev=phi_c1, 
                                       azimuth=theta_c1)
h_left_c1 = core.tf_float32(h_left1)
h_right_c1 = core.tf_float32(h_right1)
h_left_c1 = tf.concat([h_left_c1, p_s], axis=-1)[tf.newaxis, tf.newaxis, :]
h_right_c1 = tf.concat([h_right_c1, p_s], axis=-1)[tf.newaxis, tf.newaxis, :]
x1 = tf.concat([h_left_c1, h_right_c1], axis=1)

phi_c2 = 70
theta_c2 = 105
h_left2, h_right2 = utils.extract_hrir(elev=phi_c2, 
                                       azimuth=theta_c2)
h_left_c2 = core.tf_float32(h_left2)
h_right_c2 = core.tf_float32(h_right2)
h_left_c2 = tf.concat([h_left_c2, p_s], axis=-1)[tf.newaxis, tf.newaxis, :]
h_right_c2 = tf.concat([h_right_c2, p_s], axis=-1)[tf.newaxis, tf.newaxis, :]
x2 = tf.concat([h_left_c2, h_right_c2], axis=1)

print("==============================================================================================================")
print(f"Genarting HRIR pair: source1 ({phi}°,{theta}°) ")
print(f"HRIR pair used from angles: phi1 = {phi_c1}°, theta1 = {theta_c1}° & phi2 = {phi_c2}°, theta2 = {theta_c2}° ")
print("==============================================================================================================")

_, mu1, sigma1 = encoder(x1)
_, mu2, sigma2 = encoder(x2)
z1 = Sampling()([mu1, sigma1])
z2 = Sampling()([mu2, sigma2])
z = tf.concat([z1, z2], axis=0)
# utils.tsne_plotting(z_left=z[:, 0, :], 
#                     z_right=z[:, 1, :])

h_gen_left1, h_gen_right1 = CVAE(x1)
utils.plot2signals(x_hat1=h_gen_left1[0], 
                   x_hat2=h_gen_right1[0])

# h_gen_left2, h_gen_right2 = CVAE(x2)
# utils.plot2signals(x_hat1=h_gen_left2[0], 
#                    x_hat2=h_gen_right2[0])








