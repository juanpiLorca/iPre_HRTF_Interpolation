import numpy as np 
import tensorflow as tf
import core

phi_theta = "dataset/tags/hrir_elev_azimuth.npy"
phi_theta = np.load(phi_theta)
phi_theta = phi_theta[:, np.newaxis, :]
phi_theta = np.concatenate([phi_theta, phi_theta], axis=1)
xyz = "dataset/tags/cartesian_coordinates.npy"
xyz = np.load(xyz)
xyz = xyz[:, np.newaxis, :]
xyz = np.concatenate([xyz, xyz], axis=1)
xyz_norm = tf.nn.l2_normalize(xyz) 
spherical = "dataset/tags/radians_spherical.npy"
spherical = np.load(spherical)
spherical = spherical[:, np.newaxis, :]
spherical = np.concatenate([spherical, spherical], axis=1)
spherical_norm = tf.nn.l2_normalize(spherical)


hrir_left = "dataset/hrirs/hrir_left.npy"
hrir_left = np.load(hrir_left)
hrir_left = hrir_left[:, np.newaxis, :]
hrir_right = "dataset/hrirs/hrir_right.npy"
hrir_right = np.load(hrir_right)
hrir_right = hrir_right[:, np.newaxis, :]

num_train = 610
data = np.concatenate([hrir_left, hrir_right], axis=1)
"""
Data grouped like: (710, 2, 517)
    >>> 710: number of impulse responses 
    >>> 2: number of channels
    >>> 517: 512 samples at sample rate 44.1 [kHz], 3 spatial configuration (x, y, z) or (r, theta, phi) and the 
             elevation and azimuthal angles (phi, theta).
"""

# Cartesian spatial configurations: 
data_cartesian = np.concatenate([data, xyz, phi_theta], axis=-1)
data_cartesian = core.tf_float32(data_cartesian)
data_cartesian = tf.random.shuffle(data_cartesian)
dtrain_cartesian = data_cartesian[:num_train]
dval_cartesian = data_cartesian[num_train:]

np.save("trainig_data_cartesian_full.npy", data_cartesian)

# data_cartesian_norm = np.concatenate([data, xyz_norm, phi_theta], axis=-1)
# data_cartesian_norm = core.tf_float32(data_cartesian_norm)
# data_cartesian_norm = tf.random.shuffle(data_cartesian_norm)
# dtrain_cartesian_norm = data_cartesian_norm[:num_train]
# dval_cartesian_norm = data_cartesian_norm[num_train:]

# file_training = "training_cartesian_data{}.npy"
# np.save(file_training, dtrain_cartesian)
# file_validation = "validation_cartesian_data{}.npy"
# np.save(file_validation, dval_cartesian)

# np.save(file_training.format("_norm"), dtrain_cartesian_norm)
# np.save(file_validation.format("_norm"), dval_cartesian_norm)


# # Spherical configurations: 
# data_spherical = np.concatenate([data, spherical, phi_theta], axis=-1)
# data_spherical = core.tf_float32(data_spherical)
# data_spherical = tf.random.shuffle(data_spherical)
# dtrain_spherical = data_spherical[:num_train]
# dval_spherical= data_spherical[num_train:]

# data_spherical_norm = np.concatenate([data, spherical_norm, phi_theta], axis=-1)
# data_spherical_norm = core.tf_float32(data_spherical_norm)
# data_spherical_norm = tf.random.shuffle(data_spherical_norm)
# dtrain_spherical_norm = data_spherical_norm[:num_train]
# dval_spherical_norm = data_spherical_norm[num_train:]

# file_training = "training_spherical_data{}.npy"
# np.save(file_training, dtrain_spherical)
# file_validation = "validation_spherical_data{}.npy"
# np.save(file_validation, dval_spherical)

# np.save(file_training.format("_norm"), dtrain_spherical_norm)
# np.save(file_validation.format("_norm"), dval_spherical_norm)







