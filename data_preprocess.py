import numpy as np 
import core

# SPATIAL COORDINATES & INPUT NETWORK: ---------------------------------------------------------
def spherical2cartesian(phi, theta, r=1.4, tensor=False, array=False): 
    """
    Radius (r) measurement extracted from: KEMAR Manikin Measurement.
        >>> phi in [0°, 130°]: 0° == 90°; 130° == -40°
        >>> theta in [0°, 360°]
    """
    phi = (90 - phi)
    phi = phi * np.pi / 180
    theta = theta * np.pi / 180 
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    xyz_vect = [x, y, z]
    if tensor: 
        xyz_vect = core.tf_float32(xyz_vect)
    elif array: 
        xyz_vect = np.array(xyz_vect)
    return xyz_vect

def deegres2radians(phi, theta, r=1.4, tensor=False): 
    phi = phi * np.pi / 180
    theta = theta * np.pi / 180
    radians = np.array([r, theta, phi])
    if tensor: 
        radians = core.tf_float32(radians)
    return radians

def cartesian2spherical(xyz_array): 
    x = xyz_array[0]
    y = xyz_array[1]
    z = xyz_array[2]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(r, z) * 180/np.pi
    theta = np.arctan2(y, x) * 180/np.pi
    return int(round(phi)), int(round(theta))

def paramterize_by_channel(dataset, dataset_sph, alpha=0.076): 
    """
    Args: 
        - dataset: (x,y,z) position of audio source measured from the center of the head of the 
            KEMAR'S manikin.
        - dataste_sph: (r, theta, phi) position of audio source measured from the center of the head of the 
            KEMAR'S manikin.
        - alpha: distance from the center of the manikin head to each of the ears (left & right)
    Returns: 
        - updated positions for each channel: 
            * right = (x-alpha, y, z) for [0°,180°] and (x+alpha, y, z) for [180°,360°]
            * left = (x+alpha, y, z) for [0°,180°] and (x-alpha, y, z) for [180°,360°]
    """
    samples = len(dataset)
    dataset_pos_right_ear = []
    dataset_pos_left_ear = []
    for i in range(0, samples): 
        theta = dataset_sph[i, 1]
        theta = round(theta * 180 / np.pi)
        x = dataset[i, 0]
        y = dataset[i, 1]
        z = dataset[i, 2]

        if theta > 180: 
            x_right = x + alpha
            x_left = x - alpha
        else: 
            x_right = x - alpha
            x_left = x + alpha

        new_pos_right = [x_right, y, z]
        new_pos_left = [x_left, y, z]
        dataset_pos_right_ear.append(new_pos_right)
        dataset_pos_left_ear.append(new_pos_left)
    dataset_pos_right_ear = np.array(dataset_pos_right_ear)
    dataset_pos_left_ear = np.array(dataset_pos_left_ear)
    np.save("xyz_right_ear", dataset_pos_right_ear)
    np.save("xyz_left_ear", dataset_pos_left_ear)


