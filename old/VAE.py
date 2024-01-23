import tensorflow as tf
from keras import Model
import numpy as np 

tfkl = tf.keras.layers 
INPUT_SHAPE_ENCODER = (2, 1024)
INPUT_SHAPE_DECODER = (2, 16)

# ---- Kullback–Leibler Divergence ----
def kl_reconstruction_loss(mu, sigma): 
    """ Computes the Kullback-Leibler Divergence (KLD)
    Args:
    mu -- mean
    sigma -- standard deviation

    Returns:
    KLD loss
    """
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    return tf.reduce_mean(kl_loss) * -0.5

tfkl = tf.keras.layers

# sampling layer: VAE-CVAE latent mean and variance -----------------------
class Sampling(tf.keras.layers.Layer): 
    def call(self, inputs):
        """Generates a random sample and combines with the encoder output

        Args:
        inputs -- output tensor from the encoder
        
        Returns:
        `inputs` tensors combined with a random sample
        """

        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim1 = tf.shape(mu)[1]
        dim2 = tf.shape(mu)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim1, dim2))
        z = mu + tf.exp(0.5 * sigma) * epsilon
        return  z
    

# Encoder: --------------------------------------

def encoder_layers(inputs): 

    # Hyper-params: 
    layers = [2, 3, 4, 5]
    num_filters = [128, 128, 32, 16]
    kernel_size = [4, 4, 3, 2]
    latent_dim = 16 

    init_filters = 1024
    init_size = 8
    init_layer = 1
    x = tfkl.Conv1D(filters=init_filters, 
                    kernel_size=init_size, 
                    padding="same", 
                    kernel_regularizer="l2", 
                    name="encoder_conv%d" % init_layer)(inputs)
    # Layers: 
    for l, f, s in zip(layers, num_filters, kernel_size): 
        x = tfkl.Conv1D(filters=f, kernel_size=s, 
                        padding="same", activation="relu", 
                        name="encoder_conv%d" % l)(x)
        x = tfkl.BatchNormalization(name="encoder_bn%d" % (l-1))(x)
        x = tfkl.Dropout(rate=0.25, name="encoder_dout%d" % (l-1))(x)
    
    # Latent variables:
    mu = tfkl.Dense(units=latent_dim)(x)
    sigma = tfkl.Dense(units=latent_dim)(x)
    return mu, sigma 

def encoder(input_shape=INPUT_SHAPE_ENCODER): 
    inputs = tfkl.Input(shape=input_shape)
    mu, sigma = encoder_layers(inputs)
    z = Sampling(name="Sampling")([mu, sigma])
    model = tf.keras.Model(inputs=inputs, outputs=[z, mu, sigma], name="encoder")
    return model

class MLP(Model):
    def __init__(self, 
                 num_layers: int, 
                 units: int, 
                 activation: str, 
                 name: str): 
        super(MLP, self).__init__(name=name)
        self.num_layers = num_layers
        self.units = units
        self.activation = activation

    def call(self, inputs): 
        for i in range(0, self.num_layers): 
            if i == 0: 
                x = tfkl.Dense(units=self.units)(inputs)
            else: 
                x = tfkl.Dense(units=self.units)(x)

            x = tfkl.BatchNormalization()(x)

            if self.activation.lower() == "relu": 
                x = tfkl.ReLU()(x)
            elif self.activation.lower() == "prelu": 
                x = tfkl.PReLU()(x)
            elif self.activation.lower() == "sine": 
                x = tf.sin(x)
        return x

# Build weights: 
init_num_layers = 4
init_units = 32
mlp = MLP(num_layers=init_num_layers, 
        units=init_units, 
        activation="relu", 
        name="MLP")
init_tensor = tf.convert_to_tensor(np.ones(shape=INPUT_SHAPE_DECODER))
init_tensor = init_tensor[tf.newaxis, :, :]
out_tensor = mlp(init_tensor)


def decoder_layers(inputs): 
    # Hyper-params: 
    num_layers = 4
    units = 32
    activation = "relu"
    layers = [1, 2, 3, 4, 5]
    num_filters = [32, 64, 128, 128, 1024]
    kernel_size = [2, 3, 4, 4, 8]

    # Layers: 
    # Init MLP: 
    for i in range(0, num_layers):
        if i == 0: 
            x = tfkl.Dense(units=units)(inputs)
        else: 
            x = tfkl.Dense(units=units)(x)

        x = tfkl.BatchNormalization()(x)

        if activation.lower() == "relu": 
            x = tfkl.ReLU()(x)
        elif activation.lower() == "prelu": 
            x = tfkl.PReLU()(x)
        elif activation.lower() == "sine": 
            x = tf.sin(x)

    for l, f, s in zip(layers, num_filters, kernel_size): 
        x = tfkl.Conv1DTranspose(filters=f, kernel_size=s, 
                                 padding="same", name="decoder_conv%d" % l)(x)
        x = tf.sin(x)
    
    # Channels: 
    x_left = x[:, 0, :]
    x_right = x[:, 1, :]
    return x_left, x_right


def decoder(input_shape=INPUT_SHAPE_DECODER): 
    inputs = tf.keras.layers.Input(shape=input_shape)
    output_left, output_right = decoder_layers(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[output_left, output_right], 
                            name="decoder_model")
    return model


def VAE(encoder, decoder, input_shape=INPUT_SHAPE_ENCODER): 
    inputs = tf.keras.layers.Input(shape=input_shape)
    z, mu, sigma = encoder(inputs)
    x_left, x_right = decoder(z)
    model = tf.keras.Model(inputs=inputs, outputs=[x_left, x_right], 
                           name="VAE")
    loss = kl_reconstruction_loss(mu, sigma)
    model.add_loss(loss)
    return model

def get_models(): 
    enc, dec = encoder(), decoder()
    vae = VAE(encoder=enc, decoder=dec)
    return vae, enc, dec 



vae, enc, dec = get_models()
vae.summary()

        
    
    
    
        

