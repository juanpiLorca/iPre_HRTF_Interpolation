"""
Added condition to inputs and latent space.
"""
import tensorflow as tf 
from HyperConv import HyperConvBlock

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

# Encoder: ---------------------------------------------------------------------------------------------------------
num_filters = 128 
kernel_size = 4
drates = (2, 4, 8, 16)
# (CHANGE HERE TO TEST LATENT DIM 32 (num1) OR LATENT DIM 64 (num2))
latent_dim = 32

def encoder_layers(inputs, 
                   filters=num_filters, 
                   kernel_size=kernel_size, 
                   drates=drates, 
                   latent_dim=latent_dim): 
    x = tfkl.Conv1D(filters=4*filters, kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(inputs)
    x = tfkl.Conv1D(filters=4*filters, kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[3])(x)
    x = tfkl.PReLU()(x)
    x = tfkl.Conv1D(filters=2*filters, kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(x)
    x = tfkl.Conv1D(filters=2*filters, kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[2])(x)
    x = tfkl.PReLU()(x)
    x = tfkl.Conv1D(filters=filters, kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(x)
    x = tfkl.Conv1D(filters=filters, kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[1])(x)
    x = tfkl.PReLU()(x)
    x = tfkl.Conv1D(filters=int(filters/2), kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(x)
    x = tfkl.Conv1D(filters=int(filters/2), kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[0])(x)
    x = tfkl.PReLU()(x)
    x = tfkl.Conv1D(filters=int(filters/4), kernel_size=kernel_size, 
                            strides=1, padding="same", 
                            kernel_regularizer="l2")(x)
    x = tfkl.Conv1D(filters=int(filters/4), kernel_size=kernel_size, 
                            strides=1, padding="same", 
                            kernel_regularizer="l2")(x)
    mu = tfkl.Dense(units=latent_dim)(x)
    sigma = tfkl.Dense(units=latent_dim)(x)
    return mu, sigma


# 515 = 512 (hrir samples) + 3 (spatial coordinates)
INPUT_SHAPE_ENCODER = (2, 515)
def encoder(input_shape=INPUT_SHAPE_ENCODER): 
    inputs = tfkl.Input(shape=input_shape)
    mu, sigma = encoder_layers(inputs)
    z = Sampling(name="Sampling")([mu, sigma])
    model = tf.keras.Model(inputs=inputs, outputs=[z, mu, sigma], name="encoder")
    return model

    
# Temporal convolutions: 
dilations = (2, 4, 8, 16)
def TCN_block(inputs, 
              spatial_input, 
              num,
              filters=num_filters, 
              kernel_size=kernel_size, 
              dilations=dilations): 
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[0], 
                       num=f"{num}0")([inputs,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[1], 
                       num=f"{num}1")([x,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[2], 
                       num=f"{num}2")([x,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[3], 
                       num=f"{num}3")([x,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[0], 
                       num=f"{num}4")([x,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[1], 
                       num=f"{num}5")([x,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[2], 
                       num=f"{num}6")([x,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[3], 
                       num=f"{num}7")([x,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[0], 
                       num=f"{num}8")([x,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[1], 
                       num=f"{num}9")([x,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[2], 
                       num=f"{num}10")([x,spatial_input])
    x = HyperConvBlock(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation=dilations[3], 
                       num=f"{num}11")([x,spatial_input])
    return x
    

# Decoder: ---------------------------------------------------------------------------------------------------------
def MLP(inputs, units): 
    x = tf.keras.layers.Dense(units=units)(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def decoder_layers(inputs_left, 
                   inputs_right, 
                   spatial_input,
                   filters=num_filters, 
                   kernel_size=kernel_size):
    x_left = MLP(inputs=inputs_left, units=128)
    x_right = MLP(inputs=inputs_right, units=128)
    x_left = TCN_block(x_left, 
                       spatial_input, 
                       num="_left_")
    x_right = TCN_block(x_right, 
                        spatial_input, 
                        num="_right_")
    x = tfkl.Concatenate(axis=1)([x_left, x_right])
    x = tfkl.PReLU()(x)
    x = tfkl.Conv1DTranspose(filters=filters, kernel_size=kernel_size, strides=1, 
                             padding="same", kernel_regularizer="l2")(x)
    x = tfkl.Conv1DTranspose(filters=2*filters, kernel_size=kernel_size, strides=1, 
                             padding="same", kernel_regularizer="l2")(x)
    x = tfkl.Conv1DTranspose(filters=4*filters, kernel_size=4*kernel_size, strides=1, 
                             padding="same", kernel_regularizer="l2")(x)
    x_left = x[:, 0, :]
    x_right = x[:, 1, :]
    x_left = tfkl.Dense(units=4*filters)(x_left)
    x_right = tfkl.Dense(units=4*filters)(x_right)
    return x_left, x_right


# 32 (latent dim)
# (CHANGE HERE TO TEST LATENT DIM 35 (num1) OR LATENT DIM 67 (num2))
INPUT_SHAPE_DECODER = (1, 35)
# To fit adaptor's shape: 
SPATIAL_SHAPE = (134, 3) 
def decoder(input_shape=INPUT_SHAPE_DECODER, spatial_shape=SPATIAL_SHAPE): 
    inputs_left = tf.keras.layers.Input(shape=input_shape)
    inputs_right = tf.keras.layers.Input(shape=input_shape)
    spatial_input = tf.keras.layers.Input(shape=spatial_shape)
    output_left, output_right = decoder_layers(inputs_left, 
                                               inputs_right, 
                                               spatial_input)
    model = tf.keras.Model(inputs=[inputs_left, 
                                   inputs_right, 
                                   spatial_input], 
                            outputs=[output_left, 
                                    output_right], 
                            name="decoder_model")
    return model


# VAE Model: 
num_conditions = 3
weight = 1e2
c_2dim = 134
def VAE(encoder, decoder, input_shape=INPUT_SHAPE_ENCODER): 
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    z, mu, sigma = encoder(inputs)
    c = inputs[:, 0, -num_conditions:]
    c = c[:, tf.newaxis, :]

    z_left = z[:, 0, :]
    z_left = tf.concat([z_left[:, tf.newaxis, :], c], axis=-1)
    z_right = z[:, 1, :]
    z_right = tf.concat([z_right[:, tf.newaxis, :], c], axis=-1)

    c = tf.tile(c, multiples=[1,c_2dim,1])
    x_left, x_right = decoder([z_left, z_right, c])

    model = tf.keras.Model(inputs=inputs, outputs=[x_left, x_right], name="VAE")
    loss = weight * kl_reconstruction_loss(mu, sigma)
    model.add_loss(loss)
    return model


def get_models(): 
    enc, dec = encoder(), decoder()
    vae = VAE(encoder=enc, decoder=dec)
    return vae