"""
Added condition to inputs and latent space.
"""
import tensorflow as tf 

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
drates = (1, 2, 4, 8)
# (CHANGE HERE TO TEST LATENT DIM 32 (num1) OR LATENT DIM 64 (num2))
latent_dim = 64

def encoder_layers(inputs, 
                   filters=num_filters, 
                   kernel_size=kernel_size, 
                   drates=drates, 
                   latent_dim=latent_dim): 
    x = tfkl.Conv1D(filters=4*filters, kernel_size=8*kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(inputs)
    x = tfkl.Conv1D(filters=2*filters, kernel_size=4*kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[3])(x)
    x = tfkl.PReLU()(x)
    x = tfkl.Conv1D(filters=2*filters, kernel_size=4*kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(x)
    x = tfkl.Conv1D(filters=filters, kernel_size=2*kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[2])(x)
    x = tfkl.PReLU()(x)
    x = tfkl.Conv1D(filters=filters, kernel_size=2*kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(x)
    x = tfkl.Conv1D(filters=int(filters/2), kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[1])(x)
    x = tfkl.PReLU()(x)
    x = tfkl.Conv1D(filters=int(filters/2), kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(x)
    x = tfkl.Conv1D(filters=int(filters/4), kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[0])(x)
    x = tfkl.PReLU()(x)
    x = tfkl.Conv1D(filters=int(filters/2), kernel_size=kernel_size, 
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


# Decoder: ---------------------------------------------------------------------------------------------------------
def decoder_layers(inputs_left, 
                   inputs_right, 
                   spatial_input,
                   units=latent_dim, 
                   filters=num_filters, 
                   kernel_size=kernel_size):
    x_left = MLP(inputs=inputs_left, units=2*units)
    x_right = MLP(inputs=inputs_right, units=2*units)
    z = MLP(inputs=spatial_input, units=2*units)
    x = tfkl.Concatenate(axis=1)([x_left, x_right])
    reps = x.shape[1]
    z = tf.tile(input=z, multiples=[1,reps,1])
    x = tfkl.Add()([x, z])
    x = MLP(inputs=x, units=2*units)
    x = tfkl.Conv1DTranspose(filters=int(filters/2), kernel_size=int(kernel_size/2), strides=1, 
                             padding="same", kernel_regularizer="l2")(x)
    x = tfkl.Lambda(lambda x: tf.sin(x))(x)
    x = tfkl.Conv1DTranspose(filters=filters, kernel_size=kernel_size, strides=1, 
                             padding="same", kernel_regularizer="l2")(x)
    x = tfkl.Lambda(lambda x: tf.sin(x))(x)
    x = tfkl.Conv1DTranspose(filters=2*filters, kernel_size=2*kernel_size, strides=1, 
                             padding="same", kernel_regularizer="l2")(x)
    x = tfkl.Lambda(lambda x: tf.sin(x))(x)
    x = tfkl.Conv1DTranspose(filters=4*filters, kernel_size=4*kernel_size, strides=1, 
                             padding="same", kernel_regularizer="l2")(x)
    x = tfkl.Lambda(lambda x: tf.sin(x))(x)
    x_left = tfkl.Dense(units=4*filters)(x[:, 0, :])
    x_right = tfkl.Dense(units=4*filters)(x[:, 1, :])
    return x_left, x_right


# (latent dim)
# (CHANGE HERE TO TEST LATENT DIM 32 (num1) OR LATENT DIM 64 (num2))
INPUT_SHAPE_DECODER = (1, 64)
SPATIAL_SHAPE = (1, 3)
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
def VAE(encoder, decoder, input_shape=INPUT_SHAPE_ENCODER): 
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    z, mu, sigma = encoder(inputs)
    c = inputs[:, 0, -num_conditions:]
    c = c[:, tf.newaxis, :]

    z_left = z[:, 0, :]
    z_left = z_left[:, tf.newaxis, :]
    z_right = z[:, 1, :]
    z_right = z_right[:, tf.newaxis, :]
    x_left, x_right = decoder([z_left, z_right, c])

    model = tf.keras.Model(inputs=inputs, outputs=[x_left, x_right], name="VAE")
    loss = weight * kl_reconstruction_loss(mu, sigma)
    model.add_loss(loss)
    return model


def get_models(): 
    enc, dec = encoder(), decoder()
    vae = VAE(encoder=enc, decoder=dec)
    return vae