"""
Added condition to inputs and latent space.
"""
import tensorflow as tf 

tfkl = tf.keras.layers
    
# Encoder: ---------------------------------------------------------------------------------------------------------
num_filters = 128 
kernel_size = 4
drates = (2, 4, 8, 16)
latent_dim = 32

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

# (MODEL1:)
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
    x = tfkl.ReLU()(x)
    x = tfkl.Conv1D(filters=2*filters, kernel_size=4*kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(x)
    x = tfkl.Conv1D(filters=filters, kernel_size=2*kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[2])(x)
    x = tfkl.ReLU()(x)
    x = tfkl.Conv1D(filters=filters, kernel_size=2*kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(x)
    x = tfkl.Conv1D(filters=int(filters/2), kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[1])(x)
    x = tfkl.ReLU()(x)
    x = tfkl.Conv1D(filters=int(filters/2), kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(x)
    x = tfkl.Conv1D(filters=int(filters/2), kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    dilation_rate=drates[0])(x)
    x = tfkl.ReLU()(x)
    z = tfkl.Conv1D(filters=latent_dim, kernel_size=kernel_size, 
                    strides=1, padding="same", 
                    kernel_regularizer="l2")(x)
    return z

# (MODEL2:)
# def encoder_layers(inputs, 
#                    filters=num_filters, 
#                    kernel_size=kernel_size, 
#                    drates=drates, 
#                    latent_dim=latent_dim): 
#     x = tfkl.Conv1D(filters=4*filters, kernel_size=8*kernel_size, 
#                     strides=1, padding="same", 
#                     kernel_regularizer="l2")(inputs)
#     x = tfkl.Conv1D(filters=2*filters, kernel_size=4*kernel_size, 
#                     strides=1, padding="same", 
#                     dilation_rate=drates[3])(x)
#     x = tfkl.PReLU()(x)
#     x = tfkl.Conv1D(filters=2*filters, kernel_size=4*kernel_size, 
#                     strides=1, padding="same", 
#                     kernel_regularizer="l2")(x)
#     x = tfkl.Conv1D(filters=filters, kernel_size=2*kernel_size, 
#                     strides=1, padding="same", 
#                     dilation_rate=drates[2])(x)
#     x = tfkl.PReLU()(x)
#     x = tfkl.Conv1D(filters=filters, kernel_size=2*kernel_size, 
#                     strides=1, padding="same", 
#                     kernel_regularizer="l2")(x)
#     x = tfkl.Conv1D(filters=int(filters/2), kernel_size=kernel_size, 
#                     strides=1, padding="same", 
#                     dilation_rate=drates[1])(x)
#     x = tfkl.PReLU()(x)
#     x = tfkl.Conv1D(filters=int(filters/2), kernel_size=kernel_size, 
#                     strides=1, padding="same", 
#                     kernel_regularizer="l2")(x)
#     x = tfkl.Conv1D(filters=int(filters/4), kernel_size=kernel_size, 
#                     strides=1, padding="same", 
#                     dilation_rate=drates[0])(x)
#     x = tfkl.PReLU()(x)
#     z = tfkl.Conv1D(filters=latent_dim, kernel_size=kernel_size, 
#                     strides=1, padding="same", 
#                     kernel_regularizer="l2")(x)
#     return z


# 515 = 512 (hrir samples) + 3 (spatial coordinates)
INPUT_SHAPE_ENCODER = (2, 515)
def encoder(input_shape=INPUT_SHAPE_ENCODER): 
    inputs = tfkl.Input(shape=input_shape)
    z = encoder_layers(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=z, name="encoder")
    return model
    


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
    x = MLP(inputs=x, units=filters)
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


# 64 (latent dim)
INPUT_SHAPE_DECODER = (1, 32)
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
def AE(encoder, decoder, input_shape=INPUT_SHAPE_ENCODER): 
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    z = encoder(inputs)
    c = inputs[:, 0, -num_conditions:]
    c = c[:, tf.newaxis, :]

    z_left = z[:, 0, :]
    z_left = z_left[:, tf.newaxis, :]
    z_right = z[:, 1, :]
    z_right = z_right[:, tf.newaxis, :]
    x_left, x_right = decoder([z_left, z_right, c])

    model = tf.keras.Model(inputs=inputs, outputs=[x_left, x_right], name="AE")
    return model


def get_models(): 
    enc, dec = encoder(), decoder()
    ae = AE(encoder=enc, decoder=dec)
    return ae