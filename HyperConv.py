import numpy as np
import tensorflow as tf


tfk = tf.keras
class HyperConv(tf.keras.Model): 
    """
    HyperConv implements a temporal convolution that has different 
    convolution weights for each time step.
    """
    def __init__(self,
                 filters,
                 kernel_size, 
                 dilation, 
                 num=None): 
        super(HyperConv, self).__init__(name=f"HyperConv{num}")
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation = dilation 
        random_init = tf.random_uniform_initializer(minval=-np.sqrt(6.0/self.filters), 
                                                    maxval=np.sqrt(6.0/self.filters))
        self.weight_adaptor = tfk.Sequential([ 
            tfk.layers.Conv1D(filters=self.filters, 
                              kernel_size=self.kernel_size, 
                              strides=1, 
                              kernel_initializer=random_init,
                              bias_initializer=tf.zeros_initializer()), 
            tfk.layers.ReLU(), 
            tfk.layers.Conv1D(filters=self.filters, 
                              kernel_size=self.kernel_size, 
                              strides=1, 
                              kernel_initializer=random_init,
                              bias_initializer=tf.zeros_initializer())
        ])
        self.bias_adaptor = tfk.Sequential([ 
            tfk.layers.Conv1D(filters=self.filters, 
                              kernel_size=self.kernel_size, 
                              strides=1, 
                              kernel_initializer=random_init, 
                              bias_initializer=tf.zeros_initializer()), 
            tfk.layers.ReLU(), 
            tfk.layers.Conv1D(filters=self.filters, 
                              kernel_size=self.kernel_size, 
                              strides=1, 
                              kernel_initializer=random_init, 
                              bias_initializer=tf.zeros_initializer())
        ])

    def call(self, inputs): 
        """
        Args: 
            >>> inputs: [x, z]
            :params input x: input to the masking layers: (B, ch_in, T)
            :params input z: input to the weight generator layers: (B, z_dim, K) 
        """
        x, z = inputs
        # For having the same output in the axis=1
        m = x.shape[1]
        padding = self.dilation * (self.kernel_size - 1)
        x = tf.pad(x, [[0,0], [0,0], [0,padding]])
        # linearize input by appending receptive field in channels
        start, end = padding, x.shape[-1]
        x = tf.concat([x[:, :, start-i*self.dilation:end-i*self.dilation] 
                       for i in range(self.kernel_size)], axis=1)
        # compute weights and bias
        # weights:
        weights = self.weight_adaptor(z)
        # print("weights_shape:", weights.shape)
        # bias: 
        bias = self.bias_adaptor(z)
        # compute result of dynamic convolution
        y = tf.linalg.matmul(x, weights)
        y = y[:, :m, :] + bias[:, :m, :]
        return y

class HyperConvBlock(tf.keras.Model):
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 dilation, 
                 strides=1,
                 num=None, 
                 skip_connection=False):
        super(HyperConvBlock, self).__init__(name=f"HyperConvBlock{num}")
        self.filters = filters 
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.skip_connection = skip_connection
        self.conv = HyperConv(filters=filters, 
                              kernel_size=kernel_size, 
                              dilation=dilation,
                              num=num) 
        self.residual = tfk.layers.Conv1D(filters=filters, 
                                          kernel_size=1, 
                                          strides=strides)
        self.skip = tfk.layers.Conv1D(filters=filters, 
                                      kernel_size=1, 
                                      strides=strides)

    def call(self, inputs):
        """
        Args: 
            >>> inputs: [x, z]
            :params input x: input to the masking layers: (B, ch_in, T)
            :params input z: input to the weight generator layers: (B, z_dim, K) 
        """
        x, z = inputs
        y = self.conv([x,z])
        y = tf.sin(y)
        # residual and skip
        residual = self.residual(y)
        # print("residual_shpe:", residual.shape)
        if self.skip_connection:
            skip = self.skip(y)
            return skip, z
        else:
            return (residual + x) / 2

    def receptive_field(self):
        return (self.kernel_size - 1) * self.dilation + 1


# conv = tfk.layers.Conv1D(128, 2, 1)
# x = conv(tf.random.normal(shape=(1,2,128)))
# print(x.shape)
# print(conv.weights[0].shape)

# dec = HyperConvBlock(128, 2, 16, 1)
# tensor2 = tf.random.normal(shape=(34,1,128))
# c = tf.random.normal(shape=(34,130,3))
# y = dec([tensor2, c])
# print(y.shape)