import numpy as np 
import tensorflow as tf 
import core 


class MatrixHRIR(tf.keras.layers.Layer): 
    """
    Performs cyclic convolution by W.x + b (matrix multiplication).

    In order to do it, we considered the commutativity of the operation:
                        {h * x}(n) = {x * h}(n)
    To not run out of RAM memory: 
        >>> Split the signal or function used for creating the kernel matrix into: 
            (num_frames, num_samples) ==> according to the overlap used. 
        >>> Perform each frame (i) matrix multiplication and add the bias: Wi.xi + bi
    """
    def __init__(self, 
                 units=512, 
                 activation=None, 
                 signal_type="sweep", 
                 name="hrtf_matrix_convolution"):
        super(MatrixHRIR, self).__init__(name=name)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.sgnal_type = signal_type
        self.signal = core.init_signal(signal_type=self.sgnal_type)
        self.num_frames = None

    def build(self, input_shape):
        """
        Weight matrix shape: (num_frames, units, units)
        """
        sframes = core.split_into_frames(signal=self.signal)
        self.num_frames = sframes.shape[0]
        kernel = np.zeros(shape=(self.num_frames, input_shape[-1], self.units))

        for i in range(0, self.num_frames): 
            kernel[i] = core.KernelMatrix(sframes[i])
        
        kernel = core.tf_float32(kernel)
        self.w = tf.Variable(name="kernel",
                            initial_value=kernel,
                            trainable=False)
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
            initial_value=b_init(shape=(self.num_frames, self.units), dtype='float32'),
            trainable=True)
        
        super().build(input_shape)

    def call(self, inputs):
        """
        inputs: impulse response coefficients shaped (num_frames, hrir_length)
        outputs: binaural sweep shaped (num_frames, hrir_length)
            >>> Experiments: 50 % overlapping frames of 0.5 secs of audio ==> (num_frames, hrir_length) = (87, 512)
        """
        assert inputs.shape[-1] == self.units, f"Input shape should be (batch_size, {self.units})"
        output = None
        for i in range(0, self.num_frames): 
            x = tf.linalg.matvec(self.w[i], inputs[i]) + self.b[i]
            if i == 0: 
                output = x[tf.newaxis, :]
            else: 
                output = tf.concat([output, x[tf.newaxis, :]], axis=0)
        return output

