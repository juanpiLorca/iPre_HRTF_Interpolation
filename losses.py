import functools
import tensorflow as tf 
import core


# Losses -----------------------------------------------------------------------------------
def mean_difference(target, value, loss_type='L1'):
  """Common loss functions.
  Args:
    target: Target tensor.
    value: Value tensor.
    loss_type: One of 'L1', 'L2', or 'COSINE'.
    weights: A weighting mask for the per-element differences.

  Returns:
    The average loss.

  Raises:
    ValueError: If loss_type is not an allowed value.
  """
  difference = target - value
  loss_type = loss_type.upper()
  if loss_type == 'L1':
    return tf.reduce_mean(tf.abs(difference))
  elif loss_type == 'L2':
    return tf.reduce_mean(difference**2)
  elif loss_type == 'COSINE':
    return tf.compat.v1.losses.cosine_distance(target, value, axis=-1)
  else:
    raise ValueError('Loss type ({}), must be '
                     '"L1", "L2", or "COSINE"'.format(loss_type))
  

class SpectralLoss(tf.keras.losses.Loss): 
    """
    Multi-scale spectrogram loss.

    This loss is the bread-and-butter of comparing two audio signals. It offers
    a range of options to compare spectrograms, many of which are redunant, but
    emphasize different aspects of the signal. By far, the most common comparisons
    are magnitudes (mag_weight) and log magnitudes (logmag_weight).
    """

    def __init__(self, 
                 fft_sizes=(512, 256, 128, 64), 
                 complex_spectogram_weight = 1.0, 
                 logmag_weight = 0.0,
                 loss_type='L1', 
                 name='spectral_loss'):
        
        """Constructor, set loss weights of various components.
        Args:
        fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
            spectrogram has a time-frequency resolution trade-off based on fft size,
            so comparing multiple scales allows multiple resolutions.
        loss_type: One of 'L1', 'L2', or 'COSINE'.
        name: Name of the module.
        """
        super(SpectralLoss, self).__init__(name=name)
        self.fft_sizes = fft_sizes
        self.cpx_spectrogram_weight = complex_spectogram_weight
        self.logmag_weight = logmag_weight
        self.loss_type = loss_type

        self.spectrogram_ops = []
        for size in self.fft_sizes:
            spectrogram_op = functools.partial(core.stft, frame_size=size)
            self.spectrogram_ops.append(spectrogram_op)

    def call(self, target_audio, audio):
        loss = 0.0
        loss_left = 0.0
        loss_right = 0.0
        tgt_left, tgt_right = target_audio
        x_left, x_right = audio
        # Computes dft for each fft size.
        for spect_op in self.spectrogram_ops:
            Tgt_left = spect_op(tgt_left)
            Tgt_right = spect_op(tgt_right)
            X_left = spect_op(x_left)
            X_right = spect_op(x_right)
            # Add chosen loss (L1).
            loss_left +=  self.cpx_spectrogram_weight * mean_difference(Tgt_left, X_left, self.loss_type)
            loss_right +=  self.cpx_spectrogram_weight * mean_difference(Tgt_right, X_right, self.loss_type)
            # Add logmagnitude loss, reusing spectrogram.
            if self.logmag_weight != 0:
              Tgt_left_mag = core.compute_logmag(Tgt_left, ft=False)
              Tgt_right_mag = core.compute_logmag(Tgt_right, ft=False)
              X_left_mag = core.compute_logmag(X_left, ft=False)
              X_right_mag = core.compute_logmag(X_right, ft=False)
              loss_left += self.logmag_weight * mean_difference(Tgt_left_mag, X_left_mag, self.loss_type)
              loss_right += self.logmag_weight * mean_difference(Tgt_right_mag, X_right_mag, self.loss_type)
            loss += (loss_left + loss_right)
        return loss
    

class SignalPhaseLoss(tf.keras.losses.Loss): 
  """
    L2 signal-phase loss: 
  """

  def __init__(self, 
              fft_sizes=(512, 256, 128, 64), 
              logmag_weight = 0.0,
              phase_weight = 1.5,
              loss_type='L1', 
              name='spectral_loss'):
      
      """Constructor, set loss weights of various components.
      Args:
      fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
          spectrogram has a time-frequency resolution trade-off based on fft size,
          so comparing multiple scales allows multiple resolutions.
      loss_type: One of 'L1', 'L2', or 'COSINE'.
      name: Name of the module.
      """
      super(SignalPhaseLoss, self).__init__(name=name)
      self.fft_sizes = fft_sizes
      self.logmag_weight = logmag_weight
      self.phase_weight = phase_weight
      self.loss_type = loss_type

      self.spectrogram_ops = []
      for size in self.fft_sizes:
          spectrogram_op = functools.partial(core.stft_phase, frame_size=size)
          self.spectrogram_ops.append(spectrogram_op)

  def call(self, target_audio, audio):
      loss = 0.0
      loss_left = 0.0
      loss_right = 0.0
      y_left, y_right = target_audio
      x_left, x_right = audio
      # Overlap-add 50% overlapped frames:
      tgt_left, tgt_right = core.overlap_and_add(y_left, hop_size=1.5), core.overlap_and_add(y_right, hop_size=1.5)
      xh_left, xh_right = core.overlap_and_add(x_left, hop_size=1.5), core.overlap_and_add(x_right, hop_size=1.5)
      # Computes dft for each fft size ==> phase.
      for spect_op in self.spectrogram_ops:
          Tgt_left = spect_op(tgt_left)
          Tgt_right = spect_op(tgt_right)
          X_left = spect_op(xh_left)
          X_right = spect_op(xh_right)
          # Add chosen loss (L2).
          loss_left +=  self.phase_weight * mean_difference(Tgt_left, X_left, self.loss_type)
          loss_right +=  self.phase_weight * mean_difference(Tgt_right, X_right, self.loss_type)

          # Add pointwise signal loss: 
          loss_left += mean_difference(y_left, x_left, self.loss_type)
          loss_right += mean_difference(y_right, x_right, self.loss_type)
          
          loss += (loss_left + loss_right)
      return loss

