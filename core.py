import numpy as np 
import tensorflow as tf

import scipy.signal 
import scipy.linalg

# num samples head releated impulse response:
hrir_length = 512
# sampling rate: 
fs = 44100
# seconds: 
secs = 0.5

# core: ------------------------------------------------------------------
def tf_float32(array):
  """Converts a numpy array to a tensor float32""" 
  tensor = tf.convert_to_tensor(array)
  tensor = tf.cast(tensor, dtype=tf.float32)
  return tensor

def safe_log(x, eps=1e-5):
  """Avoid taking the log of a non-positive number."""
  safe_x = tf.where(x <= 0.0, eps, x)
  return tf.math.log(safe_x)

def L2_norm(tensor, axs): 
    norm_tensor = tf.linalg.normalize(tensor, ord=2, axis=axs)
    return norm_tensor

def L1_norm(tensor, axs): 
    norm_tensor = tf.linalg.normalize(tensor, ord=1, axis=axs)
    return norm_tensor

def log10(x: tf.Tensor): 
    x = tf.math.log(x)
    x = tf_float32(x)
    return x

# spatial info. : ---------------------------------------------------------
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
        xyz_vect = tf_float32(xyz_vect)
    elif array: 
        xyz_vect = np.array(xyz_vect)
    return xyz_vect

def deegres2radians(phi, theta, r=1.4, tensor=False): 
    phi = phi * np.pi / 180
    theta = theta * np.pi / 180
    radians = np.array([r, theta, phi])
    if tensor: 
        radians = tf_float32(radians)
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

# spectral ops: ---------------------------------------------------------
def stft(signal, frame_size=2048, overlap=0.75, pad_end=True):
    """Differentiable stft in tensorflow, computed in batch."""
    # Remove channel dim if present.
    signal = tf_float32(signal)
    if len(signal.shape) == 3:
        signal = tf.squeeze(signal, axis=0)

    S = tf.signal.stft(
        signals=signal,
        frame_length=int(frame_size),
        frame_step=int(frame_size * (1.0 - overlap)),
        fft_length=None,  # Use enclosing power of 2.
        pad_end=pad_end)
    return S

def stft_phase(signal, frame_size): 
    S = stft(signal=signal, frame_size=frame_size)
    angle = tf.math.angle(S)
    return angle 

def numpy_fft_mag(signal, fs=fs): 
    """
    Computes the fft of a signal using numpy library. 
    Returns the magnitude of the fft and the frequencies of it.
    """
    N = signal.shape[0]
    S = np.abs(np.fft.fft(signal))
    S = tf_float32(S)
    S = S[:S.shape[0] + 1]
    freqs = np.linspace(0, (N-1) * fs / N, N)
    freqs = tf_float32(freqs)
    freqs = freqs[:freqs.shape[0] + 1]
    return S, freqs

def compute_mag(audio, size=2048, overlap=0.75, pad_end=True, ft=False):
    if ft:  
        mag = tf.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))
    else: 
        mag = tf.abs(audio)
    return tf_float32(mag)

def compute_logmag(audio, size=2048, overlap=0.75, pad_end=True, ft=False):
    return safe_log(compute_mag(audio, size, overlap, pad_end, ft))

def metrics_logmag(audio): 
    N = audio.shape[1]
    audio = tf.cast(audio, dtype=tf.complex64)
    X = tf.signal.fft(audio)
    X = X[:, :int(N/2)+1]
    X_mag = tf.abs(X)
    X_logmag = safe_log(X_mag)
    return X_logmag

# Pink Noise: ----------------------------------------------
def gen_pink_noise(N, d=0.5): 
    X_white = np.fft.rfft(np.random.randn(N))
    S = np.fft.rfftfreq(N)
    S_pink = np.zeros_like(S)
    for i in range(0, len(S)):
        f = S[i]
        S_pink[i] = 1/np.where(f == 0, float('inf'), f**d)
    S_pink = S_pink / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S_pink
    noise = np.fft.irfft(X_shaped)
    noise = (np.max(noise) - noise) / (np.max(noise) - np.min(noise))
    return noise, S_pink

# Sine Sweep: -----------------------------------------------
# frequency range: [Hz]
f0 = 20
f1 = 30000

def SineSweep(f0=f0, secs=secs, f1=f1, method='logarithmic'):
    N = (int(secs * fs / hrir_length) + 1) * hrir_length 
    last_sample = N / fs
    t = np.linspace(0, last_sample, N)
    sineSweep = scipy.signal.chirp(t, f0, secs, f1, method=method)
    return sineSweep

def init_signal(secs=secs, fs=fs, hrir_length=hrir_length, 
                signal_type="sweep"): 
    N = (int(secs * fs / hrir_length) + 1) * hrir_length 
    last_sample = N / fs
    t = np.linspace(0, last_sample, N)

    if signal_type == "sweep": 
        # frequency range: [Hz]
        f0 = 20
        f1 = 30000
        signal = SineSweep(f0=f0,
                           secs=secs, 
                           f1=f1, 
                           method="logarithmic")
    
    if signal_type == "noise":
        signal, S = gen_pink_noise(N=N)

    return signal 

# Reshaping: ----------------------------------------------------------------------
def reshape_condition_vector(w, kernel_size, num_filters): 
    """
    Reshapes a condition vector shaped (condition_samples,) into a numpy array
    shaped (kernel_size, num_filters, condition_samples)
    """
    w = w[tf.newaxis, :]
    w_reshaped = tf.tile(tf.expand_dims(w, axis=0), [kernel_size, num_filters, 1])
    w_reshaped = tf_float32(w_reshaped)
    return w_reshaped

# Audio ops: ----------------------------------------------------------------------
def split_into_frames(signal, 
                      frame_size=hrir_length, 
                      hop_size=hrir_length/2, 
                      apply_window=True): 
    """
    50% overlapping frames. 
    """
    frames = []
    N = signal.shape[0]
    num_frames = 1 + int((N - frame_size)//hop_size)
    for k in range(0, num_frames): 
        step = int(k * hop_size)
        signal_frame = signal[step:(frame_size + step)]
        frames.append(signal_frame)
    frames = np.array(frames)
    if apply_window:
        window = np.hamming(M=frame_size)
        frames = frames * window
    return frames

def overlap_and_add(x_frames, hop_size=1.5): 
    if len(x_frames.shape) == 3:
        x_frames = tf.squeeze(x_frames, axis=0)
    samples = x_frames.shape[0]
    hop_size = int(hop_size * samples)
    x = tf.signal.overlap_and_add(x_frames, hop_size)
    return x

# Convolution as Matrix multiplication: -------------------------------------------
def KernelMatrixOp(signal, ir): 
    """
    Args: 
        >>> signal: input signal to apply cyclic convolution 
        >>> ir: impulse response coefficients shaped (N,)
        >>> M: input signal length
        >>> N: kernel lenght
    Returns: 
        >>> W: kernel matrix for Cyclic convolution. 

    More info: https://ccrma.stanford.edu/~jos/fp/Cyclic_Convolution_Matrix.html
    """
    signal = signal.astype(np.float32)
    M = signal.shape[0]
    N = ir.shape[0]
    if M > N: 
        # zero right pad 
        num_padd = (0, M - N) 
        ir = np.pad(ir, pad_width=num_padd, 
                    mode="constant", constant_values=0)

    W = scipy.linalg.circulant(signal)
    W = tf_float32(W)
    y = np.dot(W, ir)
    return y, W

def KernelMatrix(signal): 
    """
    Args: 
        >>> signal: input signal to be used in the convolution. 
    Returns: 
        >>> W: kernel matrix for Cyclic convolution. 

    More info: https://ccrma.stanford.edu/~jos/fp/Cyclic_Convolution_Matrix.html
    """
    signal = tf_float32(signal)
    W = scipy.linalg.circulant(signal)
    W = tf_float32(W)
    return W


# DDSP (Paper: Differential Digital Signal Processing) core functions: ----------------------------------------------------------------------
from typing import Text
from scipy import fftpack

def crop_and_compensate_delay(audio: tf.Tensor, audio_size: int, ir_size: int,
                              padding: Text,
                              delay_compensation: int) -> tf.Tensor:
  """Crop audio output from convolution to compensate for group delay.

  Args:
    audio: Audio after convolution. Tensor of shape [batch, time_steps].
    audio_size: Initial size of the audio before convolution.
    ir_size: Size of the convolving impulse response.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation < 0 it
      defaults to automatically calculating a constant group delay of the
      windowed linear phase filter from frequency_impulse_response().

  Returns:
    Tensor of cropped and shifted audio.

  Raises:
    ValueError: If padding is not either 'valid' or 'same'.
  """
  # Crop the output.
  if padding == 'valid':
    crop_size = ir_size + audio_size - 1
  elif padding == 'same':
    crop_size = audio_size
  else:
    raise ValueError('Padding must be \'valid\' or \'same\', instead '
                     'of {}.'.format(padding))

  # Compensate for the group delay of the filter by trimming the front.
  # For an impulse response produced by frequency_impulse_response(),
  # the group delay is constant because the filter is linear phase.
  total_size = int(audio.shape[-1])
  crop = total_size - crop_size
  start = ((ir_size - 1) // 2 -
           1 if delay_compensation < 0 else delay_compensation)
  end = crop - start
  return audio[:, start:-end]

# Time-varying convolution -----------------------------------------------------
def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
  """Calculate final size for efficient FFT.

  Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.

  Returns:
    fft_size: Size for efficient FFT.
  """
  convolved_frame_size = ir_size + frame_size - 1
  if power_of_2:
    # Next power of 2.
    fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
  else:
    fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
  return fft_size


def crop_and_compensate_delay(audio: tf.Tensor, audio_size: int, ir_size: int,
                              padding: Text,
                              delay_compensation: int) -> tf.Tensor:
  """Crop audio output from convolution to compensate for group delay.

  Args:
    audio: Audio after convolution. Tensor of shape [batch, time_steps].
    audio_size: Initial size of the audio before convolution.
    ir_size: Size of the convolving impulse response.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation < 0 it
      defaults to automatically calculating a constant group delay of the
      windowed linear phase filter from frequency_impulse_response().

  Returns:
    Tensor of cropped and shifted audio.

  Raises:
    ValueError: If padding is not either 'valid' or 'same'.
  """
  # Crop the output.
  if padding == 'valid':
    crop_size = ir_size + audio_size - 1
  elif padding == 'same':
    crop_size = audio_size
  else:
    raise ValueError('Padding must be \'valid\' or \'same\', instead '
                     'of {}.'.format(padding))

  # Compensate for the group delay of the filter by trimming the front.
  # For an impulse response produced by frequency_impulse_response(),
  # the group delay is constant because the filter is linear phase.
  total_size = int(audio.shape[-1])
  crop = total_size - crop_size
  start = ((ir_size - 1) // 2 -
           1 if delay_compensation < 0 else delay_compensation)
  end = crop - start
  return audio[:, start:-end]

def fft_convolve(audio: tf.Tensor,
                 impulse_response: tf.Tensor,
                 padding: Text = 'same',
                 delay_compensation: int = -1) -> tf.Tensor:
  """Filter audio with frames of time-varying impulse responses.

  Time-varying filter. Given audio [batch, n_samples], and a series of impulse
  responses [batch, n_frames, n_impulse_response], splits the audio into frames,
  applies filters, and then overlap-and-adds audio back together.
  Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
  convolution for large impulse response sizes.

  Args:
    audio: Input audio. Tensor of shape [batch, audio_timesteps].
    impulse_response: Finite impulse response to convolve. Can either be a 2-D
      Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
      ir_frames, ir_size]. A 2-D tensor will apply a single linear
      time-invariant filter to the audio. A 3-D Tensor will apply a linear
      time-varying filter. Automatically chops the audio into equally shaped
      blocks to match ir_frames.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation is less
      than 0 it defaults to automatically calculating a constant group delay of
      the windowed linear phase filter from frequency_impulse_response().

  Returns:
    audio_out: Convolved audio. Tensor of shape
        [batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
        [batch, audio_timesteps] ('same' padding).

  Raises:
    ValueError: If audio and impulse response have different batch size.
    ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
      number of impulse response frames is on the order of the audio size and
      not a multiple of the audio size.)
  """
  audio, impulse_response = tf_float32(audio), tf_float32(impulse_response)

  # Get shapes of audio.
  batch_size, audio_size = audio.shape.as_list()

  # Add a frame dimension to impulse response if it doesn't have one.
  ir_shape = impulse_response.shape.as_list()
  if len(ir_shape) == 2:
    impulse_response = impulse_response[:, tf.newaxis, :]

  # Broadcast impulse response.
  if ir_shape[0] == 1 and batch_size > 1:
    impulse_response = tf.tile(impulse_response, [batch_size, 1, 1])

  # Get shapes of impulse response.
  ir_shape = impulse_response.shape.as_list()
  batch_size_ir, n_ir_frames, ir_size = ir_shape

  # Validate that batch sizes match.
  if batch_size != batch_size_ir:
    raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                     'be the same.'.format(batch_size, batch_size_ir))

  # Cut audio into frames.
  frame_size = int(np.ceil(audio_size / n_ir_frames))
  hop_size = frame_size
  audio_frames = tf.signal.frame(audio, frame_size, hop_size, pad_end=True)

  # Check that number of frames match.
  n_audio_frames = int(audio_frames.shape[1])
  if n_audio_frames != n_ir_frames:
    raise ValueError(
        'Number of Audio frames ({}) and impulse response frames ({}) do not '
        'match. For small hop size = ceil(audio_size / n_ir_frames), '
        'number of impulse response frames must be a multiple of the audio '
        'size.'.format(n_audio_frames, n_ir_frames))

  # Pad and FFT the audio and impulse responses.
  fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
  audio_fft = tf.signal.rfft(audio_frames, [fft_size])
  ir_fft = tf.signal.rfft(impulse_response, [fft_size])

  # Multiply the FFTs (same as convolution in time).
  audio_ir_fft = tf.multiply(audio_fft, ir_fft)

  # Take the IFFT to resynthesize audio.
  audio_frames_out = tf.signal.irfft(audio_ir_fft)
  audio_out = tf.signal.overlap_and_add(audio_frames_out, hop_size)

  # Crop and shift the output audio.
  return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding,
                                   delay_compensation)


# Filter Design ----------------------------------------------------------------
def apply_window_to_impulse_response(impulse_response: tf.Tensor,
                                     window_size: int = 0,
                                     causal: bool = False) -> tf.Tensor:
  """Apply a window to an impulse response and put in causal form.

  Args:
    impulse_response: A series of impulse responses frames to window, of shape
      [batch, n_frames, ir_size].
    window_size: Size of the window to apply in the time domain. If window_size
      is less than 1, it defaults to the impulse_response size.
    causal: Impulse responnse input is in causal form (peak in the middle).

  Returns:
    impulse_response: Windowed impulse response in causal form, with last
      dimension cropped to window_size if window_size is greater than 0 and less
      than ir_size.
  """
  impulse_response = tf_float32(impulse_response)

  # If IR is in causal form, put it in zero-phase form.
  if causal:
    impulse_response = tf.signal.fftshift(impulse_response, axes=-1)

  # Get a window for better time/frequency resolution than rectangular.
  # Window defaults to IR size, cannot be bigger.
  ir_size = int(impulse_response.shape[-1])
  if (window_size <= 0) or (window_size > ir_size):
    window_size = ir_size
  window = tf.signal.hann_window(window_size)

  # Zero pad the window and put in in zero-phase form.
  padding = ir_size - window_size
  if padding > 0:
    half_idx = (window_size + 1) // 2
    window = tf.concat([window[half_idx:],
                        tf.zeros([padding]),
                        window[:half_idx]], axis=0)
  else:
    window = tf.signal.fftshift(window, axes=-1)

  # Apply the window, to get new IR (both in zero-phase form).
  window = tf.broadcast_to(window, impulse_response.shape)
  impulse_response = window * tf.math.real(impulse_response)

  # Put IR in causal form and trim zero padding.
  if padding > 0:
    first_half_start = (ir_size - (half_idx - 1)) + 1
    second_half_end = half_idx + 1
    impulse_response = tf.concat([impulse_response[..., first_half_start:],
                                  impulse_response[..., :second_half_end]],
                                 axis=-1)
  else:
    impulse_response = tf.signal.fftshift(impulse_response, axes=-1)

  return impulse_response


def frequency_impulse_response(magnitudes: tf.Tensor,
                               window_size: int = 0) -> tf.Tensor:
  """Get windowed impulse responses using the frequency sampling method.

  Follows the approach in:
  https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html

  Args:
    magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
      n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
      last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
      f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
      audio into equally sized frames to match frames in magnitudes.
    window_size: Size of the window to apply in the time domain. If window_size
      is less than 1, it defaults to the impulse_response size.

  Returns:
    impulse_response: Time-domain FIR filter of shape
      [batch, frames, window_size] or [batch, window_size].

  Raises:
    ValueError: If window size is larger than fft size.
  """
  # Get the IR (zero-phase form).
  magnitudes = tf.complex(magnitudes, tf.zeros_like(magnitudes))
  impulse_response = tf.signal.irfft(magnitudes)

  # Window and put in causal form.
  impulse_response = apply_window_to_impulse_response(impulse_response,
                                                      window_size)

  return impulse_response