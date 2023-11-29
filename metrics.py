import numpy as np 
import tensorflow as tf 
import core


# Metrics: -----------------------------------------------------------------------
class LSDMetric(tf.keras.metrics.Metric): 

    def __init__(self, name='log_spectral_distortion_metric', **kwargs):
        super(LSDMetric, self).__init__(name=name, **kwargs)
        self.lsd_error = self.add_weight(name='lsd_error', initializer='zeros')
        self.total_samples = self.add_weight(name='total_ssamples', initializer='zeros')

    def update_state(self, y_true, y_pred):
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        N = self.total_samples

        Y_true = core.metrics_logmag(y_true) 
        Y_pred = core.compute_logmag(y_pred)
        log_sq_error = (1/N) * (Y_true - Y_pred)**2

        self.lsd_error.assign_add(tf.sqrt(tf.reduce_sum(log_sq_error)))

    def result(self):
        return self.lsd_error