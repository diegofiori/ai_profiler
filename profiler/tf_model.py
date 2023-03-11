import numpy as np
import tensorflow as tf


def profile_tf_model(model: tf.keras.Model, input_data: np.ndarray, logdir: str = None):
    if logdir is None:
        logdir = "logdir"
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                       python_tracer_level=1,
                                                       device_tracer_level=1)
    tf.profiler.experimental.start(logdir, options=options)
    out = model.predict(input_data)
    tf.profiler.experimental.stop()
    return logdir
