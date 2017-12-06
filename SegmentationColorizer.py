import numpy as np
import matplotlib
import tensorflow as tf


class SegmentationColorizer:
    def __init__(self, min_value, max_value, cmap_name):
        self.min_value = min_value
        self.max_value = max_value
        cm = matplotlib.cm.get_cmap(cmap_name)
        cm = np.random.permutation(cm.colors)
        self.cmap = tf.constant(cm, dtype=tf.float32)

    def colorize(self, x):
        value = (x - self.min_value) / (self.max_value - self.min_value)
        value = tf.squeeze(value)
        indices = tf.to_int32(tf.round(value * 255))
        return tf.gather(self.cmap, indices)