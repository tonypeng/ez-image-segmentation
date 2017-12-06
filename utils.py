import numpy as np
import os
import scipy
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf


def tensor_shape_as_list(x: tf.Tensor) -> list:
    return x.get_shape().as_list()


def read_image(path, mode='RGB', size=None):
    img = scipy.misc.imread(path, mode=mode)
    if size is not None:
        img = scipy.misc.imresize(img, size)
    return img


def write_image(img, path, rescale=True):
    if rescale:
        img = img*255.
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def colorize(value, vmin=None, vmax=None, cmap=None):
    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = tf.subtract(value, vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    np.random.seed(42)
    cm = np.random.permutation(cm.colors)
    colors = tf.constant(cm, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value


def get_preview_file_path(root, prefix, suffix, ext):
    return os.path.join(root, prefix + '_' + suffix + '.' + ext)