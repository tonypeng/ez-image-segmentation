import numpy as np
import scipy
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
