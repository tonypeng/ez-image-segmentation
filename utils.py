import tensorflow as tf


def tensor_shape_as_list(x: tf.Tensor) -> list:
    return x.get_shape().as_list()