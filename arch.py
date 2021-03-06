import math
import tensorflow as tf
import utils


def get_activation_function(act_str):
    if act_str == 'elu':
        return tf.nn.elu
    elif act_str == 'relu':
        return tf.nn.relu
    raise NotImplementedError


def make_initializer_he_truncated(fan_in):
    stddev = math.sqrt(2 / fan_in)
    return lambda shape: tf.truncated_normal(shape, stddev=stddev)


def make_initializer_truncated_given_stddev(stddev):
    return lambda shape: tf.truncated_normal(shape, stddev=stddev)


def make_initializer_function(initializer_descriptor, fan_in):
    initializer_name = initializer_descriptor[0]
    if initializer_name == 'he_truncated':
        return make_initializer_he_truncated(fan_in)
    elif initializer_name == 'truncated_given_stddev':
        return make_initializer_truncated_given_stddev(initializer_descriptor[1])
    raise NotImplementedError


def initialize_weights(shape, initializer, weight_decay=None) -> tf.Variable:
    var = tf.Variable(initializer(shape))
    if weight_decay is not None:
        loss_weight_decay = weight_decay * tf.nn.l2_loss(var)
        tf.add_to_collection('weight_regularizers', loss_weight_decay)
    return var


def initialize_biases(shape, value: float) -> tf.Variable:
    return tf.Variable(tf.constant(value, shape=shape))


def conv2d(x: tf.Tensor, ksize, stride, chan_out, initializer_descriptor, border_mode='SAME', weight_decay=None):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    initializer = make_initializer_function(initializer_descriptor, ksize * ksize * chan_in)
    # no bias needed because we batch normalize
    W = initialize_weights([ksize, ksize, chan_in, chan_out], initializer, weight_decay)
    return tf.nn.conv2d(x, W, [1, stride, stride, 1], border_mode)


def deconv2d(x: tf.Tensor, ksize, stride, chan_out, initializer_descriptor, border_mode='SAME', weight_decay=None):
    x_size = utils.tensor_shape_as_list(x)
    _, height, width, chan_in = x_size
    batch_size = tf.shape(x)[0]
    initializer = make_initializer_function(initializer_descriptor, ksize * ksize * chan_in)
    W = initialize_weights([ksize, ksize, chan_out, chan_in], initializer, weight_decay)
    output_shape = tf.stack([batch_size, height * stride, width * stride, chan_out])
    return tf.nn.conv2d_transpose(x, W, output_shape, [1, stride, stride, 1],
                                  padding=border_mode)


def resizeconv2d(x, ksize, stride, chan_out, initializer_descriptor, border_mode='SAME', weight_decay=None):
    x_size = utils.tensor_shape_as_list(x)
    resized_height, resized_width = x_size[1] * stride * stride, x_size[2] * stride * stride
    x_resized = tf.image.resize_images(x, (resized_height, resized_width),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return conv2d(x_resized, ksize, stride, chan_out, initializer_descriptor, border_mode='SAME', weight_decay=weight_decay)


def avg_pool(x: tf.Tensor, ksize, stride=None, border_mode='SAME'):
    stride = stride or ksize
    return tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=border_mode)


def atrous_conv2d(x: tf.Tensor, ksize, rate, chan_out, initializer_descriptor, border_mode='SAME', weight_decay=None):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    initializer = make_initializer_function(initializer_descriptor, ksize * ksize * chan_in)
    W = initialize_weights([ksize, ksize, chan_in, chan_out], initializer, weight_decay)
    return tf.nn.atrous_conv2d(x, W, rate, border_mode)


def conv2d_bn_activation(x: tf.Tensor, is_training, ksize, stride, chan_out, initializer_descriptor,
                         border_mode='SAME', activation=tf.nn.relu, bn_scale=False, weight_decay=None,
                         dropout_keep_prob=None):

    outp = conv2d(x, ksize, stride, chan_out, initializer_descriptor,
                  border_mode=border_mode, weight_decay=weight_decay)
    outp = batch_norm(outp, is_training, scale=bn_scale)
    outp = activation(outp)
    if dropout_keep_prob is not None:
        outp = dropout(outp, dropout_keep_prob)
    return outp


def conv2d_bn_activation_atrous(x: tf.Tensor, is_training, ksize, stride, chan_out, initializer_descriptor,
                         border_mode='SAME', activation=tf.nn.relu, bn_scale=False, weight_decay=None,
                         dropout_keep_prob=None):

    outp = atrous_conv2d(x, ksize, stride, chan_out, initializer_descriptor,
                  border_mode=border_mode, weight_decay=weight_decay)
    outp = batch_norm(outp, is_training, scale=bn_scale)
    outp = activation(outp)
    if dropout_keep_prob is not None:
        outp = dropout(outp, dropout_keep_prob)
    return outp


def max_pool(x: tf.Tensor, ksize, stride=None, border_mode='SAME'):
    stride = stride or ksize
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=border_mode)


def dense_block(x: tf.Tensor, is_training, num_layers: int, initializer_descriptor,
                add_features_per_layer=16, activation=tf.nn.relu, feature_maps_out=None, weight_decay=None,
                dropout_keep_prob=None) -> tf.Tensor:
    output = x
    for i in range(num_layers):
        features = conv2d_bn_activation(output, is_training, 3, 1, add_features_per_layer, initializer_descriptor,
                                      activation=activation, weight_decay=weight_decay,
                                      dropout_keep_prob=dropout_keep_prob)
        if feature_maps_out is not None:
            feature_maps_out.append(features)
        output = tf.concat([output, features], 3)
    return output

def dense_block_atrous(x: tf.Tensor, is_training, num_layers: int, initializer_descriptor,
                add_features_per_layer=16, activation=tf.nn.relu, feature_maps_out=None, weight_decay=None,
                dropout_keep_prob=None) -> tf.Tensor:
    output = x
    for i in range(num_layers):
        features = conv2d_bn_activation_atrous(output, is_training, 3, 1, add_features_per_layer, initializer_descriptor,
                                      activation=activation, weight_decay=weight_decay,
                                      dropout_keep_prob=dropout_keep_prob)
        if feature_maps_out is not None:
            feature_maps_out.append(features)
        output = tf.concat([output, features], 3)
    return output

def transition_down_block(x: tf.Tensor, is_training, initializer_descriptor,
                          activation=tf.nn.relu, weight_decay=None, dropout_keep_prob=None):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    outp = conv2d_bn_activation(x, is_training, 1, 1, chan_in, initializer_descriptor,
                                activation=activation, weight_decay=weight_decay,
                                dropout_keep_prob=dropout_keep_prob)
    outp = max_pool(outp, 2)
    return outp

def transition_down_block_strided(x: tf.Tensor, is_training, initializer_descriptor,
                          activation=tf.nn.relu, weight_decay=None, dropout_keep_prob=None):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    outp = conv2d_bn_activation(x, is_training, 1, 1, chan_in, initializer_descriptor,
                                activation=activation, weight_decay=weight_decay,
                                dropout_keep_prob=dropout_keep_prob)
    outp = conv2d(outp, 3, 2, chan_in, initializer_descriptor, border_mode='SAME')
    return outp

def transition_up_block(x: tf.Tensor, is_training, initializer_descriptor,
                        activation=tf.nn.relu, weight_decay=None, dropout_keep_prob=None):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    outp = deconv2d(x, 3, 2, chan_in, initializer_descriptor, weight_decay=weight_decay)
    outp = batch_norm(outp, is_training)
    outp = activation(outp)
    # if dropout_keep_prob is not None:
    #     outp = dropout(outp, dropout_keep_prob)
    return outp


def transition_up_block_resize(x: tf.Tensor, is_training, initializer_descriptor,
                        activation=tf.nn.relu, weight_decay=None, dropout_keep_prob=None):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    outp = resizeconv2d(x, 3, 2, chan_in, initializer_descriptor, weight_decay=weight_decay)
    outp = batch_norm(outp, is_training)
    outp = activation(outp)
    # if dropout_keep_prob is not None:
    #     outp = dropout(outp, dropout_keep_prob)
    return outp


def batch_norm(x: tf.Tensor, is_training, scale=False):
    return tf.contrib.layers.batch_norm(x, is_training=is_training, scale=scale)


def dropout(x: tf.Tensor, keep_prob: float) -> tf.Tensor:
    return tf.nn.dropout(x, keep_prob)
