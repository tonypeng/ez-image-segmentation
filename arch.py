import tensorflow as tf
import utils


def get_activation_function(act_str):
    if act_str == 'elu':
        return tf.nn.elu
    elif act_str == 'relu':
        return tf.nn.relu
    raise NotImplementedError


def initialize_weights(shape, stddev: float, weight_decay=None) -> tf.Variable:
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if weight_decay is not None:
        loss_weight_decay = weight_decay * tf.nn.l2_loss(var)
        tf.add_to_collection('weight_regularizers', loss_weight_decay)
    return var


def initialize_biases(shape, value: float) -> tf.Variable:
    return tf.Variable(tf.constant(value, shape=shape))


def conv2d(x: tf.Tensor, ksize, stride, chan_out, border_mode='SAME', init_stddev=1.0, weight_decay=None):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    # no bias needed because we batch normalize
    W = initialize_weights([ksize, ksize, chan_in, chan_out], init_stddev, weight_decay)
    return tf.nn.conv2d(x, W, [1, stride, stride, 1], border_mode)


def deconv2d(x: tf.Tensor, ksize, stride, chan_out, border_mode='SAME', init_stddev=1.0, weight_decay=None):
    x_size = utils.tensor_shape_as_list(x)
    batch_size, height, width, chan_in = x_size
    W = initialize_weights([ksize, ksize, chan_in, chan_out], init_stddev, weight_decay)
    return tf.nn.conv2d_transpose(x, W, [batch_size, height * stride, width * stride, chan_out], [1, stride, stride, 1],
                                  padding=border_mode)


def resizeconv2d(x, W, stride, border_mode='SAME'):
    x_size = utils.tensor_shape_as_list(x)
    resized_height, resized_width = x_size[1] * stride * stride, x_size[2] * stride * stride
    x_resized = tf.image.resize_images(x, (resized_height, resized_width),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    W_T = tf.transpose(W, perm=[0, 1, 3, 2])
    return conv2d(x_resized, W_T, stride, border_mode='SAME')


def avg_pool(x: tf.Tensor, ksize, stride=None, border_mode='SAME'):
    stride = stride or ksize
    return tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=border_mode)


def atrous_conv2d(x: tf.Tensor, ksize, rate, chan_out, border_mode='SAME', init_stddev=1.0, weight_decay=None):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    W = initialize_weights([ksize, ksize, chan_in, chan_out], init_stddev, weight_decay)
    return tf.nn.atrous_conv2d(x, W, rate, border_mode)


def conv2d_bn_activation(x: tf.Tensor, is_training, ksize, stride, chan_out,
                         border_mode='SAME', activation=tf.nn.relu, bn_scale=False, init_stddev=1.0, weight_decay=None):
    outp = conv2d(x, ksize, stride,
                  chan_out=chan_out, border_mode=border_mode, init_stddev=init_stddev, weight_decay=weight_decay)
    outp = batch_norm(outp, is_training, scale=bn_scale)
    outp = activation(outp)
    return outp

def max_pool(x: tf.Tensor, ksize, stride=None, border_mode='SAME'):
    stride = stride or ksize
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=border_mode)


def dense_block(x: tf.Tensor, is_training, num_layers: int, add_features_per_layer=16, activation=tf.nn.relu,
                feature_maps_out=None, init_stddev=1.0, weight_decay=None) -> tf.Tensor:
    output = x
    for i in range(num_layers):
        output = conv2d_bn_activation(x, is_training, 3, 1, add_features_per_layer,
                                      activation=activation, init_stddev=init_stddev, weight_decay=weight_decay)
        if feature_maps_out is not None:
            feature_maps_out.append(output)
        output = tf.concat([x, output], 3)
    return output


def transition_down_block(x: tf.Tensor, is_training, activation=tf.nn.relu, init_stddev=1.0, weight_decay=None):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    outp = conv2d_bn_activation(x, is_training, 1, 1, chan_in,
                                activation=activation, init_stddev=init_stddev, weight_decay=weight_decay)
    outp = max_pool(outp, 2)
    return outp


def transition_up_block(x: tf.Tensor, is_training, activation=tf.nn.relu, init_stddev=1.0, weight_decay=None):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    outp = deconv2d(x, 3, 2, chan_in, init_stddev=init_stddev, weight_decay=weight_decay)
    outp = batch_norm(outp, is_training)
    outp = activation(outp)
    return outp


def fully_connected(x: tf.Tensor, num_outputs, init_weights_stddev=1.0, init_bias_value=0.0, weight_decay=None):
    num_inputs = utils.tensor_shape_as_list(x)[1]
    W = initialize_weights([num_inputs, num_outputs], init_weights_stddev, weight_decay)
    b = initialize_biases([num_outputs], init_bias_value)
    return tf.nn.xw_plus_b(x, W, b)


def batch_norm(x: tf.Tensor, is_training, scale=False):
    return tf.contrib.layers.batch_norm(x, is_training=is_training, scale=scale)