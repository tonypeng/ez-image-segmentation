import numpy as np
import tensorflow as tf

from arch import *
from Trainer import *
from TrainerOptions import *

_CONV_WEIGHT_STD_DEV = 0.1


# The Huge Image-segmentation Convolutional Classifier: A Dense Net for Per-Pixel Semantic Classification
def ThiccNet(x: tf.Tensor, is_training, opt: TrainerOptions) -> tf.Tensor:
    pass


def Tiramisu103(x: tf.Tensor, is_training, dropout_keep_prob, opt: TrainerOptions) -> tf.Tensor:
    activation_func = get_activation_function(opt.arch_activation)
    initializer_descriptor = _create_initializer_descriptor(opt)
    dense_block_layer_counts = list(map(int, opt.arch_dense_block_layer_counts.split(',')))

    # 3x3 conv
    conv1 = conv2d_bn_activation(x, is_training, 3, 1, opt.arch_first_conv_features, initializer_descriptor,
                                 activation=activation_func,
                                 weight_decay=opt.opt_weight_decay,
                                 dropout_keep_prob=dropout_keep_prob)

    # downsampling
    shortcuts = [None for i in range(len(dense_block_layer_counts))]
    downsample_blocks = conv1
    for i in range(len(dense_block_layer_counts)):
        downsample_blocks = dense_block(downsample_blocks, is_training, dense_block_layer_counts[i], initializer_descriptor,
                                        add_features_per_layer=opt.arch_dense_block_add_features_per_layer,
                                        activation=activation_func,
                                        weight_decay=opt.opt_weight_decay,
                                        dropout_keep_prob=dropout_keep_prob)
        shortcuts[i] = downsample_blocks
        downsample_blocks = transition_down_block(downsample_blocks, is_training, initializer_descriptor,
                                                  activation=activation_func,
                                                  weight_decay=opt.opt_weight_decay,
                                                  dropout_keep_prob=dropout_keep_prob)

    # reverse to make things easier on the upsampling
    shortcuts = shortcuts[::-1]
    dense_block_layer_counts = dense_block_layer_counts[::-1]

    # bottleneck
    prev_block_feature_maps = []
    # we don't care about the return value (all concatenated feature maps); just the feature maps from this
    # individual block (this prevents the number of feature maps from blowing up in size)
    dense_block(downsample_blocks, is_training, opt.arch_bottleneck_layer_count, initializer_descriptor,
                add_features_per_layer=opt.arch_dense_block_add_features_per_layer,
                activation=activation_func,
                feature_maps_out=prev_block_feature_maps,
                weight_decay=opt.opt_weight_decay,
                dropout_keep_prob=dropout_keep_prob)

    # upsampling
    upsampled_output = None
    for i in range(len(dense_block_layer_counts)):
        upsampled = transition_up_block(tf.concat(prev_block_feature_maps, 3), is_training, initializer_descriptor,
                                        activation=activation_func,
                                        weight_decay=opt.opt_weight_decay,
                                        dropout_keep_prob=dropout_keep_prob)
        shortcut = shortcuts[i]
        # concat with shortcut
        upsampled = tf.concat([upsampled, shortcut], 3)

        prev_block_feature_maps = []
        # note: we only end up actually using the output from the last dense block
        upsampled_output = dense_block(upsampled, is_training, dense_block_layer_counts[i], initializer_descriptor,
                                       add_features_per_layer=opt.arch_dense_block_add_features_per_layer,
                                       activation=activation_func,
                                       feature_maps_out=prev_block_feature_maps,
                                       weight_decay=opt.opt_weight_decay,
                                       dropout_keep_prob=dropout_keep_prob)

    # 1x1 conv
    output = conv2d(upsampled_output, 1, 1, opt.num_classes, initializer_descriptor,
                    weight_decay=opt.opt_weight_decay)

    return output


def _create_initializer_descriptor(opt):
    if opt.arch_initialization == 'he_truncated':
        return (opt.arch_initialization, )
    elif opt.arch_initialization == 'truncated_given_stddev':
        return (opt.arch_initialization, opt.arch_initialization_constant_stddev)
    raise NotImplementedError