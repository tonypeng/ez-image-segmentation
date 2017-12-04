import nets
import tensorflow as tf

from data_pipelines import *
from TrainerOptions import *

class Trainer:
    def __init__(self, opt: TrainerOptions):
        self.opt = TrainerOptions

    def train(self):
        # =================================
        # This Is Where The Sausage Is Made
        # =================================

        curr_learning_rate = self.learning_rate
        opt = self.opt

        data_loader = CocoDataLoader()

        g = tf.Graph()
        with g.as_default(), g.device(opt.device), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            is_training = tf.placeholder(tf.bool, name='is_training')

            # Hyperparameters
            learning_rate = tf.placeholder(tf.float32)
            dropout_keep_prob = tf.placeholder(tf.float32)

            # Input (x) / per-pixel output labels (y)
            x = tf.placeholder(tf.float32, [None, opt.image_height, opt.image_width, 3])
            y = tf.placeholder(tf.int64, [None, opt.image_height, opt.image_width, 1])

            # Pre-process data
            x_preproc = self._preprocess_data(x)

            # Construct network and compute spatial logits
            with tf.variable_scope(opt.model_name):
                spatial_logits = self._construct_net(x_preproc, is_training, dropout_keep_prob)

            # Compute losses
            y_squeezed = tf.squeeze(y, axis=[3])
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=spatial_logits, labels=y_squeezed))
            if opt.opt_weight_decay is not None:
                regularizer = tf.add_n(tf.get_collection('weight_regularizers'))
                loss += regularizer

            # Create optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = self._construct_optimizer(learning_rate)
                optimize = optimizer.minimize(loss)

            sess.run(tf.global_variables_initializer())

            it = 0
            while it < opt.opt_iterations:
                batch_x, batch_y = data_loader.next_batch(opt.batch_size)
                sess.run(optimize,
                         feed_dict={
                             x: batch_x,
                             y: batch_y,
                             learning_rate: curr_learning_rate,
                             dropout_keep_prob: opt.opt_dropout_keep_prob,
                             is_training: True,
                         })
                it += 1

    def _preprocess_data(self, x: tf.Tensor) -> tf.Tensor:
        return x

    def _construct_net(self, x: tf.Tensor, is_training, dropout_keep_prob) -> tf.Tensor:
        if self.opt.arch == 'tiramisu103':
            return nets.Tiramisu103(x, is_training, self.opt)
        raise NotImplementedError

    def _construct_optimizer(self, learning_rate):
        if self.opt.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif self.opt.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(
                learning_rate=learning_rate,
                decay=self.opt.opt_decay,
                momentum=self.opt.opt_momentum,
                epsilon=self.opt.opt_epsilon)
        elif self.opt.optimizer == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate, self.opt.opt_momentum)
        raise NotImplementedError