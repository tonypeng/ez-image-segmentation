import nets
import tensorflow as tf

from data_pipelines import *
from TrainerOptions import *

class Trainer:
    def __init__(self, opt: TrainerOptions):
        self.opt = opt

    def train(self):
        # =================================
        # This Is Where The Sausage Is Made
        # =================================

        curr_learning_rate = self.opt.opt_learning_rate
        opt = self.opt

        dataset = self._get_dataset()
        dl = (BatchedDataLoader
                .from_data(opt.data_root)
                .with_dataset(dataset)
                .image_dimensions(opt.image_width, opt.image_height)
                .training()
                .randomized()
              )

        # Path to save checkpoint files
        path_save = './checkpoints/'+opt.model_name+'/'

        g = tf.Graph()
        with g.as_default(), g.device(opt.device), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            is_training = tf.placeholder(tf.bool, name='is_training')

            # Hyperparameters
            learning_rate = tf.placeholder(tf.float32)

            # Input (x) / per-pixel output labels (y)
            x = tf.placeholder(tf.float32, [None, opt.image_height, opt.image_width, 3])
            y = tf.placeholder(tf.int64, [None, opt.image_height, opt.image_width, 1])

            # Pre-process data
            x_preproc = self._preprocess_data(x)

            # Construct network and compute spatial logits
            print("1. Constructing network...")
            with tf.variable_scope(opt.model_name):
                spatial_logits = self._construct_net(x_preproc, is_training)

            # Compute losses
            print("2. Creating losses...")
            y_squeezed = tf.squeeze(y, axis=[3])
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=spatial_logits, labels=y_squeezed))
            if opt.opt_weight_decay is not None:
                regularizer = tf.add_n(tf.get_collection('weight_regularizers'))
                loss += regularizer

            # Create optimizer
            print("3. Optimizing...")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = self._construct_optimizer(learning_rate)
                optimize = optimizer.minimize(loss)

            sess.run(tf.global_variables_initializer())

            # Tensorboard summary for meta-params
            # Run Tensorboard with $tensorboard --logdir=log_path
            learning_rate_summary = tf.summary.scalar('learning_rate', learning_rate)
            loss_training_summary = tf.summary.scalar('loss_training', loss)
            loss_valid_summary = tf.summary.scalar('loss_validation', loss)
            writer = tf.summary.FileWriter(os.path.join(opt.log_path, opt.model_name), graph=tf.get_default_graph())

            # Checkpointing
            print("4. Saving at " + path_save + "...")
            saver = tf.train.Saver(max_to_keep=5)
            it = 0
            if len(opt.checkpoint_name)>1:
                saver.restore(sess, opt.checkpoint_name)
                it = opt.start_from_iteration

            while it < opt.opt_iterations:
                batch_x, batch_y = dl.next_batch(opt.batch_size)
                sess.run(optimize,
                         feed_dict={
                             x: batch_x,
                             y: batch_y,
                             learning_rate: curr_learning_rate,
                             is_training: True,
                         })

                # Compute validation loss
                if it % opt.val_loss_iter_print == 0:
                    images_batch_val, labels_batch_val = dl.next_batch(opt.batch_size)
                    curr_val_loss, val_loss_summ, learning_rate_summ = sess.run([loss, loss_valid_summary, learning_rate_summary],
                                            feed_dict={
                                                x: images_batch_val,
                                                y: labels_batch_val,
                                                learning_rate: curr_learning_rate,
                                                keep_dropout: opt.dropout_keep_prob,
                                                is_training: False})

                    # adjust loss if we need to                                                                                                                                                                                        │··············
                    if opt._should_adjust_learning_rate(curr_val_loss) and curr_learning_rate > 5e-5:
                        print ("Dropping learning rate from: " + str(curr_learning_rate))
                        curr_learning_rate = curr_learning_rate/opt.loss_adjustment_factor
                        curr_learning_rate = max(curr_learning_rate, opt.min_learning_rate)
                        print ("                       to: " + str(curr_learning_rate))
                    writer.add_summary(val_loss_summ, it)
                    writer.add_summary(learning_rate_summ, it)

                    print("Iteration " + str(it + 1) + ": Val Loss=" + str(curr_val_loss))

                # Compute training loss
                if it % opt.train_loss_iter_print == 0:
                    curr_loss, loss_summ = sess.run([loss, loss_training_summary],
                                                    feed_dict={
                                                        x: images_batch,
                                                        y: labels_batch,
                                                        learning_rate: curr_learning_rate,
                                                        keep_dropout: opt.dropout_keep_prob,
                                                        is_training: False})
                    writer.add_summary(loss_summ, it)

                    print("Iteration " + str(it + 1) + ": Loss=" + str(curr_loss))

                # Save the model
                if it % opt.checkpoint_iterations == 0:
                    saver.save(sess, path_save, global_step=it)
                    print("Model saved at Iter %d !" %(it))
                it += 1

    def _preprocess_data(self, x: tf.Tensor) -> tf.Tensor:
        return x

    def _get_dataset(self):
        if self.opt.dataset == 'mitade':
            return MitAde
        raise NotImplementedError

    def _construct_net(self, x: tf.Tensor, is_training) -> tf.Tensor:
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
