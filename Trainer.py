import nets
import Phases
import random
import tensorflow as tf

from data_pipelines import *
from TrainerOptions import *


class Trainer:
    def __init__(self, opt: TrainerOptions):
        self.opt = opt
        self.loss_history = []

    def train(self):
        # =================================
        # This Is Where The Sausage Is Made
        # =================================

        curr_learning_rate = self.opt.opt_learning_rate
        opt = self.opt

        # Path to save checkpoint files
        path_save = 'checkpoints/' + opt.model_name + '/'

        g = tf.Graph()
        with g.as_default(), g.device(opt.device), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            phase = tf.placeholder(tf.int32, name='phase')
            is_training = tf.placeholder(tf.bool, name='is_training')

            # Create data loader
            print('1. Creating data loader...')
            dataset = self._get_dataset()
            dl = (ParallelizedBatchedDataLoader
                  .from_data_root(opt.data_root)
                  .with_dataset(dataset)
                  .phase(phase)
                  .batch_size(opt.batch_size)
                  .num_readers(opt.num_readers)
                  .num_preprocessing_threads(opt.num_preprocessing_threads)
                  .pipeline_stage(Ade20kPreprocessingStage(is_training, opt.image_width, opt.image_height))
                  )

            # Hyperparameters
            learning_rate = tf.placeholder(tf.float32)

            # Input / annotations
            x, y = dl.next_batch()
            y_color = utils.colorize(y[0], vmin=0, vmax=dataset.num_classes(), cmap='viridis')

            # Construct network and compute spatial logits
            print("2. Constructing network...")
            with tf.variable_scope(opt.model_name):
                spatial_logits = self._construct_net(x, is_training)
            flattened_logits = tf.reshape(spatial_logits, (-1, dataset.num_classes()))
            preds = tf.argmax(spatial_logits, axis=3)
            pred_color = utils.colorize(preds[0], vmin=0, vmax=dataset.num_classes(), cmap='viridis')

            # Compute losses
            print("3. Setting up losses...")
            y_flattened = tf.squeeze(tf.reshape(y, (-1, 1)), axis=[1])
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flattened_logits, labels=y_flattened))
            if opt.opt_weight_decay is not None:
                regularizer = tf.add_n(tf.get_collection('weight_regularizers'))
                loss += regularizer

            # Create optimizer
            print("4. Optimizing...")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = self._construct_optimizer(learning_rate)
                optimize = optimizer.minimize(loss)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(tf.global_variables_initializer())

            # Tensorboard summaries
            learning_rate_summary = tf.summary.scalar('learning_rate', learning_rate)
            loss_training_summary = tf.summary.scalar('loss_training', loss)
            loss_valid_summary = tf.summary.scalar('loss_validation', loss)
            writer = tf.summary.FileWriter(os.path.join(opt.log_path, opt.model_name))
            writer.add_graph(sess.graph)

            # Checkpointing
            saver = tf.train.Saver(max_to_keep=5)
            it = 0
            if len(opt.checkpoint_name) > 1:
                saver.restore(sess, opt.checkpoint_name)
                it = opt.start_from_iteration

            while it < opt.opt_iterations:
                sess.run(optimize,
                         feed_dict={
                             learning_rate: curr_learning_rate,
                             phase: Phases.TRAINING,
                             is_training: True
                         })

                # Compute validation loss
                if it % opt.val_loss_iter_print == 0:
                    curr_val_loss, val_loss_summ, learning_rate_summ, val_x, yc, pc = sess.run(
                        [loss, loss_valid_summary, learning_rate_summary, x, y_color, pred_color],
                        feed_dict={
                            learning_rate: curr_learning_rate,
                            phase: Phases.VALIDATING,
                            is_training: False})

                    # output preview images
                    utils.write_image(val_x[0],
                                      utils.get_preview_file_path(opt.preview_images_path, 'inp', str(it), 'png'))
                    utils.write_image(yc,
                                      utils.get_preview_file_path(opt.preview_images_path, 'annotated', str(it), 'png'))
                    utils.write_image(pc,
                                      utils.get_preview_file_path(opt.preview_images_path, 'predicted', str(it), 'png'))

                    # adjust loss if we need to
                    if self._should_adjust_learning_rate(curr_val_loss) and curr_learning_rate > 5e-5:
                        print("Dropping learning rate from: " + str(curr_learning_rate))
                        curr_learning_rate = curr_learning_rate / opt.loss_adjustment_factor
                        curr_learning_rate = max(curr_learning_rate, opt.opt_min_learning_rate)
                        print("                       to: " + str(curr_learning_rate))
                    writer.add_summary(val_loss_summ, it)
                    writer.add_summary(learning_rate_summ, it)

                    print("Iteration " + str(it) + ": Val Loss=" + str(curr_val_loss))

                # Compute training loss
                if it % opt.train_loss_iter_print == 0:
                    curr_loss, loss_summ = sess.run([loss, loss_training_summary],
                                                    feed_dict={
                                                        learning_rate: curr_learning_rate,
                                                        phase: Phases.VALIDATING,
                                                        is_training: False})
                    writer.add_summary(loss_summ, it)

                    print("Iteration " + str(it) + ": Loss=" + str(curr_loss))

                # Save the model
                if it % opt.checkpoint_iterations == 0:
                    saver.save(sess, path_save, global_step=it)
                    print("Model saved at Iter %d !" % (it))
                it += 1
            coord.join(threads)
            sess.close()

    def _get_dataset(self):
        if self.opt.dataset == 'ade20k':
            return Ade20kTfRecords
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

    def _should_adjust_learning_rate(self, val_loss):
        self.loss_history.append(val_loss)
        if len(self.loss_history) > 3 * self.opt.loss_adjustment_sample_interval:
            self.loss_history.pop(0)
            old_loss = sum(
                self.loss_history[:self.opt.loss_adjustment_sample_interval]) / self.opt.loss_adjustment_sample_interval
            recent_loss = sum(self.loss_history[
                              2 * self.opt.loss_adjustment_sample_interval:]) / self.opt.loss_adjustment_sample_interval
            if recent_loss > old_loss:
                self.loss_history = []
                return random.uniform(0, 1) < self.opt.loss_adjustment_coin_flip_prob
        return False
