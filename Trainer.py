import nets
import Phases
import random
from SegmentationColorizer import *
import tensorflow as tf

from data_pipelines import *
# from data_pipelines2 import *
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
            dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

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
                  .pipeline_stage(Ade20kPreprocessingStage(opt.image_width, opt.image_height))
                  )
            # dl_train = (
            #     BatchedDataLoader
            #         .from_data(opt.data_root)
            #         .with_dataset(dataset)
            #         .randomized()
            #         .image_dimensions(opt.image_width, opt.image_height)
            #         .training()
            # )
            # dl_val = (
            #     BatchedDataLoader
            #         .from_data(opt.data_root)
            #         .with_dataset(dataset)
            #         .randomized()
            #         .image_dimensions(opt.image_width, opt.image_height)
            #         .training()
            # )

            # Hyperparameters
            learning_rate = tf.placeholder(tf.float32)

            # Input / annotations
            x, y = dl.next_batch()
            x_pre = x - dataset.mean_pixel()
            # x = tf.placeholder(tf.float32, [None, opt.image_height, opt.image_width, 3])
            # y = tf.placeholder(tf.int32, [None, opt.image_height, opt.image_width, 1])
            colorizer = SegmentationColorizer(0, dataset.num_classes(), opt.colorizer_map)
            y0_color = colorizer.colorize(y[0])

            # Construct network and compute spatial logits
            print("2. Constructing network...")
            with tf.variable_scope(opt.model_name):
                outputs = self._construct_net(x_pre, is_training, dropout_keep_prob, dataset.num_classes())
            preds = tf.argmax(outputs[0][0], axis=3)
            pred0_color = colorizer.colorize(preds[0])

            # Compute losses
            print("3. Setting up losses and accuracies...")
            y_one_hot = tf.one_hot(y, dataset.num_classes())
            output_shape = utils.tensor_shape_as_list(outputs[0][0])
            output_shape[0] = -1
            y_one_hot = tf.reshape(y_one_hot, output_shape)

            loss = 0
            for i in range(len(outputs)):
                spatial_logits = outputs[i][0]
                loss += outputs[i][1] * tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=spatial_logits, labels=y_one_hot))
            if opt.opt_weight_decay is not None:
                regularizer = tf.add_n(tf.get_collection('weight_regularizers'))
                loss += regularizer

            pixels_equal = tf.equal(tf.squeeze(tf.cast(preds, tf.int32)), tf.squeeze(y))
            acc_per_pixel = tf.reduce_mean(tf.cast(pixels_equal, tf.float32))

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
            acc_training_summary = tf.summary.scalar('acc_training', acc_per_pixel)
            acc_valid_summary = tf.summary.scalar('acc_validation', acc_per_pixel)
            writer = tf.summary.FileWriter(os.path.join(opt.log_path, opt.model_name))
            writer.add_graph(sess.graph)

            # Checkpointing
            saver = tf.train.Saver(max_to_keep=5)
            it = 0
            processed_samples = 0
            last_loss_change_processed_samples = 0
            if len(opt.checkpoint_name) > 1:
                saver.restore(sess, opt.checkpoint_name)
                it = opt.start_from_iteration

            while it < opt.opt_iterations:
                # x_batch, y_batch = dl_train.next_batch(opt.batch_size)
                _, train_loss, train_loss_summ, train_acc, train_acc_summ = sess.run(
                    [optimize, loss, loss_training_summary, acc_per_pixel, acc_training_summary],
                    feed_dict={
                        # x: x_batch,
                        # y: y_batch,
                        learning_rate: curr_learning_rate,
                        phase: Phases.TRAINING,
                        is_training: True,
                        dropout_keep_prob: opt.opt_dropout_keep_prob,
                    })
                # utils.write_image(val_x[0]+dataset.mean_pixel(),
                #                   utils.get_preview_file_path(opt.preview_images_path, 'inp', str(it), 'png'))
                writer.add_summary(train_loss_summ, it)
                writer.add_summary(train_acc_summ, it)

                print("Iteration " + str(it) + ": Loss=" + str(train_loss) + "; Acc=" + str(train_acc * 100.) + "%")

                processed_samples += opt.batch_size

                # Compute validation loss
                if it % opt.val_loss_iter_print == 0:
                    # x_val_batch, y_val_batch = dl_val.next_batch(opt.batch_size)
                    curr_val_loss, val_loss_summ, learning_rate_summ, val_x, yc, pc, val_acc, val_acc_summ = sess.run(
                        [loss, loss_valid_summary, learning_rate_summary, x, y0_color, pred0_color, acc_per_pixel,
                         acc_valid_summary],
                        feed_dict={
                            # x: x_val_batch,
                            # y: y_val_batch,
                            learning_rate: curr_learning_rate,
                            phase: Phases.VALIDATING,
                            is_training: False,
                            dropout_keep_prob: 1.0,
                        })

                    # output preview images
                    utils.write_image(val_x[0],
                                      utils.get_preview_file_path(opt.preview_images_path, 'inp', str(it), 'png'))
                    utils.write_image(yc,
                                      utils.get_preview_file_path(opt.preview_images_path, 'annotated', str(it), 'png'))
                    utils.write_image(pc,
                                      utils.get_preview_file_path(opt.preview_images_path, 'predicted', str(it), 'png'))

                    # adjust loss if we need to
                    if (self._should_adjust_learning_rate(curr_val_loss)
                        and curr_learning_rate > opt.opt_min_learning_rate
                        and (processed_samples - last_loss_change_processed_samples)
                            >= dataset.num_training_samples() * opt.loss_adjustment_min_epochs
                        or (processed_samples - last_loss_change_processed_samples)
                            >= dataset.num_training_samples() * opt.loss_adjustment_max_epochs):
                        print("Dropping learning rate from: " + str(curr_learning_rate))
                        curr_learning_rate = curr_learning_rate / opt.loss_adjustment_factor
                        curr_learning_rate = max(curr_learning_rate, opt.opt_min_learning_rate)
                        print("                       to: " + str(curr_learning_rate))
                        last_loss_change_processed_samples = processed_samples
                    writer.add_summary(val_loss_summ, it)
                    writer.add_summary(learning_rate_summ, it)
                    writer.add_summary(val_acc_summ, it)

                    print("Iteration " + str(it) + ": Val Loss=" + str(curr_val_loss) + "; Acc=" + str(
                        val_acc * 100.) + "%")

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
        if self.opt.dataset == 'camvid':
            return CamVidTfRecords
        raise NotImplementedError

    # def _get_dataset(self):
    #     if self.opt.dataset == 'ade20k':
    #         return MitAde
    #     if self.opt.dataset == 'camvid':
    #         return CamVid
    #     raise NotImplementedError

    def _construct_net(self, x: tf.Tensor, is_training, dropout_keep_prob, num_classes):
        if self.opt.arch == 'tiramisu':
            return [(nets.Tiramisu(x, is_training, dropout_keep_prob, num_classes, self.opt), 1.0)]
        if self.opt.arch == 'AtrousStridedNet':
            return [(nets.AtrousStridedNet(x, is_training, dropout_keep_prob, num_classes, self.opt), 1.0)]
        if self.opt.arch == 'AtrousStridedResizeNet':
            return [(nets.AtrousStridedResizeNet(x, is_training, dropout_keep_prob, num_classes, self.opt), 1.0)]
        if self.opt.arch == 'StridedNet':
            return [(nets.StridedNet(x, is_training, dropout_keep_prob, num_classes, self.opt), 1.0)]
        if self.opt.arch == 'AtrousNet':
            return [(nets.AtrousNet(x, is_training, dropout_keep_prob, num_classes, self.opt), 1.0)]
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
