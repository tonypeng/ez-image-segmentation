import nets
import Phases
import random
from SegmentationColorizer import *
import tensorflow as tf

from data_pipelines import *
# from data_pipelines2 import *
from TrainerOptions import *


class Evaluator:
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
                  .batch_size(1)
                  .num_readers(1)
                  .num_preprocessing_threads(1)
                  .pipeline_stage(Ade20kPreprocessingStage(opt.image_width, opt.image_height))
                  )

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

            # Create optimizer
            print("4. Optimizing...")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = self._construct_optimizer(learning_rate)
                optimize = optimizer.minimize(loss)

            pixel_acc = tf.contrib.metrics.streaming_accuracy(tf.squeeze(tf.cast(preds, tf.int32)), tf.squeeze(y))
            mean_iou = tf.contrib.metrics.streaming_mean_iou(tf.squeeze(tf.cast(preds, tf.int32)), tf.squeeze(y), dataset.num_classes()+1)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(tf.global_variables_initializer())

            # Checkpointing
            saver = tf.train.Saver(max_to_keep=5)
            if len(opt.checkpoint_name) > 1:
                saver.restore(sess, opt.checkpoint_name)
                it = opt.start_from_iteration

            with tf.name_scope("streaming"):
                # clear counters for a fresh evaluation
                sess.run(tf.local_variables_initializer())
            for _ in range (dataset.num_validation_samples()):
                (pa, _), (mi, _)  = sess.run(
                    [pixel_acc, mean_iou],
                    feed_dict={
                        learning_rate: curr_learning_rate,
                        phase: Phases.VALIDATING,
                        is_training: False,
                        dropout_keep_prob: 1.0,
                    })
            print("Val Mean pixel accuracy: " + pa)
            print("Val Mean IoU: " + mi)
            for _ in range(dataset.num_training_samples()):
                (pa, _), (mi, _) = sess.run(
                    [pixel_acc, mean_iou],
                    feed_dict={
                        learning_rate: curr_learning_rate,
                        phase: Phases.TRAINING,
                        is_training: False,
                        dropout_keep_prob: 1.0,
                    })
                print(pa, mi)
            print("Train Mean pixel accuracy: " + pa)
            print("Train Mean IoU: " + mi)
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
