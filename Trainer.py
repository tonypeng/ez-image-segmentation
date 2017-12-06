import nets
import tensorflow as tf
import utils
import random
slim = tf.contrib.slim

from data_pipelines import *
from TrainerOptions import *
import preprocessor as pre
import ade20k

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

        # Get datasets
        train_data = ade20k.get_split('training', 'data/ade20k/records')
        val_data  = ade20k.get_split('validation', 'data/ade20k/records')

        # dataset = self._get_dataset()
        # dl = (BatchedDataLoader
        #         .from_data(opt.data_root)
        #         .with_dataset(dataset)
        #         .image_dimensions(opt.image_width, opt.image_height)
        #         .training()
        #         .randomized()
        #       )

        # Path to save checkpoint files
        path_save = 'checkpoints/'+opt.model_name+'/'

        g = tf.Graph()
        with g.as_default(), g.device(opt.device), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            is_training = tf.placeholder(tf.bool, name='is_training')

            # Hyperparameters
            learning_rate = tf.placeholder(tf.float32)

            # Get data providers
            training_provider = slim.dataset_data_provider.DatasetDataProvider(
                                train_data,
                                num_readers=opt.num_readers,
                                common_queue_capacity=20 * opt.batch_size,
                                common_queue_min=10 * opt.batch_size)
            validation_provider = slim.dataset_data_provider.DatasetDataProvider(
                                val_data,
                                num_readers=opt.num_readers,
                                common_queue_capacity=20 * opt.batch_size,
                                common_queue_min=10 * opt.batch_size)
            training_queue = self._batch_queue(training_provider, True)
            validation_queue = self._batch_queue(validation_provider, False)

            # Input (x) / per-pixel output labels (y)
            q_selector = tf.cond(is_training,
                     lambda: tf.constant(0),
                     lambda: tf.constant(1))

            # select_q = tf.placeholder(tf.int32, [])
            q = tf.QueueBase.from_list(q_selector, [training_queue, validation_queue])

            # # Create batch of items.
            x, y = q.dequeue()

            # x = tf.placeholder(tf.float32, [None, opt.image_height, opt.image_width, 3])
            # y = tf.placeholder(tf.int64, [None, opt.image_height, opt.image_width, 1])

            # x_preproc = self._preprocess_data(x)

            # Construct network and compute spatial logits
            print("1. Constructing network...")
            with tf.variable_scope(opt.model_name):
                spatial_logits = self._construct_net(x, is_training)
            pred = tf.argmax(spatial_logits[0,:,:,:], axis=2)

            # Compute losses
            print("2. Creating losses...")
            # y_squeezed = tf.squeeze(y, axis=[3])
            labels = slim.one_hot_encoding(y, self.opt.num_classes)
            labels = tf.reshape(labels, spatial_logits.get_shape())
            loss = slim.losses.softmax_cross_entropy(spatial_logits, labels)
            # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=spatial_logits, labels=y_squeezed))
            # print(y.shape)
            # y_squeezed = tf.squeeze(y, axis=[3])
            # print(y_squeezed.shape)
            # loss = tf.reduce_mean(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=spatial_logits, labels=y_squeezed))
            if opt.opt_weight_decay is not None:
                regularizer = tf.add_n(tf.get_collection('weight_regularizers'))
                loss += regularizer

            # Create optimizer
            print("3. Optimizing...")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = self._construct_optimizer(learning_rate)
                optimize = optimizer.minimize(loss)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
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
                _, batch_x, batch_y = sess.run([optimize, x, y],
                         feed_dict={
                             learning_rate: curr_learning_rate,
                             is_training: True,
                         })

                # Compute validation loss
                if it % opt.val_loss_iter_print == 0:
                    curr_val_loss, val_loss_summ, learning_rate_summ, val_x, val_y, val_p = sess.run([loss, loss_valid_summary, learning_rate_summary, x, y, pred],
                                            feed_dict={
                                                learning_rate: curr_learning_rate,
                                                is_training: False})
                    utils.write_image(val_x[0],'imgs/x'+str(it)+'.png')
                    utils.write_image(val_y[0,:,:,0],'imgs/y'+str(it)+'.png', False)
                    utils.write_image(val_p,'imgs/p'+str(it)+'.png', False)


                    # adjust loss if we need to                                                                                                                                                                                        │··············
                    if self._should_adjust_learning_rate(curr_val_loss) and curr_learning_rate > 5e-5:
                        print ("Dropping learning rate from: " + str(curr_learning_rate))
                        curr_learning_rate = curr_learning_rate/opt.loss_adjustment_factor
                        curr_learning_rate = max(curr_learning_rate, opt.opt_min_learning_rate)
                        print ("                       to: " + str(curr_learning_rate))
                    writer.add_summary(val_loss_summ, it)
                    writer.add_summary(learning_rate_summ, it)

                    print("Iteration " + str(it + 1) + ": Val Loss=" + str(curr_val_loss))

                # Compute training loss
                if it % opt.train_loss_iter_print == 0:
                    curr_loss, loss_summ = sess.run([loss, loss_training_summary],
                                                    feed_dict={
                                                        learning_rate: curr_learning_rate,
                                                        is_training: False})
                    writer.add_summary(loss_summ, it)

                    print("Iteration " + str(it + 1) + ": Loss=" + str(curr_loss))

                # Save the model
                if it % opt.checkpoint_iterations == opt.checkpoint_iterations-1:
                    saver.save(sess, path_save, global_step=it)
                    print("Model saved at Iter %d !" %(it))
                it += 1
            coord.join(threads)
            sess.close()


    # def _preprocess_data(self, x: tf.Tensor) -> tf.Tensor:
    #     return x

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

    def _batch_queue(self, provider, is_training=True):
        [image, label] = provider.get(['image', 'label'])
        image, label = pre.preprocess_image(image, self.opt.image_height, self.opt.image_width, label=label, is_training=is_training)
        images, labels = tf.train.batch(
            [image, label],
            batch_size=self.opt.batch_size,
            num_threads=self.opt.num_preprocessing_threads,
            capacity=5 * self.opt.batch_size)
        return slim.prefetch_queue.prefetch_queue([images, labels], capacity=2)

    def _should_adjust_learning_rate(self, val_loss):
        self.loss_history.append(val_loss)
        if len(self.loss_history) > 3*self.opt.loss_adjustment_sample_interval:
            self.loss_history.pop(0)
            old_loss = sum(self.loss_history[:self.opt.loss_adjustment_sample_interval])/self.opt.loss_adjustment_sample_interval
            recent_loss = sum(self.loss_history[2*self.opt.loss_adjustment_sample_interval:])/self.opt.loss_adjustment_sample_interval
            if recent_loss > old_loss:
                self.loss_history = []
                return random.uniform(0, 1) < self.opt.loss_adjustment_coin_flip_prob
        return False
