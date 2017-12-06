import libs.ade20k as ade20k
import libs.preprocessor as ade20k_pre
import numpy as np
import os
import Phases
import tensorflow as tf
import utils
from random import shuffle

slim = tf.contrib.slim


class ParallelizedBatchedDataLoader:
    @classmethod
    def from_data_root(cls, data_root) -> 'ParallelizedBatchedDataLoader':
        return ParallelizedBatchedDataLoader(data_root)

    def __init__(self, source):
        self.data_root = source
        self.dataset = None
        self._queue = None
        self._phase = None
        self._batch_size = None
        self._num_readers = None
        self._num_preproc_threads = None
        self._pipeline_stages = []

    def with_dataset(self, dataset) -> 'ParallelizedBatchedDataLoader':
        self.dataset = dataset(self.data_root)
        return self

    def phase(self, phase) -> 'ParallelizedBatchedDataLoader':
        self._phase = phase
        return self

    def batch_size(self, size) -> 'ParallelizedBatchedDataLoader':
        self._batch_size = size
        return self

    def num_readers(self, num) -> 'ParallelizedBatchedDataLoader':
        self._num_readers = num
        return self

    def num_preprocessing_threads(self, num) -> 'ParallelizedBatchedDataLoader':
        self._num_preproc_threads = num
        return self

    def pipeline_stage(self, stage) -> 'ParallelizedBatchedDataLoader':
        self._pipeline_stages.append(stage)
        return self

    def next_batch(self):
        if self._queue is None:
            self._initialize_queue()

        return self._queue.dequeue()

    def _initialize_queue(self):
        train_queue = self._create_queue_from(self.dataset.get_train_data(), True)
        val_queue = self._create_queue_from(self.dataset.get_validation_data(), False)
        test_queue = self._create_queue_from(self.dataset.get_test_data(), False)
        queues = [train_queue, val_queue, test_queue]
        selector = tf.case({
            tf.equal(self._phase, Phases.TRAINING): lambda: tf.constant(0),
            tf.equal(self._phase, Phases.VALIDATING): lambda: tf.constant(1),
            tf.equal(self._phase, Phases.TESTING): lambda: tf.constant(2),
        })
        self._queue = tf.QueueBase.from_list(selector, queues)

    def _create_queue_from(self, data, is_training):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            data,
            num_readers=self._num_readers,
            common_queue_capacity=20 * self._batch_size,
            common_queue_min=10 * self._batch_size)
        [image, label] = provider.get(['image', 'label'])
        for stage in self._pipeline_stages:
            image, label = stage.apply(image, label, is_training)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=self._batch_size,
            num_threads=self._num_preproc_threads,
            capacity=5 * self._batch_size)
        return slim.prefetch_queue.prefetch_queue([images, labels], capacity=2)


class Ade20kTfRecords:
    @classmethod
    def num_classes(cls):
        return ade20k._NUM_CLASSES

    def __init__(self, data_root):
        self.data_root = os.path.join(data_root, 'ade20k', 'records')

    def get_train_data(self):
        return ade20k.get_split('training', self.data_root)

    def get_validation_data(self):
        return ade20k.get_split('validation', self.data_root)

    def get_test_data(self):
        # TODO: replace with actual test data
        return ade20k.get_split('validation', self.data_root)

class Ade20kPreprocessingStage:
    def __init__(self, resize_image_width, resize_image_height):
        self.resize_image_width = resize_image_width
        self.resize_image_height = resize_image_height

    def apply(self, image, label, is_training):
        return ade20k_pre.preprocess_image(image, self.resize_image_height, self.resize_image_width,
                                           label=label, is_training=is_training)
