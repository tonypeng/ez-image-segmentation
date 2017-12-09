import numpy as np
import os
import utils
from random import shuffle

import libs.ade20k as ade20k
import libs.camvid as camvid
import libs.ade20k_preprocessor as ade20k_pre
import libs.camvid_preprocessor as camvid_pre


class BatchedDataLoader:
    _TRAIN = 0
    _VALID = 1
    _TEST = 2

    @classmethod
    def from_data(cls, data_root):
        return BatchedDataLoader(data_root)

    def __init__(self, source):
        self.data_root = source
        self.randomize = False
        self.data_idx = None
        self.dataset = None
        self.image_width = None
        self.image_height = None
        self.phase = BatchedDataLoader._TRAIN
        self.data_file_paths = None

    def with_dataset(self, dataset):
        self.dataset = dataset(self.data_root)
        return self

    def randomized(self):
        self.randomize = True
        return self

    def image_dimensions(self, width, height):
        self.image_width = width
        self.image_height = height
        return self

    def training(self):
        self.phase = BatchedDataLoader._TRAIN
        return self

    def validation(self):
        self.phase = BatchedDataLoader._VALID
        return self

    def testing(self):
        self.phase = BatchedDataLoader._TEST
        return self

    def next_batch(self, n):
        if self.data_idx is None:
            self._initialize_dataset()

        batch_x = [None] * n
        batch_y = [None] * n

        for i in range(n):
            batch_x_file_path, batch_y_file_path = self.data_file_paths[self.data_idx]
            x = utils.read_image(batch_x_file_path, size=(self.image_height, self.image_width))
            batch_x[i] = x.astype(np.float32) / 255. - self.dataset.mean_pixel()
            y = utils.read_image(batch_y_file_path,
                                          mode='I', size=(self.image_height, self.image_width)).astype('int32')
            y = np.expand_dims(y, 3)
            batch_y[i] = y

            self.data_idx += 1
            if self.data_idx == len(self.data_file_paths):
                self.data_idx = 0
                shuffle(self.data_file_paths)

        return batch_x, batch_y

    def _initialize_dataset(self):
        self.data_idx = 0
        if self.phase == BatchedDataLoader._TRAIN:
            self.data_file_paths = self.dataset.get_train_file_paths()
        elif self.phase == BatchedDataLoader._VALID:
            self.data_file_paths = self.dataset.get_valid_file_paths()
        elif self.phase == BatchedDataLoader._TEST:
            self.data_file_paths = self.dataset.get_test_file_paths()

class MitAde:
    def __init__(self, data_root):
        self.data_root = os.path.join(data_root, 'mitade')
        self.data_mean = np.array([124.6901 / 255., 118.6897 / 255., 109.5388 / 255.])

    def get_data_mean(self):
        return self.data_mean

    def get_train_file_paths(self):
        train_data_root_x = os.path.join(self.data_root, 'images', 'training')
        train_data_root_y = os.path.join(self.data_root, 'annotations', 'training')
        train_data_filepaths_x = []
        train_data_filepaths_y = []
        for f in os.listdir(train_data_root_x):
            if not os.path.exists(os.path.join(train_data_root_y, f)):
                print("Warning: Annotation doesn't exist for file ", f, ".")
                continue
            train_data_filepaths_x.append(os.path.join(train_data_root_x, f))
            train_data_filepaths_y.append(os.path.join(train_data_root_y, f))

        return list(zip(train_data_filepaths_x, train_data_filepaths_y))

    def get_valid_file_paths(self):
        valid_data_root_x = os.path.join(self.data_root, 'images', 'validation')
        valid_data_root_y = os.path.join(self.data_root, 'annotations', 'validation')
        valid_data_filepaths_x = []
        valid_data_filepaths_y = []
        for f in os.listdir(valid_data_root_x):
            if not os.path.exists(os.path.join(valid_data_root_y, f)):
                print("Warning: Annotation doesn't exist for file ", f, ".")
                continue
            valid_data_filepaths_x.append(os.path.join(valid_data_root_x, f))
            valid_data_filepaths_y.append(os.path.join(valid_data_root_y, f))

        return list(zip(valid_data_filepaths_x, valid_data_filepaths_y))

    def get_test_file_paths(self):
        test_data_root_x = os.path.join(self.data_root, 'images', 'test')
        test_data_root_y = os.path.join(self.data_root, 'annotations', 'test')
        test_data_filepaths_x = []
        test_data_filepaths_y = []
        for f in os.listdir(test_data_root_x):
            if not os.path.exists(os.path.join(test_data_root_y, f)):
                print("Warning: Annotation doesn't exist for file ", f, ".")
                continue
            test_data_filepaths_x.append(os.path.join(test_data_root_x, f))
            test_data_filepaths_y.append(os.path.join(test_data_root_y, f))

        return list(zip(test_data_filepaths_x, test_data_filepaths_y))

class CamVid:
    @classmethod
    def num_classes(cls):
        return camvid._NUM_CLASSES

    @classmethod
    def num_training_samples(cls):
        return camvid.SPLITS_TO_SIZES['training']

    @classmethod
    def mean_pixel(cls):
        return np.array([camvid_pre._R_MEAN, camvid_pre._G_MEAN, camvid_pre._B_MEAN])

    def __init__(self, data_root):
        self.data_root = os.path.join(data_root, 'CamVid', 'raw')

    def get_train_file_paths(self):
        train_data_root_x = os.path.join(self.data_root, 'train')
        train_data_root_y = os.path.join(self.data_root, 'trainannot')
        train_data_filepaths_x = []
        train_data_filepaths_y = []
        for f in os.listdir(train_data_root_x):
            if not os.path.exists(os.path.join(train_data_root_y, f)):
                print("Warning: Annotation doesn't exist for file ", f, ".")
                continue
            train_data_filepaths_x.append(os.path.join(train_data_root_x, f))
            train_data_filepaths_y.append(os.path.join(train_data_root_y, f))

        return list(zip(train_data_filepaths_x, train_data_filepaths_y))

    def get_valid_file_paths(self):
        valid_data_root_x = os.path.join(self.data_root, 'val')
        valid_data_root_y = os.path.join(self.data_root, 'valannot')
        valid_data_filepaths_x = []
        valid_data_filepaths_y = []
        for f in os.listdir(valid_data_root_x):
            if not os.path.exists(os.path.join(valid_data_root_y, f)):
                print("Warning: Annotation doesn't exist for file ", f, ".")
                continue
            valid_data_filepaths_x.append(os.path.join(valid_data_root_x, f))
            valid_data_filepaths_y.append(os.path.join(valid_data_root_y, f))

        return list(zip(valid_data_filepaths_x, valid_data_filepaths_y))

    def get_test_file_paths(self):
        test_data_root_x = os.path.join(self.data_root, 'test')
        test_data_root_y = os.path.join(self.data_root, 'testannot')
        test_data_filepaths_x = []
        test_data_filepaths_y = []
        for f in os.listdir(test_data_root_x):
            if not os.path.exists(os.path.join(test_data_root_y, f)):
                print("Warning: Annotation doesn't exist for file ", f, ".")
                continue
            test_data_filepaths_x.append(os.path.join(test_data_root_x, f))
            test_data_filepaths_y.append(os.path.join(test_data_root_y, f))

        return list(zip(test_data_filepaths_x, test_data_filepaths_y))
