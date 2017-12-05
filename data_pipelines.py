import os
from random import shuffle


class BatchedDataLoader:
    _TRAIN = 0
    _VALID = 1
    _TEST = 2

    @classmethod
    def from_dataset(cls, dataset):
        return BatchedDataLoader(dataset)

    def __init__(self, source):
        self.data_root = source
        self.randomize = False
        self.data_idx = None
        self.pipeline = None
        self.phase = BatchedDataLoader._TRAIN
        self.data_file_paths = None

    def with_pipeline(self, pipeline):
        self.pipeline = pipeline(self.data_root)
        return self

    def randomized(self):
        self.randomize = True
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

        batch_x_file_paths = [None] * n

        for i in range(n):
            batch_x_file_paths[i] = self.data_file_paths[self.data_idx]
            self.data_idx += 1
            if self.data_idx == len(self.data_file_paths):
                self.data_idx = 0
                shuffle(self.data_file_paths)

        return batch_x_file_paths

    def _initialize_dataset(self):
        if self.phase == BatchedDataLoader._TRAIN:
            self.data_file_paths = self.pipeline.get_train_file_paths()
        elif self.phase == BatchedDataLoader._VALID:
            self.data_file_paths = self.pipeline.get_valid_file_paths()
        elif self.phase == BatchedDataLoader._TEST:
            self.data_file_paths = self.pipeline.get_test_file_paths()
        raise NotImplementedError


class Coco:
    def __init__(self, data_root):
        self.data_root = data_root

    def get_train_file_paths(self):
        return [os.path.join(self.data_root, f) for f in os.listdir(self.data_root)]

    def get_valid_file_paths(self):
        pass

    def get_test_file_paths(self):
        pass


class MitAde:
    def __init__(self, data_root):
        self.data_root = data_root

    def get_train_file_paths(self):
        pass

    def get_valid_file_paths(self):
        pass

    def get_test_file_paths(self):
        pass


