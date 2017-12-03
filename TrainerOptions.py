import argparse

class TrainerOptions:
    def __init__(self):
        self.options = {}

    def __getattr__(self, item):
        return self.options.get(item)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--arch', type=str, default='densenet')
        parser.add_argument('--data_root', type=str, default='data/')
        parser.add_argument('--opt_learning_rate', type=float, default=1e-2)
        self.options = vars(parser.parse_args())
