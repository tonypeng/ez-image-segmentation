import argparse

class TrainerOptions:
    def __init__(self):
        self.options = {}

    def __getattr__(self, item):
        return self.safe_get(item, None)

    def safe_get(self, key, exp_type):
        if key not in self.options:
            raise KeyError
        val = self.options.get(key)
        if exp_type is not None:
            # never forget the Image Mean Incident of 2017
            if not isinstance(val, exp_type):
                print('!!! WARNING: Trainer option ' + str(key) + ' was of type ' + type(
                    val).__name__ + '; expected ' + exp_type.__name__ + '.')
        return val

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('model_name', type=str)
        parser.add_argument('--arch', type=str, default='tiramisu103')
        parser.add_argument('--data_root', type=str, default='data/')
        parser.add_argument('--image_width', type=int, default=256)
        parser.add_argument('--image_height', type=int, default=256)
        parser.add_argument('--num_classes', type=int, default=100)
        parser.add_argument('--device', type=str, default='/gpu:0')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--dataset', type=str, default='mitade')
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--arch_first_conv_features', type=int, default=48)
        parser.add_argument('--arch_activation', type=str, default='relu')
        parser.add_argument('--arch_dense_block_layer_counts', type=str, default='4,5,7,10,12')
        parser.add_argument('--arch_dense_block_add_features_per_layer', type=int, default=16)
        parser.add_argument('--arch_bottleneck_layer_count', type=int, default=15)
        parser.add_argument('--opt_iterations', type=int, default=100000)
        parser.add_argument('--opt_learning_rate', type=float, default=1e-2)
        parser.add_argument('--opt_momentum', type=float, default=0.0)
        parser.add_argument('--opt_decay', type=float, default=0.9)
        parser.add_argument('--opt_epsilon', type=float, default=1e-10)
        parser.add_argument('--opt_weight_decay', type=float, default=1e-4)
        parser.add_argument('--opt_dropout_keep_prob', type=float, default=None)
        self.options = vars(parser.parse_args())
