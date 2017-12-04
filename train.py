from TrainerOptions import *

opt = TrainerOptions()
opt.parse_args()

print(opt.safe_get('opt_learning_rate', int))