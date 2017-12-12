from Evaluate import *
from TrainerOptions import *

opt = TrainerOptions()
opt.parse_args()

evaluator = Evaluator(opt)
evaluator.train()
