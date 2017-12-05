from Trainer import *
from TrainerOptions import *

opt = TrainerOptions()
opt.parse_args()

trainer = Trainer(opt)
trainer.train()