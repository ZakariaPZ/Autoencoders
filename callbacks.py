import lightning as pl
import torch

class LossCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # do something with all training_step outputs, for example:
        epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
        pl_module.log("training_epoch_mean", epoch_mean)
        # free up the memory
        pl_module.training_step_outputs.clear()

    