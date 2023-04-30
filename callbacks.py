from typing import Any, Optional
import lightning as pl
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from numpy.random import randint 
import typing as th

class LossCallback(pl.Callback):
    def on_validation_batch_end(self, 
                                trainer: pl.Trainer, 
                                pl_module: pl.LightningModule, 
                                outputs: th.Union[STEP_OUTPUT, None], 
                                batch: Any, 
                                batch_idx: int, 
                                dataloader_idx: int = 0) -> None:
    
        x, _ = batch
        idx = randint(0, x.shape[0])
        image = x[idx]
        reconstruction = pl_module(image.unsqueeze(0))
        pl_module.logger.experiment.add_image('Original', image, 0)
        pl_module.logger.experiment.add_image('Reconstruction', reconstruction.squeeze(0), 1)