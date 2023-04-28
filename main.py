import lightning as pl
from autoencoders import AutoEncoder
from callbacks import PlotLossCallback

if __name__ == "__main__":
    model = AutoEncoder()
    trainer = pl.Trainer(max_epochs=5, callbacks=[PlotLossCallback()])
    trainer.fit(model)

    # save the model checkpoint
    checkpoint_path = "my_model.ckpt"
    trainer.save_checkpoint(checkpoint_path)