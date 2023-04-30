import lightning as pl
from autoencoders import ConvAutoEncoder, LinearAutoEncoder
from callbacks import LossCallback
from torchvision import datasets, transforms
from torch.utils import data
import torch

if __name__ == "__main__":

    # Transformations applied on each image => only make them a tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # Loading the test set
    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False)

    model = ConvAutoEncoder()
    trainer = pl.Trainer(max_epochs=5, callbacks=[LossCallback()]) # add callbacks for sample reconstruction per epoch
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(test_result)
    # result = {"test": test_result, "val": val_result}
    
    # save the model checkpoint
    checkpoint_path = "my_model.ckpt" 
    trainer.save_checkpoint(checkpoint_path)

    # model = Autoencoder.load_from_checkpoint(pretrained_filename)
    # above: load model and use for generation
    # fix activations in models final layers