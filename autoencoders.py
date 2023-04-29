import lightning as pl
from torch import nn
import torch
import typing as th
from torchvision import datasets, transforms
from torch.utils import data
from blocks import ConvBlock, DeconvBlock, LinearBlock, Reshape

# TODO: add docstrings, type-hinting, VAE

class LinearAutoEncoder(pl.LightningModule):
    def __init__(self,
                 latent_dim = 2,
                 encoder_layers = [784, 392, 196],
                 decoder_layers = [196, 392, 784],
                 regularizer = None):
        super(LinearAutoEncoder, self).__init__()

        self.training_step_outputs = []

        # REFACTOR TO USE ENCODER/DECODER LIST
        self.encoder = nn.Sequential(
            LinearBlock(784, 392),
            LinearBlock(392, 196),
            LinearBlock(196, 2)
        ) 

        self.decoder = nn.Sequential(
            LinearBlock(2, 196),
            LinearBlock(196, 392),
            LinearBlock(392, 784)
        ) 


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, 28*28)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        # Add optional L2 penalty if REG = TRUES
        self.training_step_outputs.append(loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        mnist_data = datasets.MNIST(root='./data', train=False, download=True, 
                                    transform=transforms.ToTensor())

        batch_size = 64
        dataloader = data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

        return dataloader

    def forward(self, input):
        z = self.encoder(input)
        x_hat = self.decoder(z)
        return x_hat


class ConvAutoEncoder(LinearAutoEncoder):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            Reshape(-1, 64, 4, 4),
            DeconvBlock(64, 32, output_padding=0),
            DeconvBlock(32, 16, output_padding=1),
            DeconvBlock(16, 1, output_padding=1, 
                        activation=nn.Sigmoid)
        )
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.training_step_outputs.append(loss)

        return loss
