import lightning as pl
from torch import nn
import torch
import typing as th
from torchvision import datasets, transforms
from torch.utils import data
from blocks import ConvBlock, DeconvBlock, LinearBlock, Reshape

# TODO: add docstrings, type-hinting, VAE

class LinearAutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim):
        super(LinearAutoEncoder, self).__init__()
        self.training_step_outputs = []

        if latent_dim >= input_dim:
            raise ValueError("Latent dimension cannot be larger than input dimension.")

        self.encoder = nn.Sequential(
            LinearBlock(input_dim, 392),
            LinearBlock(392, 196),
            LinearBlock(196, latent_dim)
        ) 

        self.decoder = nn.Sequential(
            LinearBlock(latent_dim, 196),
            LinearBlock(196, 392),
            LinearBlock(392, input_dim)
        ) 

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, 28*28)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
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


class SparseAutoencoder(LinearAutoEncoder):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        if latent_dim < input_dim:
            raise ValueError("Latent dimension cannot be smaller than input dimension.")

        self.encoder = nn.Sequential(
            LinearBlock(input_dim, 1024),
            LinearBlock(1024, latent_dim),
        ) 

        self.decoder = nn.Sequential(
            LinearBlock(latent_dim, 1024),
            LinearBlock(1024, input_dim)
        ) 


class ConvAutoEncoder(LinearAutoEncoder):
    def __init__(self, kernel_size=3, padding=1, latent_dim = 8):
        super().__init__()
        stride = 2
    
        self.encoder = nn.Sequential(
            ConvBlock(1, 16, kernel_size, stride, padding),
            ConvBlock(16, 32, kernel_size, stride, padding),
            ConvBlock(32, 64, kernel_size, stride, padding),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            Reshape(-1, 64, 4, 4),
            DeconvBlock(64, 32, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0),
            DeconvBlock(32, 16, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
            DeconvBlock(16, 1, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1, 
                        activation=nn.Sigmoid)
        )
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.training_step_outputs.append(loss)

        return loss

    def forward(self, input):
        z = self.encoder(input)
        x_hat = self.decoder(z)
        return x_hat