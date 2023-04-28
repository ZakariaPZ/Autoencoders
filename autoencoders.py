import lightning as pl
from torch import nn
import torch
import typing as th
from torchvision import datasets, transforms
from torch.utils import data

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class LinearBlock(nn.Module):
    def __init__(self, 
                 input_size,
                 output_size,
                 activation : th.Callable[[torch.Tensor], torch.Tensor] = nn.ReLU):
        super(ConvBlock, self).__init__()
        self.linear = nn.Linear(input_size,
                              output_size)
        self.bn = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 activation : th.Callable[[torch.Tensor], torch.Tensor] = nn.ReLU, 
                 kernel_size=3, 
                 stride=2, 
                 padding=1,
                 output_padding=0):
        super(DeconvBlock, self).__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_channels, 
                                                 out_channels, 
                                                 kernel_size=kernel_size, 
                                                 stride=stride, 
                                                 padding=padding, 
                                                 output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ConvAutoEncoder(pl.LightningModule):
    def __init__(self, kernel_size=3, padding=1, latent_dim = 8):
        super(ConvAutoEncoder, self).__init__()
        self.training_step_outputs = []
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
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
    
class SparseAutoencoder(AutoEncoder):
    def __init__(self, kernel_size=3, padding=1, latent_dim=8):
        super().__init__(kernel_size, padding, latent_dim)