import torch.nn.functional as F
from torch import nn
import torch
import typing as th
from blocks import ConvBlock, DeconvBlock, Reshape

# TODO: add docstrings, type-hinting, VAE

class Encoder(nn.Module):
    def __init__(self,
                 latent_dim = 2,
                 input_dim = 1) -> None:
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(input_dim, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            nn.Flatten(),
        )        

        self.mu = nn.Linear(64 * 4 * 4, latent_dim)
        self.log_var = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):

        z = self.encoder(x)
        mu = self.mu(z)
        log_var = self.log_var(z)
        return mu, log_var
    

class Decoder(nn.Module):
    def __init__(self,
                 latent_dim = 2,
                 output_dim = 1) -> None:
        super(Decoder, self).__init__()

        self.from_latent = nn.Linear(latent_dim, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            Reshape(-1, 64, 4, 4),
            DeconvBlock(64, 32, output_padding=0),
            DeconvBlock(32, 16, output_padding=1),
            DeconvBlock(16, output_dim, output_padding=1, 
                        activation=nn.Tanh)
        )     

    def forward(self, x):
        x = self.from_latent(x)
        x = self.decoder(x)
        return x 
    

class BetaVAE(nn.Module):
    def __init__(self, 
                 latent_dim=2,
                 beta=100.0):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mu = torch.randn(latent_dim)
        self.log_var = torch.randn(latent_dim)
        self.beta = beta

    def reconstruction_loss(self, x, xhat):
        return torch.sum(F.mse_loss(xhat, x))

    def KLD(self):
        return -0.5 * torch.sum(1 + self.log_var - torch.square(self.mu) - torch.exp(self.log_var))

    def VAE_loss(self, x, xhat):
        return self.KLD() + self.beta * self.reconstruction_loss(x, xhat)

    def sample_latent(self):
        eps = torch.randn_like(self.log_var)
        return self.mu + torch.exp(0.5 * self.log_var)*eps
        
    def forward(self, x):
        self.mu, self.log_var = self.encoder(x)
        z = self.sample_latent()
        xtilde = self.decoder(z)

        return xtilde