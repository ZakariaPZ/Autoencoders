from torch import nn
import torch
import typing as th


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
        self.activation = activation()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 activation : th.Callable[[torch.Tensor], torch.Tensor] = nn.ReLU, 
                 kernel_size=3, 
                 stride=2, 
                 padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
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