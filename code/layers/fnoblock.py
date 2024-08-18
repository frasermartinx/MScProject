import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.spectralconv import SpectralConv2D
from neuralop.layers.spectral_convolution import SpectralConv
from layers.mlp2d import MLP2D

class FNOBlock2D(nn.Module):
    def __init__(
        self,
        n_modes,
        in_ch,
        out_ch,
        non_linearity = F.gelu,
        normalization = False
        ):
        super().__init__()
        self.n_modes = n_modes
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.non_linearity= non_linearity
        self.normalization = normalization

        self.W = nn.Conv2d(in_channels=self.in_ch, out_channels= self.out_ch, kernel_size=1)
        #self.mlp = MLP2D(in_channels=self.out_ch, out_channels=self.out_ch, hidden_channels=self.out_ch)
        #self.spectral = SpectralConv2D(n_modes = self.n_modes,in_ch=self.in_ch, out_ch=self.out_ch)
        self.spectral = SpectralConv(in_channels=self.in_ch, out_channels=self.out_ch, n_modes=self.n_modes)
        if normalization:
            self.bn = nn.BatchNorm2d(self.out_ch)

    def forward(self,x):
        h = self.W(x)
        out = h + self.spectral(x)
        if self.normalization:
            out = self.bn(out)
        return self.non_linearity(out)
