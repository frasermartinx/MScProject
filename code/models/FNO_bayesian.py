import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers.mlp2d import MLP2D
from layers.fnoblock import FNOBlock2D

class FNO2D_Bayesian(nn.Module):
    def __init__(
            self,
            n_modes,
            hidden_channels,
            in_ch = 3,
            out_ch = 1,
            uplift_channels = 256,
            projection_channels = 256,
            n_layers = 4,
            non_linearity = F.gelu
            ):
        
        super().__init__()
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.uplift_channels = uplift_channels
        self.projection_channels = projection_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        # P - lifting MLP
        self.p = MLP2D(
            in_channels=self.in_ch,
            out_channels=self.hidden_channels,
            hidden_channels=self.uplift_channels,
            n_layers = 2)

        # Q - projection MLP - means
        self.q_mean = MLP2D(
            in_channels=self.hidden_channels,
            out_channels=self.out_ch,
            hidden_channels=self.projection_channels,
            n_layers = 2,
            non_linearity=non_linearity)
        
        self.q_cov = MLP2D(
            in_channels=self.hidden_channels,
            out_channels=self.out_ch,
            hidden_channels=self.projection_channels,
            n_layers = 2,
            non_linearity=non_linearity)

        
        #fourier blocks
        self.fcs = nn.ModuleList([
            FNOBlock2D(
                n_modes = self.n_modes,
                in_ch = self.hidden_channels,
                out_ch=self.hidden_channels,
                non_linearity=non_linearity
                ) for _ in range(self.n_layers)
        ])

    def forward(self,x ,**kwargs):
        
        #uplift
        x = self.p(x)
        #blocks
        for i in range(self.n_layers):
            x = self.fcs[i](x)
        mu = self.q_mean(x)
        cov = self.q_cov(x)
        return mu, cov

            

