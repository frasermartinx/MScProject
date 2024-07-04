import torch 
import torch.nn as nn
from spectralconv import SpectralConv2D



        

class FNO2D(nn.Module):
    def __init__(
            self,
            in_ch = 1,
            out_ch = 1,
            modes = 20,
            n_layers = 4,
            uplift_dim = 256,
            proj_dim = 256,
            non_linearity = "relu",
            pos_embedding_flag = True,
            grid_boundaries = [[0,1],[0,1]]):
        
        super(FNO2D, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes = modes
        self.n_layers = n_layers
        self.uplift_dim = uplift_dim
        self.proj_dim = proj_dim
        self.non_linearity = non_linearity
        self.pos_embedding_flag = pos_embedding_flag
        self.grid_boundaries = grid_boundaries


    
        def forward(self,x):
            pass
            

x = torch.rand(32,1,11,11)


