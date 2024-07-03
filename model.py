import torch.nn as nn
import torch



class SpectralConv2D(nn.Module):
    def __init__(self,in_ch,out_ch,modes) -> None:
        super().__init__(self)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes = modes
        self.scale = (1 / (self.in_ch * self.out_ch))
        #this is the R matrix as per the paper which we parameterize directly in fourier space 
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_ch, self.out_ch, self.modes, dtype=torch.cfloat))




    def forward(self,x):
        '''
        x has shape (b,c,x,y)
        we truncate the fourier series of x at some number of modes self.modes
        '''
        batch_size = x.shape[0]
        x = torch.fft.rfft2(x)

        out = torch.fft.rfft2()
