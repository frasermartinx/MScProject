import torch.nn as nn
import torch



class SpectralConv2D(nn.Module):
    def __init__(self,in_ch,out_ch,modes) -> None:
        super(SpectralConv2D,self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes = modes
        self.scale = (1 / (self.in_ch * self.out_ch))
        #this is the R matrix as per the paper which we parameterize directly in fourier space 
        #cfloat makes it complex
        self.R = nn.Parameter(
            self.scale * torch.rand(self.in_ch, self.out_ch, self.modes,self.modes,dtype=torch.cfloat))

    def fourier_tensor_mult(self,input, weights):
        # input has shape (batch,  in_ch, x, y )
        # R has shape (in_Ch,out_ch,x,y)
        #returns shape (b,out_ch,x,y)
        return torch.einsum("bixy, ioxy -> boxy", input, weights)

    def forward(self,x):
        '''
        x has shape (b,in_ch,x,y)
        we truncate the fourier series of x at some number of modes self.modes
        '''
        batch_size = x.shape[0]
        height = x.shape[-2]
        width = x.shape[-1]
        
        #need specific height and width size to work
        assert height >= self.modes, f"Must have x.shape[-2] ({height})>=modes ({self.modes})"
        assert width//2 + 1 >= self.modes, f"Must have x.shape[-1]//2 + 1 ({width//2+1}) >= modes ({self.modes})"


        #FT - only has to store certain values due to being symmetric so size is reduced
        x_ft = torch.fft.rfft2(x)
        #we want the output to be the same shape as the input, so take this shape so it will work, and make it complex valued since still in fourier domain
        out_ft = torch.zeros(batch_size,self.out_ch, x.shape[-2], x.shape[-1]//2 + 1, device = x.device, dtype=torch.cfloat)
        out_ft[:,:,:self.modes,:self.modes] =self.fourier_tensor_mult(x_ft[:,:,:self.modes,:self.modes],self.R)
        out = torch.fft.irfft2(out_ft)
        return out





