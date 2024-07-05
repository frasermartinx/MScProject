import torch.nn as nn
import torch



class SpectralConv2D(nn.Module):
    '''
    A block that performs the FFT, multiplication and inverse FFT as described in "Fourier Neural Operator for Parametric Partial Differential Equations (Li et al., 2021)

    Parameters 
    ----------
    in_ch: Int
        The input channels for the layer
    out_ch: Int
        The output channels for the layer
    n_modes: List, int
        Either a list containing two mode numbers, or one integer to be used for both
    '''
    def __init__(self,n_modes,in_ch,out_ch):
        super().__init__()
        
        if isinstance(n_modes, int):
            self.n_modes1 = self.n_modes2 = n_modes
        elif isinstance(n_modes, list) and len(n_modes) == 2 and all(isinstance(m, int) for m in n_modes):
            self.n_modes1, self.n_modes2 = n_modes
        else:
            raise ValueError("n_modes must be either an integer or a list containing exactly 2 integers")
        
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.scale = (1 / (self.in_ch * self.out_ch))
        #this is the R matrix as per the paper which we parameterize directly in fourier space 
        #cfloat makes it complex
        self.R = nn.Parameter(
            self.scale * torch.rand(self.in_ch, self.out_ch, self.n_modes1,self.n_modes2,dtype=torch.cfloat))

    def fourier_tensor_mult(self,input, weights):
        # input has shape (batch,  in_ch, x, y )
        # R has shape (in_Ch,out_ch,x,y)
        #returns shape (b,out_ch,x,y)
        return torch.einsum("bixy, ioxy -> boxy", input, weights)

    def forward(self,x):
        '''
        x has shape (b,in_ch,x,y)
        we truncate the fourier series of x at some number of n_modes self.n_modes
        '''
        batch_size = x.shape[0]
        height = x.shape[-2]
        width = x.shape[-1]
        
        #need specific height and width size to work
        assert height >= self.n_modes1, f"Must have x.shape[-2] ({height})>=n_modes_1 ({self.n_modes1})"
        assert width//2 + 1 >= self.n_modes2, f"Must have x.shape[-1]//2 + 1 ({width//2+1}) >= n_modes_2 ({self.n_modes2})"


        #FT - only has to store certain values due to being symmetric so size is reduced
        x_ft = torch.fft.rfft2(x)
        #we want the output to be the same shape as the input, so take this shape so it will work, and make it complex valued since still in fourier domain
        out_ft = torch.zeros(batch_size,self.out_ch, x.shape[-2], x.shape[-1]//2 + 1, device = x.device, dtype=torch.cfloat)
        out_ft[:,:,:self.n_modes1,:self.n_modes2] =self.fourier_tensor_mult(x_ft[:,:,:self.n_modes1,:self.n_modes2],self.R)
        out = torch.fft.irfft2(out_ft)
        return out
