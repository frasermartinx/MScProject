import torch

class PositionalEmbedding2D_polar():
    '''
    This block adds positional embeddings to the input

    Parameters
    ----------
    Grid boundaries: List 
        Contains the grid boundaries for each dimension [[,],[,]]
    '''
    def __init__(self,grid_boundaries):
        self.grid_boundaries_r = grid_boundaries[0]
        self.grid_boundaries_theta = grid_boundaries[1]

    def __call__(self,input):
        #input has shape (b,c,x,y)
        batch_size = input.shape[0]
        r_range = torch.linspace(self.grid_boundaries_r[0], self.grid_boundaries_r[1], input.shape[-2])
        theta_range = torch.linspace(self.grid_boundaries_theta[0], self.grid_boundaries_theta[1], input.shape[-1])
        r_grid, theta_grid = torch.meshgrid(r_range,theta_range)
        x_grid, y_grid = r_grid * torch.cos(theta_grid), r_grid * torch.sin(theta_grid)
        grid = torch.cat([x_grid.unsqueeze(0), y_grid.unsqueeze(0)],dim = 0).unsqueeze(0).repeat(batch_size,1,1,1)
        data_grid = torch.cat([input,grid], dim = 1)
        return data_grid.to(input.device)
