import torch

class PositionalEmbedding2D():
    '''
    This block adds positional embeddings to the input

    Parameters
    ----------
    Grid boundaries: List 
        Contains the grid boundaries for each dimension [[,],[,]]
    '''
    def __init__(self,grid_boundaries):
        self.grid_boundaries_x = grid_boundaries[0]
        self.grid_boundaries_y = grid_boundaries[1]

    def __call__(self,input):
        #input has shape (b,c,x,y)
        batch_size = input.shape[0]
        x_range = torch.linspace(self.grid_boundaries_x[0], self.grid_boundaries_x[1], input.shape[-2])
        y_range = torch.linspace(self.grid_boundaries_y[0], self.grid_boundaries_y[1], input.shape[-1])
        x_grid, y_grid = torch.meshgrid(x_range,y_range)
        grid = torch.cat([x_grid.unsqueeze(0), y_grid.unsqueeze(0)],dim = 0).unsqueeze(0).repeat(batch_size,1,1,1)
        return grid.to(input.device)
