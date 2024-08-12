from typing import List
import internals_2d
from symengine import symbols, function_symbol

class DataGenerator: 
    """
    Only generates the f function for use in active learning
    """

    def __init__(self, data_points: int = 25, dimension: int = 2, grid_size: int =100, truncation_order = 20, x_save_path = None) -> None:
        
        self.data_points = data_points
        self.dimension = dimension
        self.grid_size = grid_size
        self.truncation_order = truncation_order

        self.x_save_path = x_save_path
    
        if not self.x_save_path:
            self.x_save_path = f"../../data/data_f/{self.data_points}_f_x.pt"
    

    def generate(self) -> None:
        internals_2d.save_data_in_parallel(
            self.data_points, 
            self.dimension,
            self.grid_size,
            self.truncation_order,
            self.x_save_path, 
        )



if __name__ == '__main__':
    DataGenerator(data_points=100,grid_size=32).generate()