import torch

def min_max_scale(inputs, outputs):
    """
    Perform min-max scaling on each image independently.

    Parameters:
    images (numpy.ndarray): Input array of shape (b, x, y) where b is the batch size and x, y are spatial dimensions.

    Returns:
    numpy.ndarray: Scaled images of the same shape.
    """
    # Initialize the output array with the same shape as the input
    scaled_inputs = torch.zeros_like(inputs)
    scaled_outputs = torch.zeros_like(outputs)
    
    # Iterate over each image in the batch
    for i in range(inputs.shape[0]):
        input = inputs[i]
        output = outputs[i]
        min_val = torch.min(input)
        max_val = torch.max(input)
        
        # Perform min-max scaling
        scaled_inputs[i] = (input - min_val) / (max_val - min_val) 
        scaled_outputs[i] = (output - min_val) / (max_val - min_val) 
    
    return scaled_inputs, scaled_outputs