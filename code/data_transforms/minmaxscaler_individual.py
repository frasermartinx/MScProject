import torch

def min_max_scale(images):
    """
    Perform min-max scaling on each image independently.

    Parameters:
    images (numpy.ndarray): Input array of shape (b, x, y) where b is the batch size and x, y are spatial dimensions.

    Returns:
    numpy.ndarray: Scaled images of the same shape.
    """
    # Initialize the output array with the same shape as the input
    scaled_images = torch.zeros_like(images)
    
    # Iterate over each image in the batch
    for i in range(images.shape[0]):
        image = images[i]
        min_val = torch.min(image)
        max_val = torch.max(image)
        
        # Perform min-max scaling
        scaled_images[i] = (image - min_val) / (max_val - min_val) 
    
    return scaled_images