import torch

class UnitGaussianNormalizer:
    def __init__(self, data,eps=0.00001):
        self.mean = torch.mean(data,dim = 0).unsqueeze(0)
        self.std = torch.std(data,dim = 0).unsqueeze(0)
        self.eps = eps
    
    def normalize(self, data):
        normalized_data = (data - self.mean) / (self.std + self.eps)
        return normalized_data

    def denormalize(self, data):
        denormalized_data = (data * (self.std+self.eps)) + self.mean
        return denormalized_data
