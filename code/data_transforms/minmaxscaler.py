import torch

class MinMaxScaler:
    def __init__(self,eps = 1e-8):
        self.min = None
        self.max = None
        self.eps = eps

    def fit(self, data):
        # Compute min and max for each channel
        self.min = torch.min(data,dim=0, keepdim=True)[0]
        self.max = torch.max(data,dim=0, keepdim=True)[0]

    def transform(self, data):
        return (data - self.min) / (self.max - self.min + self.eps)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * (self.max - self.min+ self.eps) + self.min