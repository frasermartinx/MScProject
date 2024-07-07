import torch

class LoggerCallback:
    def __init__(self):
        self.train_loss = []
        self.test_loss = []

    def __call__(self, model,train_loss, test_loss):
        self.train_loss.append(train_loss)
        self.test_loss.append(test_loss)