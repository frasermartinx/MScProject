import torch
import torch.nn as nn


#add input normalisation?

class Trainer:
    def __init__(
        self,
        model,
        n_epochs,
        device=None,
        callbacks=None,
        log_test_interval=1,
        log_output=False,
        verbose=False,
    ):
        self.n_epochs = n_epochs
        self.device = device
        self.model = model.to(self.device)
        self.callbacks = callbacks
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        
    
    def train(self,
              train_loader,
              test_loaders,
              optimizer,
              scheduler, 
              regularizer, 
              training_loss,
              eval_losses
              ):
        for epoch in range(self.n_epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = training_loss(*output, target)
                if regularizer is not None:
                    loss = regularizer(loss, self.model)
                loss.backward()
                optimizer.step()
                if batch_idx % self.log_test_interval == 0:
                    if self.verbose:
                        print(
                            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                                epoch,
                                batch_idx * len(data),
                                len(train_loader.dataset),
                                100.0 * batch_idx / len(train_loader),
                                loss.item(),
                            )
                        )
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self.model)
            if epoch % self.log_test_interval == 0:
                self.test(test_loaders, eval_losses)
            if scheduler is not None:
                scheduler.step()
        
    def evaluate(self, test_loader, eval_losses):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += eval_losses(*output, target).item()
        test_loss /= len(test_loader.dataset)
        if self.verbose:
            print("\nTest set: Average loss: {:.4f}".format(test_loss))
        return test_loss
            
    
                 