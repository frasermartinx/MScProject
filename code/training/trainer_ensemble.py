import torch
import torch.nn as nn


#add input normalisation?

#pass a model list instead of a single model
class Trainer_Ensemble:
    def __init__(
        self,
        model_list,
        n_epochs,
        device=None,
        callbacks=None,
        log_test_interval=1,
        log_output=False,
        verbose=False,
    ):
        self.n_epochs = n_epochs
        self.device = device
        self.model_list = [model.to(self.device) for model in model_list]
        self.callbacks = callbacks
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose


        
    def train(self,
              train_loader_list,
              test_loaders_list,
              optimizer_list,
              scheduler_list, 
              regularizer, 
              training_loss,
              eval_losses
              ):
        for epoch in range(self.n_epochs):
            for i, model in enumerate(self.model_list):
                model.train()
                train_err = 0.0
                for data, target in train_loader_list[i]:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer_list[i].zero_grad()
                    output = model(data)
                    loss = training_loss(*output, target)
                    if regularizer is not None:
                        loss = regularizer(loss, model)
                    loss.backward()
                    optimizer_list[i].step()
                    train_err += loss.item()
                
                
                train_err /= len(train_loader_list[i])
                if epoch % self.log_test_interval == 0:
                    if self.verbose:
                        print(
                            "Model {} - Train Epoch: {} \tLoss: {:.6f}".format(
                                i,
                                epoch,
                                train_err,
                            )
                        )
                if epoch % self.log_test_interval == 0:
                    test_err = self.evaluate(model,test_loaders_list[i], eval_losses)
                if self.callbacks is not None:
                    for callback in self.callbacks:
                        callback(model,train_loss = train_err,test_loss = test_err, epoch = epoch,model_num = i)
                if scheduler_list[i] is not None:
                    scheduler_list[i].step()
        
    def evaluate(self,model,test_loader, eval_losses):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += eval_losses(*output, target).item()
        test_loss /= len(test_loader)
        if self.verbose:
            print("Test set: Average loss: {:.4f}\n".format(test_loss))
        return test_loss
            
    
                 