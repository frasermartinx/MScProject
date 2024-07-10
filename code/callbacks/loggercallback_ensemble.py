import torch
import csv
import os

class LoggerCallbackEnsemble:
    def __init__(self,save_dir,n_models):
        self.save_dir = save_dir
        # Create the directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(n_models):
            dir = os.path.join(self.save_dir, f"loss_{i+1}.csv")
            # Initialize the CSV file with headers
            with open(dir, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])
        

    def __call__(self, model,train_loss, test_loss,epoch,model_num):
        # Append the new losses to the CSV file
        with open(os.path.join(self.save_dir, f"loss_{model_num+1}.csv"), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, test_loss])