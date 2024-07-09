import torch
import csv
import os

class LoggerCallback:
    def __init__(self,save_dir):
        self.save_dir = os.path.join(save_dir, f"loss.csv")
        # Create the directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Initialize the CSV file with headers
        with open(self.save_dir, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])

    def __call__(self, model,train_loss, test_loss,epoch):
        # Append the new losses to the CSV file
        with open(self.save_dir, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, test_loss])