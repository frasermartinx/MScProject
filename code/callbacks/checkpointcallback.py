import torch
import os

class CheckpointCallback:
    def __init__(self, save_dir, save_freq):
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.epoch_counter = 0

    def __call__(self, model):
        self.epoch_counter += 1
        if self.epoch_counter % self.save_freq == 0:
            checkpoint_path = os.path.join(self.save_dir, f"model_epoch_{self.epoch_counter}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at epoch {self.epoch_counter}.")
