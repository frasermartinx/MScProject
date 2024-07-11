import torch
import os

class CheckpointCallbackEnsemble:
    def __init__(self, save_dir, save_freq):
        self.save_dir = save_dir
        self.save_freq = save_freq
        # Create the directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __call__(self, model,train_loss, test_loss,epoch, model_num):
        if epoch % self.save_freq == 0:
            checkpoint_path = os.path.join(self.save_dir, f"model_state_{model_num+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at epoch {epoch}.")
