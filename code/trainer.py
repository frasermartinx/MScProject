import torch
import torch.nn as nn

class Trainer(nn.Module):
    def __init__(
        self,
        model,
        n_epochs,
        wandb_log=True,
        device=None,
        amp_autocast=False,
        data_processor=None,
        callbacks=None,
        log_test_interval=1,
        log_output=False,
        use_distributed=False,
        verbose=False,
    ):
        super().__init__()