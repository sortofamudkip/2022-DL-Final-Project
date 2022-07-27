WANDB_PROJECT_NAME = "histopathologic-cancer-dDetection-dl2022"

import torch

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")