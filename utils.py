import torch
import os
import time

WANDB_PROJECT_NAME = "histopathologic-cancer-dDetection-dl2022"
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
KAGGLE_DATASET = "histopathologic-cancer-detection"


def timestamp():
    return int(time.time_ns() / 1000)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
