from pkg_resources import require


import argparse
from pickletools import optimize
import architecture
from data_loading import load_data
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim import lr_scheduler
from utils import WANDB_PROJECT_NAME, get_device

def test(model, test_loader, device):
    pass

def main():
    parser = argparse.ArgumentParser(description='Test an import model')
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"),
        help="Path to the stored raw data. Downloads the data if it cannot be found.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Test batch size"
    )
    parser.add_argument(
        "--model_path", require=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--model",
        choices=architecture.models.keys(),
        required=True,
        help="Architecture to train. Check architecture.py",
    )
    args = parser.parse_args()

    wandb.init(
        project=WANDB_PROJECT_NAME,
        tags=args.tags
    )
    wandb.config.update({'phase': 'test'})
    wandb.config.update(args)

    device = get_device()
    _, test_loader = load_data(args.data_path, batch_size=args.batch_size)
    model = architecture.models[args.model]()
    model.load(torch.load(args.model_path))
    model.eval()

    test(model, test_loader)