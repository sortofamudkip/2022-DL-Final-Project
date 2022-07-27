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
from sklearn.metrics import confusion_matrix

# TODO: New scores such F1, AUC, Recall etc
def evaluate_model(model, val_loader):
    """
    Method to check the model preformance per different classification metrics
    """
    was_training = model.training
    model.eval()
    y_pred = []
    y_true = []
    CM=0
    with torch.no_grad():
        for i, s in enumerate(val_loader):
            inputs = s[0].to(device)
            
            #casting labels to long as float doesn't work to train the resnet
            labels = (s[1]).type(torch.LongTensor)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.data.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())
    cf_matrix = confusion_matrix(y_true, y_pred)   
    
    return cf_matrix

def test(model, test_loader, device):
    wandb.watch(model, log_freq=100)

    with torch.no_grad():
        for batch in test_loader:
            input_images, ouput_labels = batch

            input_images.to(device)
            ouput_labels.type(torch.LongTensor).to(device)

            predicted_outputs = model(input_images)

            _, predicted = torch.max(predicted_outputs, 1)
            total += ouput_labels.size(0)
            running_accuracy += (predicted == ouput_labels).sum().item()

            wandb.log({"accuracy": running_accuracy})


def main():
    parser = argparse.ArgumentParser(description="Test an import model")
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"),
        help="Path to the stored raw data. Downloads the data if it cannot be found.",
    )
    parser.add_argument(
        "--download_data",
        type=bool,
        default=False,
        help="Force downloading the data into the data_path",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Test batch size")
    parser.add_argument("--model_path", require=True, help="Path to the trained model")
    parser.add_argument(
        "--tags", nargs="+", help="List of tags to find your results in Wandb"
    )
    parser.add_argument(
        "--model",
        choices=architecture.models.keys(),
        required=True,
        help="Architecture to train. Check architecture.py",
    )
    args = parser.parse_args()

    wandb.init(project=WANDB_PROJECT_NAME, tags=args.tags)
    wandb.config.update({"phase": "test"})
    wandb.config.update(args)

    device = get_device()
    _, test_loader = load_data(
        args.data_path, download=args.download_data, batch_size=args.batch_size
    )
    model = architecture.models[args.model]()
    model.load(torch.load(args.model_path))
    model.eval()

    test(model, test_loader)
