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
import sklearn.metrics

# Classify random imagee
# Call Imbalanc
# TODO: New scores such F1, AUC, Recall etc


def test(model, test_loader, device):
    wandb.watch(model, log_freq=100)

    with torch.no_grad():
        for input_images, output_labels in test_loader:

            input_images.to(device)
            output_labels.type(torch.LongTensor).to(device)

            predicted_outputs = model(input_images)

            _, predicted = torch.max(predicted_outputs, 1)
            total += output_labels.size(0)
            running_accuracy += (predicted == output_labels).sum().item()

            f1 = sklearn.metrics.f1_score(output_labels, predicted)
            recall = sklearn.metrics.recall_score(output_labels, predicted)
            precision = sklearn.metrics.precision_score(output_labels, predicted)
            auc = sklearn.metrics.roc_auc_score(output_labels, predicted)

            wandb.log(
                {
                    "accuracy": running_accuracy,
                    "f1": f1,
                    "recall": recall,
                    "precision": precision,
                    "auc": auc,
                    "conf_mat": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=output_labels,
                        preds=predicted,
                        class_names=[0, 1],
                    ),
                }
            )


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
    input_transforms, model_klass = architecture.models[args.model]
    model = model_klass()
    _, test_loader = load_data(
        args.data_path, transforms=input_transforms, download=args.download_data, batch_size=args.batch_size
    )
    model.load(torch.load(args.model_path))
    model.eval()

    test(model, test_loader)


if __name__ == "__main__":
    main()
