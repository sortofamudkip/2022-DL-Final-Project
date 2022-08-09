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

# TODO: Validation?, take X % as validation
# TODO: Checkpointing


def train(model, criterion, optimizer, scheduler, train_loader, device, num_epochs=25):
    wandb.watch(model, log_freq=100)

    running_loss = 0.0
    model.train()

    for epoch in range(num_epochs):
        for input_images, output_labels in train_loader:
            input_images.to(device)
            output_labels.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(input_images)
                loss = criterion(outputs, output_labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * input_images.size(0)
            epoch_loss = running_loss / len(train_loader)

            wandb.log(
                {"train_loss": running_loss, "epoch": epoch, "epoch_loss": epoch_loss}
            )

            scheduler.step()


def main():
    parser = argparse.ArgumentParser(
        description="Train a model with given parameters and save its to some path"
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"),
        help="Path to the stored raw data.",
    )
    parser.add_argument(
        "--download_data",
        type=bool,
        default=False,
        help="Force downloading the data into the data_path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--tags", nargs="+", help="List of tags to find your results in Wandb"
    )
    parser.add_argument(
        "--output_path",
        help="Default path for the training model including a name such as data/fancy_model.pt",
        required=True
    )
    parser.add_argument(
        "--model",
        choices=architecture.models.keys(),
        required=True,
        help="Architecture to train. Check architecture.py",
    )
    parser.add_argument(
        "--first_n_rows",
        type=int,
        default=0,
        help="Only use the first N rows of the dataset (for debugging)",
    )
    args = parser.parse_args()

    # Setup Wandb with the arguments from ArgumentParser
    wandb.init(project=WANDB_PROJECT_NAME, tags=args.tags)
    wandb.config.update({"phase": "train"})
    wandb.config.update(args)

    # Setup all the things required for training
    criterion = nn.CrossEntropyLoss()
    device = get_device()
    # Lazily setup the model
    input_transforms, model_klass = architecture.models[args.model]
    model = model_klass()
    # Load the training data and apply image transformation and the badge size
    train_loader, _ = load_data(
        args.data_path,
        transforms=input_transforms,
        download=args.download_data,
        batch_size=args.batch_size,
        first_n_rows=args.first_n_rows
    )
    # print("number of rows:", len(train_loader), args.first_n_rows); return
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"training model {args.model}...")

    train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        scheduler=scheduler,
        device=device,
        num_epochs=5
    )
    print(f"saving model {args.model} to {args.output_path}...")

    torch.save(model, args.output_path)


if __name__ == "__main__":
    main()
