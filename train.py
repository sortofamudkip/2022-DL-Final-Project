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


def train(model, criterion, optimizer, scheduler, train_loader, device, num_epochs=25):
    wandb.watch(model, log_freq=100)

    running_loss = 0.0
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_images = batch[0].to(device)
            input_labels = (batch[1]).type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(input_images)
                loss = criterion(outputs, input_labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * input_images.size(0)
            epoch_loss = running_loss / len(train_loader)

            wandb.log(
                {"train_loss": running_loss, "epoch": epoch, "epoch_loss": epoch_loss}
            )

            scheduler.step()


def main():
    parser = argparse.ArgumentParser(description='Train a model with given parameters and save its to some path')
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"),
        help="Path to the stored raw data. Downloads the data if it cannot be found.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--tags", nargs='+', help="List of tags to find your results in Wandb"
    )
    parser.add_argument(
        "--output_path",
        help="Default path for the training model including a name such as data/fancy_model.pt",
    )
    parser.add_argument(
        "--model",
        choices=architecture.models.keys(),
        required=True,
        help="Architecture to train. Check architecture.py",
    )
    args = parser.parse_args()

    # Setup Wandb with the arguments from ArgumentParser 
    wandb.init(
        project=WANDB_PROJECT_NAME,
        tags=args.tags
    )
    wandb.config.update({'phase': 'train'})
    wandb.config.update(args)

    # Setup all the things required for training
    criterion = nn.CrossEntropyLoss()
    device = get_device()
    train_loader, _ = load_data(args.data_path, batch_size=args.batch_size)
    model = architecture.models[args.model]() # Lazily setup the model
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        scheduler=scheduler,
        device=device,
    )

    torch.save(model, args.output_path)

if __name__ == "__main__":
    main()